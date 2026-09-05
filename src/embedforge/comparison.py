"""Compare embedding models offline.

Choosing a model means answering two questions on your own data: does it rank the
right things first, and is it fast enough. This runs both, one model at a time.

Deliberately a library plus a CLI, not an API endpoint. Serving several models from
one process either pays the load cost per request or holds every model in memory,
which is exactly what the server avoids; and a "load any model" code path is one that
must never be reachable in production. Offline, it can also do things a request never
could, such as measuring load time and comparing rankings across models.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from embedforge.config import Settings
from embedforge.engine.base import TaskType, TextInput
from embedforge.engine.registry import create_backend, get_spec
from embedforge.logging import get_logger

log = get_logger(__name__)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of `left` and every row of `right`."""
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return left_norm @ right_norm.T


def _ranks(values: np.ndarray) -> np.ndarray:
    """Rank positions of `values`, averaging ranks across ties.

    Ties are not a corner case here: duplicate documents, or a symmetric model seeing
    repeated text, produce identical scores. Ranking them by argsort order would invent
    an ordering out of nothing and skew the correlation.
    """
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0
        start = end + 1
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation: 1.0 means identical ordering, -1.0 reversed."""
    if len(left) < 2:
        return float("nan")
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return float("nan")
    return float(left_centered @ right_centered / denominator)


@dataclass(frozen=True, slots=True)
class ModelRun:
    """What one model produced for the same inputs."""

    model_id: str
    name: str
    dimension: int
    symmetric: bool
    load_seconds: float
    embed_seconds: float
    """Mean wall time of one full pass over the inputs, excluding warmup."""

    item_count: int
    document_vectors: np.ndarray
    query_vectors: np.ndarray

    @property
    def ms_per_item(self) -> float:
        return (self.embed_seconds / self.item_count * 1000) if self.item_count else 0.0

    @property
    def items_per_second(self) -> float:
        return (self.item_count / self.embed_seconds) if self.embed_seconds else float("inf")

    def scores(self, query_index: int) -> np.ndarray:
        """Similarity of every document to one query."""
        return cosine_similarity(
            self.query_vectors[query_index : query_index + 1], self.document_vectors
        )[0]

    def ranking(self, query_index: int, top_k: int) -> list[tuple[int, float]]:
        """The top `top_k` documents for a query, best first."""
        scores = self.scores(query_index)
        order = np.argsort(-scores)[:top_k]
        return [(int(index), float(scores[index])) for index in order]

    def document_similarity(self) -> np.ndarray:
        return cosine_similarity(self.document_vectors, self.document_vectors)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    documents: list[str]
    queries: list[str]
    runs: list[ModelRun]
    failures: dict[str, str] = field(default_factory=dict)
    """Models that could not be loaded, mapped to why."""

    def agreement(self, left: ModelRun, right: ModelRun) -> float:
        """Mean rank correlation between two models across all queries.

        High agreement means the models would retrieve the same things, so the
        cheaper one is the better choice. Low agreement means the choice matters and
        deserves evaluation on real queries.
        """
        if not self.queries or len(self.documents) < 2:
            return float("nan")
        values = [
            spearman(left.scores(index), right.scores(index)) for index in range(len(self.queries))
        ]
        finite = [value for value in values if not np.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")


def compare_models(
    settings: Settings,
    model_ids: list[str],
    documents: list[str],
    queries: list[str] | None = None,
    *,
    rounds: int = 1,
) -> ComparisonResult:
    """Load each model in turn, embed the same inputs, and collect the results.

    Models are loaded one at a time and released before the next, so peak memory is
    one model rather than all of them.
    """
    queries = queries or []
    if not documents:
        raise ValueError("At least one document is required.")
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")

    document_inputs = [TextInput(text) for text in documents]
    query_inputs = [TextInput(text) for text in queries]

    runs: list[ModelRun] = []
    failures: dict[str, str] = {}

    for model_id in model_ids:
        try:
            spec = get_spec(model_id)
            backend = create_backend(settings, model_id)
            started = time.perf_counter()
            backend.load()
            load_seconds = time.perf_counter() - started
        except Exception as error:
            log.warning("model_load_failed", model=model_id, error=str(error))
            failures[model_id] = str(error)
            continue

        try:
            # Warm up first: the first call pays for lazy allocation, and timing it
            # would say more about the runtime than about the model.
            backend.embed(document_inputs[:1], TaskType.DOCUMENT)

            durations: list[float] = []
            document_vectors = np.empty((0, spec.info.dimension), dtype=np.float32)
            query_vectors = np.empty((0, spec.info.dimension), dtype=np.float32)
            for _ in range(rounds):
                started = time.perf_counter()
                document_vectors = backend.embed(document_inputs, TaskType.DOCUMENT)
                if query_inputs:
                    query_vectors = backend.embed(query_inputs, TaskType.QUERY)
                durations.append(time.perf_counter() - started)

            runs.append(
                ModelRun(
                    model_id=model_id,
                    name=spec.info.name,
                    dimension=spec.info.dimension,
                    symmetric=spec.info.symmetric,
                    load_seconds=load_seconds,
                    embed_seconds=float(np.mean(durations)),
                    item_count=len(documents) + len(queries),
                    document_vectors=document_vectors,
                    query_vectors=query_vectors,
                )
            )
        finally:
            backend.close()

    return ComparisonResult(documents=documents, queries=queries, runs=runs, failures=failures)
