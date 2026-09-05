# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# onnxruntime and tokenizers ship without type stubs; the relaxation stops here
# rather than being turned off for the whole project.
"""ONNX Runtime backend.

ONNX is the runtime because it is portable: one exported graph runs on CPU everywhere,
and on GPU by swapping the execution provider, with no framework in the serving path.

Two properties matter for the way this server handles concurrency: `session.run`
releases the GIL, and so does the Rust tokenizer, so the engine's worker threads do
genuinely parallel work. See docs/performance.md.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from embedforge.config import Settings
from embedforge.engine.base import EmbeddingBackend, EmbedInput, ModelInfo, TaskType
from embedforge.logging import get_logger

if TYPE_CHECKING:
    from tokenizers import Tokenizer

log = get_logger(__name__)


class Pooling(StrEnum):
    """How token vectors are reduced to one vector per input."""

    MEAN = "mean"
    """Average over real tokens. What the E5 family was trained with."""

    CLS = "cls"
    """The first token's vector. What BGE and GTE were trained with."""

    LAST_TOKEN = "last_token"
    """The final real token's vector. What decoder-style models such as Qwen3 use.

    "Final real" matters: with right padding the last row of the tensor is a pad
    token, so the index has to come from the attention mask.
    """


@dataclass(frozen=True, slots=True)
class OnnxModelConfig:
    """Everything needed to fetch and run one exported model.

    Using the wrong pooling or dropping the prefixes silently produces vectors that
    look fine and retrieve badly, so these travel with the model rather than being
    guessed at load time.
    """

    repo_id: str
    revision: str
    """Pinned commit sha. Never a branch: downloads must be reproducible."""

    onnx_file: str
    pooling: Pooling
    max_seq_length: int
    tokenizer_file: str = "tokenizer.json"
    extra_files: tuple[str, ...] = ()
    """Companion files, e.g. external weights for graphs over 2 GB."""

    query_prefix: str = ""
    document_prefix: str = ""
    """Text prefixes the model was trained with. Empty when it uses none."""

    query_task_id: int | None = None
    document_task_id: int | None = None
    """Index into the model's task adapters, fed as a `task_id` input.

    Some models select the task with a LoRA adapter instead of a text prefix
    (jina-embeddings-v3 does). Same purpose, different mechanism.
    """

    export_from: str | None = None
    """Source repository to export locally, for models that publish no ONNX build."""

    @property
    def needs_export(self) -> bool:
        return self.export_from is not None

    def prefix_for(self, task: TaskType) -> str:
        return self.query_prefix if task is TaskType.QUERY else self.document_prefix

    def task_id_for(self, task: TaskType) -> int | None:
        return self.query_task_id if task is TaskType.QUERY else self.document_task_id

    @property
    def distinguishes_tasks(self) -> bool:
        """Whether queries and documents are encoded differently at all.

        True for asymmetric models, by whichever mechanism they use.
        """
        return (
            self.query_prefix != self.document_prefix or self.query_task_id != self.document_task_id
        )

    @property
    def files(self) -> tuple[str, ...]:
        """Every file to fetch, as repository-relative paths.

        The layout is mirrored locally: external weight files are referenced from
        inside the graph by name and must stay beside it.
        """
        return (self.onnx_file, *self.extra_files, self.tokenizer_file)


def model_directory(settings: Settings, model_id: str) -> Path:
    """Where one model's files live. Keyed by model id, so precision variants of the
    same repository do not collide."""
    return settings.model_dir / model_id


def pool(hidden_states: np.ndarray, attention_mask: np.ndarray, pooling: Pooling) -> np.ndarray:
    """Reduce `(batch, tokens, hidden)` to `(batch, hidden)`."""
    if pooling is Pooling.CLS:
        return hidden_states[:, 0].astype(np.float32, copy=False)
    if pooling is Pooling.LAST_TOKEN:
        # Index the last unmasked position per row. Taking hidden_states[:, -1] would
        # read padding for every sequence shorter than the longest in the batch, which
        # silently produces garbage for exactly the short inputs.
        lengths = attention_mask.sum(axis=1) - 1
        lengths = np.maximum(lengths, 0)
        rows = np.arange(hidden_states.shape[0])
        return hidden_states[rows, lengths].astype(np.float32, copy=False)
    weights = attention_mask[..., None].astype(np.float32)
    summed = (hidden_states.astype(np.float32, copy=False) * weights).sum(axis=1)
    # Padding tokens must not dilute the average, and an all-padding row must not divide
    # by zero.
    counts = np.maximum(weights.sum(axis=1), 1e-9)
    return summed / counts


_CACHE_PREFIX = "past_key_values"

_ONNX_DTYPES: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(bfloat16)": "float32",
}


def cache_specs(session: Any) -> list[tuple[str, tuple[int | str, ...], np.dtype[Any]]]:
    """Describe the empty key/value cache a decoder-style export expects."""
    specs: list[tuple[str, tuple[int | str, ...], np.dtype[Any]]] = []
    for node in session.get_inputs():
        if node.name.startswith(_CACHE_PREFIX):
            dtype = np.dtype(_ONNX_DTYPES.get(node.type, "float32"))
            specs.append((node.name, tuple(node.shape), dtype))
    return specs


def _resolve_padding(tokenizer: "Tokenizer") -> tuple[int, str]:
    """Find the tokenizer's pad token, which differs between model families."""
    for token in ("<pad>", "[PAD]", "<|endoftext|>"):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return token_id, token
    return 0, tokenizer.id_to_token(0) or "<pad>"


class OnnxTextBackend(EmbeddingBackend):
    """Runs an exported text embedding model."""

    def __init__(self, info: ModelInfo, config: OnnxModelConfig, settings: Settings) -> None:
        super().__init__(info, normalize=settings.normalize_embeddings)
        self.config = config
        self.directory = model_directory(settings, info.id)
        self._device = settings.device
        self._threads = settings.effective_intra_op_threads()
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: frozenset[str] = frozenset()
        self._cache_specs: list[tuple[str, tuple[int | str, ...], np.dtype[Any]]] = []

    # ---- Lifecycle ----

    def _providers(self) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self._device in ("cuda", "auto") and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self._device == "cuda":
            raise RuntimeError(
                "EMBEDFORGE_DEVICE=cuda but CUDAExecutionProvider is unavailable. "
                "Install onnxruntime-gpu in place of onnxruntime."
            )
        return ["CPUExecutionProvider"]

    def load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        graph_path = self.directory / self.config.onnx_file
        tokenizer_path = self.directory / self.config.tokenizer_file
        for path in (graph_path, tokenizer_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing. Fetch the model first: "
                    f"embedforge model download {self.info.id}"
                )

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=self.config.max_seq_length)
        pad_id, pad_token = _resolve_padding(tokenizer)
        # Pad to the longest item in each batch rather than to max_seq_length: padding
        # is wasted compute, and short batches are the common case.
        tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        self._tokenizer = tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = self._threads
        # Batches are already parallel work; inter-op threads would only oversubscribe.
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = self._providers()
        self._session = ort.InferenceSession(str(graph_path), options, providers=providers)
        self._input_names = frozenset(node.name for node in self._session.get_inputs())
        self._cache_specs = cache_specs(self._session)

        log.info(
            "onnx_model_loaded",
            model=self.info.id,
            providers=providers,
            intra_op_threads=self._threads,
            inputs=sorted(name for name in self._input_names if not name.startswith(_CACHE_PREFIX)),
            cache_inputs=len(self._cache_specs),
            path=str(graph_path),
        )

    def close(self) -> None:
        self._session = None
        self._tokenizer = None

    # ---- Inference ----

    def _feeds(self, texts: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.asarray(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )
        batch, length = len(encodings), int(input_ids.shape[-1])
        candidates = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Single-segment input, so all zeros. Only some exports declare it.
            "token_type_ids": np.zeros_like(input_ids),
            # Decoder-style exports want explicit positions. Padding is on the right and
            # masked out, so a plain range is correct for every real token.
            "position_ids": np.tile(np.arange(length, dtype=np.int64), (batch, 1)),
        }
        feeds = {name: value for name, value in candidates.items() if name in self._input_names}
        # A cached decoder export declares one key/value input per layer. We do a single
        # forward pass and never generate, so every cache starts empty.
        for name, shape, dtype in self._cache_specs:
            dimensions = tuple(
                batch if index == 0 else (0 if isinstance(size, str) else size)
                for index, size in enumerate(shape)
            )
            feeds[name] = np.zeros(dimensions, dtype=dtype)
        return feeds, attention_mask

    def embed(self, inputs: Sequence[EmbedInput], task: TaskType) -> np.ndarray:
        if self._session is None or self._tokenizer is None:
            raise RuntimeError("Backend is not loaded.")
        texts = [self.config.prefix_for(task) + item.text for item in inputs]
        feeds, attention_mask = self._feeds(texts)
        task_id = self.config.task_id_for(task)
        if task_id is not None and "task_id" in self._input_names:
            # A 0-d array: the graph selects one adapter for the whole batch, which is
            # exactly why the engine never mixes document and query work in one batch.
            feeds["task_id"] = np.array(task_id, dtype=np.int64)
        hidden_states = self._session.run(None, feeds)[0]
        vectors = pool(hidden_states, attention_mask, self.config.pooling)
        return self.l2_normalize(vectors) if self.normalize else vectors
