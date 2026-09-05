"""Deterministic hash backend.

This is *not* a semantic model: vectors are derived from a hash of the input, so
identical text gives identical vectors and unrelated text gives unrelated vectors,
but "cat" and "kitten" are as far apart as any other pair.

It exists so the server is complete and runnable with no model download at all: it
backs the test suite, CI, and container smoke checks. Real ONNX models register
alongside it in `embedforge.engine.registry`.
"""

import hashlib
from collections.abc import Sequence

import numpy as np

from embedforge.engine.base import EmbeddingBackend, EmbedInput, ModelInfo, TaskType

DEV_HASH_INFO = ModelInfo(
    id="dev-hash",
    name="embedforge/dev-hash",
    dimension=384,
    max_input_tokens=1_000_000,
    symmetric=True,
    description=(
        "Deterministic hash-based pseudo-embeddings. No download, no inference cost, "
        "no semantics. For development, tests, and deployment smoke checks."
    ),
    pros=(
        "Zero setup: no model files, no ONNX Runtime, instant startup.",
        "Deterministic and fast, so tests and load checks are reproducible.",
    ),
    cons=(
        "Vectors carry no meaning; similarity between different texts is noise.",
        "Never use it to serve real retrieval traffic.",
    ),
    license="MIT",
    size_mb=0,
)


class HashEmbeddingBackend(EmbeddingBackend):
    """Backend that hashes each input into a stable pseudo-random unit vector."""

    def embed(self, inputs: Sequence[EmbedInput], task: TaskType) -> np.ndarray:
        # Symmetric model: the task makes no difference, so /embed and /query agree.
        del task
        dimension = self.info.dimension
        matrix = np.empty((len(inputs), dimension), dtype=np.float32)
        for row, item in enumerate(inputs):
            digest = hashlib.blake2b(item.text.encode("utf-8"), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            matrix[row] = rng.standard_normal(dimension, dtype=np.float32)
        return self.l2_normalize(matrix) if self.normalize else matrix
