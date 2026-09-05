"""Backend interface and model description.

Everything above this module speaks in `EmbedInput` objects rather than raw strings.
Text is the only implemented modality, but the seam is where image (or audio) inputs
plug in later: add an input type, declare the modality in `ModelInfo.modalities`, and
teach the API schema to build it. Nothing in the queueing or HTTP layer changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class Modality(StrEnum):
    """Kinds of input a model can embed."""

    TEXT = "text"
    IMAGE = "image"


class TaskType(StrEnum):
    """What the caller intends the vector for.

    Asymmetric models encode a search query differently from a stored passage,
    usually via an instruction prefix. Symmetric models ignore this, which is why
    `/embed` and `/query` legitimately return identical vectors for them.
    """

    DOCUMENT = "document"
    QUERY = "query"


@dataclass(frozen=True, slots=True)
class TextInput:
    """A single piece of text to embed."""

    text: str

    modality: Modality = field(default=Modality.TEXT, init=False)

    def __len__(self) -> int:
        return len(self.text)


EmbedInput = TextInput
"""Union point for future modalities, e.g. `TextInput | ImageInput`."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Everything the API and CLI need to describe a model without loading it."""

    id: str
    """Stable identifier used by `EMBEDFORGE_MODEL_ID`."""

    name: str
    """Upstream model name, e.g. a Hugging Face repo id."""

    dimension: int
    max_input_tokens: int
    symmetric: bool
    """True when queries and documents share one representation."""

    description: str = ""
    modalities: tuple[Modality, ...] = (Modality.TEXT,)
    pros: tuple[str, ...] = ()
    cons: tuple[str, ...] = ()
    license: str = "unknown"
    size_mb: int | None = None
    revision: str | None = None

    def supports(self, modality: Modality) -> bool:
        return modality in self.modalities


class EmbeddingBackend(ABC):
    """A loaded model that turns inputs into vectors.

    Implementations are synchronous and thread-safe: `embed` is called from the
    engine's worker threads and must release the GIL for real work (ONNX Runtime and
    the Hugging Face tokenizers both do), which is what makes CPU parallelism here
    actually parallel.
    """

    def __init__(self, info: ModelInfo, *, normalize: bool = True) -> None:
        self.info = info
        self.normalize = normalize

    def load(self) -> None:
        """Prepare the model. Called once, off the event loop."""

    def close(self) -> None:
        """Release resources. Called once at shutdown."""

    @abstractmethod
    def embed(self, inputs: Sequence[EmbedInput], task: TaskType) -> np.ndarray:
        """Return a `(len(inputs), dimension)` float32 matrix, one row per input."""

    def warmup_inputs(self) -> list[EmbedInput]:
        """A tiny batch used to force lazy allocation before serving traffic."""
        return [TextInput("warmup")]

    @staticmethod
    def l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        return matrix / norms
