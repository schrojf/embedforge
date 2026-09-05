"""Model registry.

Every model the server can serve is declared here with its `ModelInfo` and a factory.
One model is loaded per process, chosen by `EMBEDFORGE_MODEL_ID`; the registry is what
`embedforge model list` reads and what documents the trade-offs between them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from embedforge.config import Settings
from embedforge.engine.base import EmbeddingBackend, ModelInfo
from embedforge.errors import InvalidRequestError

BackendFactory = Callable[[Settings, ModelInfo], EmbeddingBackend]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A model that can be selected at startup."""

    info: ModelInfo
    factory: BackendFactory


_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    _REGISTRY[spec.info.id] = spec
    return spec


def list_specs() -> list[ModelSpec]:
    return sorted(_REGISTRY.values(), key=lambda spec: spec.info.id)


def known_ids() -> list[str]:
    return sorted(_REGISTRY)


def get_spec(model_id: str) -> ModelSpec:
    try:
        return _REGISTRY[model_id]
    except KeyError:
        raise InvalidRequestError(
            f"Unknown model {model_id!r}. Known models: {', '.join(known_ids())}."
        ) from None


def create_backend(settings: Settings) -> EmbeddingBackend:
    """Instantiate the backend for the configured model. Does not load it."""
    spec = get_spec(settings.model_id)
    return spec.factory(settings, spec.info)


# ---- Built-in models ----

from embedforge.engine.dev_hash import DEV_HASH_INFO, HashEmbeddingBackend  # noqa: E402

register(
    ModelSpec(
        info=DEV_HASH_INFO,
        factory=lambda settings, info: HashEmbeddingBackend(
            info, normalize=settings.normalize_embeddings
        ),
    )
)
