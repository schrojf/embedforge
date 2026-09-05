"""Model registry.

Every model the server can serve is declared here with its `ModelInfo` and a factory.
One model is loaded per process, chosen by `EMBEDFORGE_MODEL_ID`; the registry is what
`embedforge model list` reads and what documents the trade-offs between them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from embedforge.config import Settings
from embedforge.engine.base import EmbeddingBackend, ModelInfo
from embedforge.errors import InvalidRequestError

if TYPE_CHECKING:
    from embedforge.engine.onnx_backend import OnnxModelConfig

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


def create_backend(settings: Settings, model_id: str | None = None) -> EmbeddingBackend:
    """Instantiate a backend. Does not load it.

    Defaults to the configured model; `model_id` overrides it for tools that work
    across models, such as `embedforge model compare`.
    """
    spec = get_spec(model_id or settings.model_id)
    return spec.factory(settings, spec.info)


def onnx_config(model_id: str) -> "OnnxModelConfig":
    """The download and runtime configuration for an ONNX model.

    Raises for models that are not ONNX-backed, such as `dev-hash`.
    """
    from embedforge.engine.catalog import ONNX_CONFIGS

    try:
        return ONNX_CONFIGS[model_id]
    except KeyError:
        get_spec(model_id)  # Raises with the list of known ids if it is not a model at all.
        raise InvalidRequestError(f"Model {model_id!r} has no files to manage.") from None


def is_onnx_model(model_id: str) -> bool:
    from embedforge.engine.catalog import ONNX_CONFIGS

    return model_id in ONNX_CONFIGS


# ---- Built-in models ----

from embedforge.engine.catalog import CATALOG, build_backend  # noqa: E402
from embedforge.engine.dev_hash import DEV_HASH_INFO, HashEmbeddingBackend  # noqa: E402

register(
    ModelSpec(
        info=DEV_HASH_INFO,
        factory=lambda settings, info: HashEmbeddingBackend(
            info, normalize=settings.normalize_embeddings
        ),
    )
)

for _info, _config in CATALOG:
    register(
        ModelSpec(
            info=_info,
            # Bind the config per entry; a late-bound closure would give every model the
            # last one in the loop.
            factory=lambda settings, info, _config=_config: build_backend(settings, info, _config),
        )
    )
