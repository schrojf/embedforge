"""Model catalog responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from embedforge.engine.base import Modality, ModelInfo


class ModelDescription(BaseModel):
    """One selectable model, with the trade-offs that matter when choosing it."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    name: str
    dimension: int
    max_input_tokens: int
    symmetric: bool
    modalities: list[Modality]
    description: str
    pros: list[str]
    cons: list[str]
    license: str
    size_mb: int | None = None
    active: bool = Field(description="True for the model this server has loaded.")

    @classmethod
    def from_info(cls, info: ModelInfo, *, active: bool) -> ModelDescription:
        return cls(
            id=info.id,
            name=info.name,
            dimension=info.dimension,
            max_input_tokens=info.max_input_tokens,
            symmetric=info.symmetric,
            modalities=list(info.modalities),
            description=info.description,
            pros=list(info.pros),
            cons=list(info.cons),
            license=info.license,
            size_mb=info.size_mb,
            active=active,
        )


class ModelListResponse(BaseModel):
    active: str
    models: list[ModelDescription]
