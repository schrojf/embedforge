"""Read-only model catalog."""

from typing import Annotated

from fastapi import APIRouter, Depends

from embedforge.api.deps import SettingsDep, require_scope
from embedforge.auth.models import Principal, Scope
from embedforge.engine import registry
from embedforge.schemas.models import ModelDescription, ModelListResponse

router = APIRouter(tags=["models"])

ModelsScope = Annotated[Principal, Depends(require_scope(Scope.MODELS_READ))]


@router.get("/models", response_model=ModelListResponse, summary="List known models")
async def list_models(settings: SettingsDep, _: ModelsScope) -> ModelListResponse:
    """Every model this build can serve, and which one is loaded right now.

    A process serves exactly one model; switching means changing
    `EMBEDFORGE_MODEL_ID` and restarting.
    """
    return ModelListResponse(
        active=settings.model_id,
        models=[
            ModelDescription.from_info(spec.info, active=spec.info.id == settings.model_id)
            for spec in registry.list_specs()
        ],
    )


@router.get("/models/{model_id}", response_model=ModelDescription, summary="Describe a model")
async def get_model(model_id: str, settings: SettingsDep, _: ModelsScope) -> ModelDescription:
    spec = registry.get_spec(model_id)
    return ModelDescription.from_info(spec.info, active=spec.info.id == settings.model_id)
