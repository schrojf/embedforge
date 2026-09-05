"""The embedding endpoints.

`/embed` encodes content to store; `/query` encodes a search string. For symmetric
models the two produce identical vectors, and the response says so via `symmetric`.
Keeping them as separate endpoints means client code does not change when the server
later runs an asymmetric model that does distinguish them.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from embedforge.api.deps import EngineDep, SettingsDep, require_scope
from embedforge.api.responses import ORJSONResponse
from embedforge.auth.models import Principal, Scope
from embedforge.config import Settings
from embedforge.engine.base import TaskType, TextInput
from embedforge.engine.batching import InferenceEngine
from embedforge.errors import InvalidRequestError, PayloadTooLargeError
from embedforge.schemas.embeddings import EmbedRequest, EmbedResponse

router = APIRouter(tags=["embeddings"])

EmbedScope = Annotated[Principal, Depends(require_scope(Scope.EMBED))]
QueryScope = Annotated[Principal, Depends(require_scope(Scope.QUERY))]


def _validate(body: EmbedRequest, settings: Settings, engine: InferenceEngine) -> list[str]:
    """Check limits before any work is queued, so abuse is cheap to reject."""
    if body.model is not None and body.model != engine.info.id:
        raise InvalidRequestError(
            f"This server has {engine.info.id!r} loaded, not {body.model!r}. "
            "One model is active per process; see GET /v1/models."
        )
    texts = body.as_texts()
    if not texts:
        raise InvalidRequestError("`input` must contain at least one item.")
    if len(texts) > settings.max_items_per_request:
        raise PayloadTooLargeError(
            f"{len(texts)} inputs exceeds the limit of {settings.max_items_per_request} "
            "per request. Split the work across requests."
        )
    for index, text in enumerate(texts):
        if not text.strip():
            raise InvalidRequestError(f"Input at index {index} is empty.")
        if len(text) > settings.max_input_chars:
            raise PayloadTooLargeError(
                f"Input at index {index} is {len(text)} characters, over the "
                f"{settings.max_input_chars} character limit."
            )
    return texts


async def _embed(
    body: EmbedRequest,
    task: TaskType,
    settings: Settings,
    engine: InferenceEngine,
) -> ORJSONResponse:
    texts = _validate(body, settings, engine)
    matrix = await engine.embed([TextInput(text) for text in texts], task)
    info = engine.info
    payload: dict[str, Any] = {
        "model": info.id,
        "task": task.value,
        "dimension": info.dimension,
        "normalized": engine.backend.normalize,
        "symmetric": info.symmetric,
        # Serialized straight from numpy by ORJSONResponse; no per-float Python objects.
        "data": [{"index": index, "embedding": row} for index, row in enumerate(matrix)],
        "usage": {"items": len(texts), "characters": sum(len(text) for text in texts)},
    }
    return ORJSONResponse(payload)


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Embed content for storage",
)
async def embed(
    body: EmbedRequest,
    settings: SettingsDep,
    engine: EngineDep,
    _: EmbedScope,
) -> ORJSONResponse:
    """Vectorize documents, passages, or any text you intend to index."""
    return await _embed(body, TaskType.DOCUMENT, settings, engine)


@router.post(
    "/query",
    response_model=EmbedResponse,
    summary="Embed a search query",
)
async def query(
    body: EmbedRequest,
    settings: SettingsDep,
    engine: EngineDep,
    _: QueryScope,
) -> ORJSONResponse:
    """Vectorize a search query, to compare against vectors from `/embed`."""
    return await _embed(body, TaskType.QUERY, settings, engine)
