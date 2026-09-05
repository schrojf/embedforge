"""Liveness, readiness, and service metadata.

Unauthenticated by design: load balancers and orchestrators need them, and they
expose nothing sensitive.
"""

from typing import Any

from fastapi import APIRouter, Request

from embedforge.api.responses import ORJSONResponse
from embedforge.engine.batching import InferenceEngine
from embedforge.version import __version__

router = APIRouter(tags=["health"])


@router.get("/", summary="Service banner")
async def root(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return {
        "service": "embedforge",
        "version": __version__,
        "model": settings.model_id,
        "docs": "/docs" if settings.docs_enabled else None,
    }


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Alive and serving HTTP. Never touches the model, so it cannot be blocked by it."""
    return {"status": "ok", "version": __version__}


@router.get("/readyz", summary="Readiness probe")
async def readyz(request: Request) -> ORJSONResponse:
    """Ready to serve embeddings: the model is loaded and warm.

    Returns 503 while the model is still loading, so a rolling deploy does not send
    traffic to a container that would only queue it.
    """
    engine: InferenceEngine | None = getattr(request.app.state, "engine", None)
    ready = engine is not None and engine.ready
    body: dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "version": __version__,
        "model": request.app.state.settings.model_id,
    }
    if engine is not None:
        body["engine"] = engine.stats()
    return ORJSONResponse(body, status_code=200 if ready else 503)
