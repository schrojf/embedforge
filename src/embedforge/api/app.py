"""Application factory."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from embedforge import metrics
from embedforge.api.errors import install_error_handlers
from embedforge.api.middleware import RequestContextMiddleware
from embedforge.api.responses import ORJSONResponse
from embedforge.api.routers import embeddings, health, models, user
from embedforge.auth.store import TokenStore
from embedforge.config import Settings, get_settings
from embedforge.engine.batching import InferenceEngine
from embedforge.engine.registry import create_backend
from embedforge.logging import configure_logging, get_logger
from embedforge.version import __version__

log = get_logger(__name__)

DESCRIPTION = """
Embedding server with a single active model.

* `POST /v1/embed` — vectorize content you intend to store.
* `POST /v1/query` — vectorize a search query.
* `GET /v1/user` — check which token is calling.

Authenticate with `Authorization: Bearer ef_<id>_<secret>`; create tokens with
`embedforge token create`.
""".strip()


def build_engine(settings: Settings) -> InferenceEngine:
    """Construct (but do not start) the engine described by the settings."""
    return InferenceEngine(
        create_backend(settings),
        max_batch_size=settings.max_batch_size,
        batch_wait_ms=settings.batch_wait_ms,
        queue_max_size=settings.queue_max_size,
        workers=settings.inference_workers,
        request_timeout=settings.request_timeout,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    Used as a factory (`embedforge.api.app:create_app`) so each uvicorn worker
    process constructs its own engine and model.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        engine = build_engine(settings)
        app.state.engine = engine
        # Loading here (not on first request) is what makes /readyz meaningful.
        await engine.start()
        info = engine.info
        metrics.MODEL_INFO.info(
            {
                "id": info.id,
                "name": info.name,
                "dimension": str(info.dimension),
                "symmetric": str(info.symmetric).lower(),
                "version": __version__,
            }
        )
        log.info(
            "server_ready",
            version=__version__,
            model=info.id,
            env=settings.env,
            auth_enabled=settings.auth_enabled,
        )
        try:
            yield
        finally:
            await engine.aclose()

    app = FastAPI(
        title="EmbedForge",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
        root_path=settings.root_path,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    # Available before startup so health checks and OpenAPI work without the engine.
    app.state.settings = settings
    app.state.token_store = TokenStore(
        settings.tokens_file, reload_interval=settings.tokens_reload_interval
    )
    app.state.engine = None

    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    app.include_router(health.router)
    app.include_router(user.router, prefix="/v1")
    app.include_router(embeddings.router, prefix="/v1")
    app.include_router(models.router, prefix="/v1")

    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def prometheus_metrics() -> Response:
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
