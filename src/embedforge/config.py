"""Application settings.

All settings are read from environment variables prefixed with ``EMBEDFORGE_``
(or from a ``.env`` file in the working directory). The layout is deliberately
flat: environment variables are the primary deployment interface, so
``EMBEDFORGE_MAX_BATCH_SIZE`` beats a nested equivalent.

See ``docs/configuration.md`` for the full reference.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "prod"]
LogFormat = Literal["json", "console"]
Device = Literal["cpu", "cuda", "auto"]


class Settings(BaseSettings):
    """Effective runtime configuration for the server and the CLIs."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Our own settings use the `model_` prefix; stop pydantic from claiming it.
        protected_namespaces=(),
    )

    # ---- Runtime ----

    env: Environment = "prod"
    """Deployment environment. `dev` enables friendlier defaults, never use in production."""

    log_level: str = "INFO"
    log_format: LogFormat = "json"

    # ---- HTTP server ----

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    workers: int = Field(default=1, ge=1)
    """OS processes serving HTTP.

    Keep this at 1 for CPU inference: every worker loads its own copy of the model
    and starts its own inference threads, so more workers means more memory and CPU
    oversubscription, not more throughput. Scale with `inference_workers`,
    `max_batch_size`, and horizontally with containers instead.
    """

    root_path: str = ""
    """ASGI root path, when mounted behind a reverse proxy on a subpath."""

    proxy_headers: bool = True
    forwarded_allow_ips: str = "127.0.0.1"
    timeout_keep_alive: int = Field(default=5, ge=0)
    timeout_graceful_shutdown: int = Field(default=30, ge=0)

    limit_concurrency: int = Field(default=512, ge=1)
    """Maximum concurrent connections uvicorn accepts before returning 503."""

    cors_origins: list[str] = Field(default_factory=list)
    docs_enabled: bool = True
    """Serve the OpenAPI schema and interactive docs at /docs and /redoc."""

    # ---- Data locations ----

    tokens_file: Path = Path("data/tokens.json")
    model_dir: Path = Path("data/models")

    # ---- Authentication ----

    auth_enabled: bool = True
    """Require a bearer token on API endpoints. Only disable for local development."""

    tokens_reload_interval: float = Field(default=5.0, ge=0)
    """Seconds between checks for changes to the token file, so the CLI can add or
    revoke tokens without restarting the server. Set to 0 to reload on every request."""

    # ---- Model and inference engine ----

    model_id: str = "dev-hash"
    """The single model loaded at startup. See `embedforge model list`."""

    device: Device = "cpu"
    """`cpu`, `cuda`, or `auto` to prefer GPU when its runtime is available."""

    inference_workers: int = Field(default=1, ge=1)
    """Threads running model inference concurrently.

    Inference releases the GIL, so these are real parallel workers. One is right for
    most CPU deployments because a single session already uses every core; raise it
    only alongside a matching drop in `intra_op_threads`.
    """

    intra_op_threads: int = Field(default=0, ge=0)
    """Threads used *inside* one inference call. 0 means auto (cores // inference_workers)."""

    max_batch_size: int = Field(default=32, ge=1)
    """Largest batch handed to the model in one call."""

    batch_wait_ms: float = Field(default=5.0, ge=0)
    """How long the dispatcher waits to merge concurrent requests into one batch.

    This is the main latency/throughput dial: a few milliseconds buys a large
    throughput win under concurrency and is invisible next to inference time.
    Set to 0 to disable waiting (batches still form from already-queued work).
    """

    queue_max_size: int = Field(default=512, ge=1)
    """Items allowed to wait for inference. Beyond this the server sheds load with 503."""

    request_timeout: float = Field(default=30.0, gt=0)
    """Seconds a request may wait for its embeddings before returning 504."""

    max_items_per_request: int = Field(default=256, ge=1)
    max_input_chars: int = Field(default=32_768, ge=1)
    """Per-item input limit, checked before tokenization to bound the work a caller can buy."""

    normalize_embeddings: bool = True
    """L2-normalize output vectors, so cosine similarity is a plain dot product."""

    # ---- Observability ----

    metrics_enabled: bool = True
    """Expose Prometheus metrics at /metrics. Keep this endpoint off the public internet."""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept `a,b` as well as a JSON list, which is friendlier in env vars."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @property
    def is_dev(self) -> bool:
        return self.env == "dev"

    def effective_intra_op_threads(self) -> int:
        """Resolve `intra_op_threads`, dividing available cores across inference workers."""
        if self.intra_op_threads > 0:
            return self.intra_op_threads
        cores = _available_cores()
        return max(1, cores // self.inference_workers)


def _available_cores() -> int:
    """CPU cores actually usable by this process, honoring cgroup/affinity limits."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once."""
    return Settings()
