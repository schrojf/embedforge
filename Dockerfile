# syntax=docker/dockerfile:1.9
#
# Production image for EmbedForge.
#
#   docker build -t embedforge:local --build-arg VERSION="$(git describe --tags --always)" .
#   docker run --rm -p 8000:8000 -v embedforge-data:/var/lib/embedforge embedforge:local
#
# See docs/deployment.md for the full deployment guide.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


# ---- Build stage: resolve and install into a self-contained virtualenv ----

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_CONFIG_FILE=/app/uv.toml

WORKDIR /app

# Dependencies first, from the lockfile only. This layer is rebuilt when uv.lock
# changes, not when application code does, which is most of the build time.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=uv.toml,target=uv.toml \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project --no-editable

COPY pyproject.toml uv.lock uv.toml README.md LICENSE /app/
COPY src /app/src

# The version normally comes from git tags, and .dockerignore keeps .git out of the
# build context. Pass --build-arg VERSION=... to stamp a real version into the image.
ARG VERSION=0.0.0
ENV UV_DYNAMIC_VERSIONING_BYPASS=${VERSION}

# --no-editable installs a real wheel, so the runtime image needs only the venv.
# No cache mount here, and --reinstall-package: uv keys its build cache on the source
# tree, not on UV_DYNAMIC_VERSIONING_BYPASS, so a cached wheel would silently carry
# the version from a previous build.
RUN uv sync --frozen --no-dev --no-editable --reinstall-package embedforge


# ---- Runtime stage ----

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG VERSION=0.0.0
LABEL org.opencontainers.image.title="embedforge" \
      org.opencontainers.image.description="Embedding server with embed and query endpoints" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT"

# tini reaps orphaned children and forwards signals, so SIGTERM reaches the server
# and shutdown stays graceful even with multiple worker processes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 embedforge \
    && useradd --system --uid 10001 --gid embedforge --no-create-home embedforge

# Application files stay root-owned and read-only to the service account.
COPY --from=builder --chown=root:root /app/.venv /app/.venv

# Writable state: the token file, and downloaded model files. Mount a volume here.
RUN install -d -o embedforge -g embedforge -m 0750 /var/lib/embedforge

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    EMBEDFORGE_HOST=0.0.0.0 \
    EMBEDFORGE_PORT=8000 \
    EMBEDFORGE_LOG_FORMAT=json \
    EMBEDFORGE_TOKENS_FILE=/var/lib/embedforge/tokens.json \
    EMBEDFORGE_MODEL_DIR=/var/lib/embedforge/models

USER embedforge
WORKDIR /var/lib/embedforge
EXPOSE 8000

# Readiness, not liveness: the container is only healthy once the model is loaded.
# start-period covers model loading, which grows with the model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('EMBEDFORGE_PORT','8000')+'/readyz', timeout=4).status == 200 else 1)"

# ENTRYPOINT is the CLI, so `docker run <image> token list` works as well as serving.
ENTRYPOINT ["/usr/bin/tini", "--", "embedforge"]
CMD ["serve"]
