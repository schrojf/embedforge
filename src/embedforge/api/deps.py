"""Dependency injection for routes."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Annotated, Any

import structlog
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from embedforge.auth.models import ANONYMOUS, Principal, Scope
from embedforge.auth.store import TokenStore
from embedforge.config import Settings
from embedforge.engine.batching import InferenceEngine
from embedforge.errors import AuthenticationError, ModelNotReadyError, PermissionDeniedError

bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="API token",
    description="Send `Authorization: Bearer ef_<id>_<secret>`.",
)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_token_store(request: Request) -> TokenStore:
    return request.app.state.token_store


def get_engine(request: Request) -> InferenceEngine:
    engine: InferenceEngine | None = getattr(request.app.state, "engine", None)
    if engine is None or not engine.ready:
        raise ModelNotReadyError("Model is not loaded yet.")
    return engine


async def get_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> Principal:
    """Resolve the caller, or fail with 401."""
    settings: Settings = request.app.state.settings
    if not settings.auth_enabled:
        return ANONYMOUS
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Missing bearer token.")
    store: TokenStore = request.app.state.token_store
    principal = store.authenticate(credentials.credentials)
    if principal is None:
        raise AuthenticationError()
    structlog.contextvars.bind_contextvars(principal=principal.id)
    return principal


SettingsDep = Annotated[Settings, Depends(get_settings)]
EngineDep = Annotated[InferenceEngine, Depends(get_engine)]
TokenStoreDep = Annotated[TokenStore, Depends(get_token_store)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]


def require_scope(scope: Scope) -> Callable[..., Coroutine[Any, Any, Principal]]:
    """Dependency factory: authenticate, then check one scope."""

    async def dependency(principal: PrincipalDep) -> Principal:
        if not principal.has_scope(scope):
            raise PermissionDeniedError(f"This token lacks the {scope.value!r} scope.")
        return principal

    return dependency
