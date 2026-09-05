"""Information about the authenticated client."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from embedforge.api.deps import PrincipalDep
from embedforge.auth.models import Scope
from embedforge.schemas.user import UserResponse

router = APIRouter(tags=["user"])

_SCOPE_ENDPOINTS: dict[Scope, str] = {
    Scope.EMBED: "POST /v1/embed",
    Scope.QUERY: "POST /v1/query",
    Scope.MODELS_READ: "GET /v1/models",
}


@router.get("/user", response_model=UserResponse, summary="Who am I?")
async def read_user(principal: PrincipalDep) -> UserResponse:
    """Report the identity, scopes, and lifetime of the token making this call.

    Useful as a credential check: a 200 here means the token is valid, live, and
    reaches this server.
    """
    expires_in: int | None = None
    if principal.expires_at is not None:
        remaining = principal.expires_at - datetime.now(UTC)
        expires_in = max(0, int(remaining.total_seconds()))
    return UserResponse(
        id=principal.id,
        name=principal.name,
        scopes=sorted(principal.scopes),
        authenticated=not principal.anonymous,
        created_at=principal.created_at,
        expires_at=principal.expires_at,
        expires_in_seconds=expires_in,
        endpoints=[
            endpoint for scope, endpoint in _SCOPE_ENDPOINTS.items() if principal.has_scope(scope)
        ],
    )
