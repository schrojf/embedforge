"""Authenticated-client description."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from embedforge.auth.models import Scope


class UserResponse(BaseModel):
    """What `/v1/user` reports about the caller."""

    id: str = Field(description="Token id, the part before the secret in the token.")
    name: str
    scopes: list[Scope]
    authenticated: bool = Field(
        description="False when the server runs with authentication disabled.",
    )
    created_at: datetime | None = None
    expires_at: datetime | None = None
    expires_in_seconds: int | None = Field(
        default=None, description="Seconds until this token stops working, if it expires."
    )
    endpoints: list[str] = Field(
        default_factory=list, description="API endpoints this token's scopes allow."
    )
