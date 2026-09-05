"""Token and principal models.

Tokens look like ``ef_<id>_<secret>``. The id is stored in the clear so lookup is a
dict hit; only a SHA-256 digest of the whole token is persisted, so a leaked token
file cannot be replayed. SHA-256 (rather than a password hash like argon2) is the
right choice here because the secret is 256 bits of CSPRNG output: there is no
low-entropy password to protect and verification sits on the hot request path.
"""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

TOKEN_PREFIX = "ef"
"""Namespace marker, so a leaked token is greppable and identifiable."""


class Scope(StrEnum):
    """Capabilities a token may carry."""

    EMBED = "embed"
    QUERY = "query"
    MODELS_READ = "models:read"
    ADMIN = "admin"
    """Implies every other scope."""


DEFAULT_SCOPES: tuple[Scope, ...] = (Scope.EMBED, Scope.QUERY, Scope.MODELS_READ)


def utcnow() -> datetime:
    return datetime.now(UTC)


def generate_token() -> tuple[str, str]:
    """Return a fresh `(token_id, plaintext_token)` pair."""
    token_id = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    return token_id, f"{TOKEN_PREFIX}_{token_id}_{secret}"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_token_id(token: str) -> str | None:
    """Extract the id from a presented token, or None if it is not our format."""
    # maxsplit=2: the secret is url-safe base64 and may itself contain "_".
    parts = token.split("_", 2)
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX or not parts[1] or not parts[2]:
        return None
    return parts[1]


class TokenRecord(BaseModel):
    """One API token as persisted in the token file."""

    id: str
    name: str
    token_hash: str
    scopes: list[Scope] = Field(default_factory=lambda: list(DEFAULT_SCOPES))
    created_at: datetime
    expires_at: datetime | None = None
    disabled: bool = False
    note: str | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and (now or utcnow()) >= self.expires_at

    def is_active(self, now: datetime | None = None) -> bool:
        return not self.disabled and not self.is_expired(now)

    def status(self, now: datetime | None = None) -> str:
        if self.disabled:
            return "disabled"
        if self.is_expired(now):
            return "expired"
        return "active"

    def verify(self, token: str) -> bool:
        return hmac.compare_digest(self.token_hash, hash_token(token))

    def to_principal(self) -> "Principal":
        return Principal(
            id=self.id,
            name=self.name,
            scopes=frozenset(self.scopes),
            created_at=self.created_at,
            expires_at=self.expires_at,
        )


class Principal(BaseModel):
    """The authenticated client behind a request."""

    model_config = {"frozen": True}

    id: str
    name: str
    scopes: frozenset[Scope]
    created_at: datetime | None = None
    expires_at: datetime | None = None
    anonymous: bool = False
    """True when authentication is disabled, so responses can say so plainly."""

    def has_scope(self, scope: Scope) -> bool:
        return Scope.ADMIN in self.scopes or scope in self.scopes


ANONYMOUS = Principal(
    id="anonymous",
    name="anonymous",
    scopes=frozenset({Scope.ADMIN}),
    anonymous=True,
)
"""Principal used when `auth_enabled` is false."""


class TokenFile(BaseModel):
    """On-disk container, versioned so the format can change safely."""

    version: int = 1
    tokens: dict[str, TokenRecord] = Field(default_factory=dict)
