"""Application errors.

Every failure the API returns is one of these, so responses share a single
envelope: `{"error": {"type": ..., "message": ...}, "request_id": ...}`.
"""

from __future__ import annotations


class EmbedForgeError(Exception):
    """Base class for errors that map to an HTTP response."""

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(self, message: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.headers = headers or {}


class AuthenticationError(EmbedForgeError):
    """No credentials, or credentials that do not identify a live token."""

    status_code: int = 401
    error_type: str = "authentication_error"

    def __init__(self, message: str = "Invalid or missing API token.") -> None:
        super().__init__(message, headers={"WWW-Authenticate": "Bearer"})


class PermissionDeniedError(EmbedForgeError):
    """A valid token that lacks the scope for this endpoint."""

    status_code: int = 403
    error_type: str = "permission_error"


class InvalidRequestError(EmbedForgeError):
    status_code: int = 400
    error_type: str = "invalid_request_error"


class PayloadTooLargeError(EmbedForgeError):
    status_code: int = 413
    error_type: str = "payload_too_large"


class ModelNotReadyError(EmbedForgeError):
    """The engine is not (yet) able to serve, e.g. during startup or shutdown."""

    status_code: int = 503
    error_type: str = "model_not_ready"


class OverloadedError(EmbedForgeError):
    """The inference queue is full; the caller should retry with backoff."""

    status_code: int = 503
    error_type: str = "overloaded"

    def __init__(self, message: str = "Server is overloaded, retry shortly.") -> None:
        super().__init__(message, headers={"Retry-After": "1"})


class RequestTimeoutError(EmbedForgeError):
    """The request waited longer than `request_timeout` for its embeddings."""

    status_code: int = 504
    error_type: str = "timeout"
