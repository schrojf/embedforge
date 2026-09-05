"""Exception handlers, so every failure shares one response envelope."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from embedforge.api.responses import ORJSONResponse
from embedforge.errors import EmbedForgeError
from embedforge.logging import get_logger

log = get_logger(__name__)

_STATUS_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "invalid_request_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


def _request_id() -> str | None:
    value = structlog.contextvars.get_contextvars().get("request_id")
    return value if isinstance(value, str) else None


def error_response(
    status_code: int,
    error_type: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> ORJSONResponse:
    error: dict[str, Any] = {"type": error_type, "message": message}
    if extra:
        error.update(extra)
    return ORJSONResponse(
        status_code=status_code,
        content={"error": error, "request_id": _request_id()},
        headers=headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(EmbedForgeError)
    async def _app_error(_: Request, exc: EmbedForgeError) -> ORJSONResponse:
        return error_response(exc.status_code, exc.error_type, exc.message, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> ORJSONResponse:
        return error_response(
            422,
            "invalid_request_error",
            "Request body failed validation.",
            extra={"details": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> ORJSONResponse:
        error_type = _STATUS_TYPES.get(exc.status_code, "http_error")
        headers = dict(exc.headers) if exc.headers else None
        return error_response(exc.status_code, error_type, str(exc.detail), headers=headers)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> ORJSONResponse:
        # Log the detail, return none of it: internals are not the caller's business.
        log.error("unhandled_exception", error=str(exc), exc_info=exc)
        return error_response(500, "internal_error", "Internal server error.")
