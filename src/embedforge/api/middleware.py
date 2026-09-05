"""Request context, access logging, and HTTP metrics.

Written as a raw ASGI middleware rather than `BaseHTTPMiddleware`: it costs less per
request and does not interfere with streaming responses or background tasks.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import MutableMapping
from typing import Any

import structlog

from embedforge import metrics
from embedforge.logging import get_logger

log = get_logger("embedforge.access")

REQUEST_ID_HEADER = "x-request-id"

Scope = MutableMapping[str, Any]


class RequestContextMiddleware:
    """Attach a request id, log one access line per request, and record metrics."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(REQUEST_ID_HEADER.encode())
        request_id = incoming.decode("latin-1")[:128] if incoming else uuid.uuid4().hex

        method: str = scope.get("method", "")
        path: str = scope.get("path", "")
        started = time.perf_counter()
        status_code = 500

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, method=method, path=path)

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"].append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode("latin-1"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            duration = time.perf_counter() - started
            self._record(scope, method, status_code, duration)
            log.exception("request_failed", duration_ms=round(duration * 1000, 2))
            raise
        else:
            duration = time.perf_counter() - started
            self._record(scope, method, status_code, duration)
            log.info(
                "request",
                status=status_code,
                duration_ms=round(duration * 1000, 2),
            )
        finally:
            structlog.contextvars.clear_contextvars()

    @staticmethod
    def _record(scope: Scope, method: str, status: int, duration: float) -> None:
        # Use the templated route path, so path parameters cannot explode label cardinality.
        route = scope.get("route")
        endpoint = getattr(route, "path", None) or "unmatched"
        metrics.HTTP_REQUESTS.labels(method, endpoint, str(status)).inc()
        metrics.HTTP_DURATION.labels(method, endpoint).observe(duration)
