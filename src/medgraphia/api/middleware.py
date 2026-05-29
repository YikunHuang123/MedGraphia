"""
Request-level middleware: assigns a unique request_id to every incoming request
and emits a structured access log entry on completion.
"""
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

import structlog
from medgraphia.logger import get_logger

logger = get_logger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Attaches request_id to structlog context vars so every log line emitted
    during request handling automatically includes request_id.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        # Bind request_id to the structlog context for this coroutine tree
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "http_request",
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
        )

        # Propagate request_id to the caller for correlation
        response.headers["X-Request-ID"] = request_id
        return response
