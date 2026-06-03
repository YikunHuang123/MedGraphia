"""
Request-level middleware (Pure ASGI).

AuditMiddleware
───────────────
For every incoming HTTP request:

  1. Generate a UUID4 request_id and attach it to:
     - structlog context vars  → all log lines during the request carry it
     - request.state           → downstream handlers can read it
     - response header         → X-Request-ID for client-side correlation

  2. Bind the request_id as a Langfuse `observation_id` (via structlog/chat context)
     so that every chat trace created during this request is linked to the same
     request in Langfuse's session view.

  3. Emit a structured access log on completion with method, path,
     status code, and elapsed milliseconds.

Why Pure ASGI?
──────────────
Starlette's `BaseHTTPMiddleware` uses an internal asyncio.Queue to bridge the
high-level `dispatch` method with the low-level ASGI interface. This breaks
backpressure on `StreamingResponse` (like our SSE chat), potentially causing
unbounded memory growth if the client is slow.

This Pure ASGI implementation avoids the queue and task overhead.
"""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

from medgraphia.logger import get_logger

if TYPE_CHECKING:
    from starlette.types import Message

logger = get_logger(__name__)


class AuditMiddleware:
    """
    Pure ASGI middleware for request auditing and request_id propagation.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # ── Context Initialisation ───────────────────────────────────────────
        # Clear previous and bind current request context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=scope["method"],
            path=scope["path"],
        )

        # ── Request State ───────────────────────────────────────────────────
        # Starlette/FastAPI Request objects look at scope["state"]
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Inject X-Request-ID into response headers
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
            
            await send(message)

            if message["type"] == "http.response.body" and not message.get("more_body", False):
                # Request is finished (last chunk of body sent)
                # We can't easily get the status code here without wrapping 'send' 
                # to catch http.response.start, but we'll log based on what we have.
                # Actually, status is in http.response.start.
                pass

        # We need to capture the status code from http.response.start
        status_code = [200]  # mutable container to capture value from closure

        async def logging_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_code[0] = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
            
            await send(message)

            if message["type"] == "http.response.body" and not message.get("more_body", False):
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "http_request",
                    status_code=status_code[0],
                    elapsed_ms=elapsed_ms,
                )

        await self.app(scope, receive, logging_send)
