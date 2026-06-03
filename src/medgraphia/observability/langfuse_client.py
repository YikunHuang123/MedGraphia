"""
Langfuse observability client.

Wraps the Langfuse SDK with:
  - Graceful degradation when langfuse is not installed or TRACING_ENABLED=false
  - Context-manager API (trace → span / generation)
  - Structlog context binding so every log line inside a trace carries trace_id
  - Module-level singleton accessed via get_langfuse_client()
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Generator

import structlog

from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SpanContext — wraps a single Langfuse span
# ---------------------------------------------------------------------------

@dataclass
class SpanContext:
    """
    Thin wrapper around a langfuse Span object.

    Acts as a context manager so callers can write::

        with trace.span("retrieval", input=query) as span:
            items = await retrieve(query)
            span.end(output=f"{len(items)} items")
    """
    _span: Any = field(default=None, repr=False)

    def end(self, output: Any = None, metadata: dict | None = None) -> None:
        """Close the span and optionally attach output/metadata."""
        if self._span is None:
            return
        try:
            self._span.end(
                output=_truncate(output, 4000),
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("langfuse_span_end_failed", error=str(exc))

    def __enter__(self) -> "SpanContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Auto-close on exit. If an exception occurred, mark the span as ERROR.
        if self._span is not None:
            try:
                if exc_type is not None:
                    self._span.end(level="ERROR", status_message=str(exc_val))
                else:
                    self._span.end()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TraceContext — wraps a single Langfuse trace
# ---------------------------------------------------------------------------

@dataclass
class TraceContext:
    """
    Active Langfuse trace handle.

    Created by LangfuseClient.trace() and yielded to the caller.  All child
    spans and generation events attach to this trace automatically.

    When Langfuse is disabled (not installed / missing keys), all methods are
    no-ops and the object still satisfies the interface.
    """
    trace_id: str = ""
    _trace: Any = field(default=None, repr=False)
    _langfuse: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Span
    # ------------------------------------------------------------------

    def span(
        self,
        name: str,
        input: Any = None,
        metadata: dict | None = None,
    ) -> SpanContext:
        """Create a child span.  Returns a no-op SpanContext if tracing is off."""
        if self._trace is None:
            return SpanContext()
        try:
            span = self._trace.span(
                name=name,
                input=_truncate(input, 2000),
                metadata=metadata,
            )
            return SpanContext(_span=span)
        except Exception as exc:
            logger.debug("langfuse_span_create_failed", error=str(exc))
            return SpanContext()

    # ------------------------------------------------------------------
    # Generation event (LLM call)
    # ------------------------------------------------------------------

    def generation(
        self,
        name: str,
        model: str,
        input: Any = None,
        output: Any = None,
        usage: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Record an LLM call as a Langfuse generation event.

        *usage* should follow the format::

            {"promptTokens": 120, "completionTokens": 300, "totalTokens": 420}
        """
        if self._trace is None:
            return
        try:
            self._trace.generation(
                name=name,
                model=model,
                input=_truncate(input, 2000),
                output=_truncate(output, 4000),
                usage=usage,
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("langfuse_generation_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Update trace output on completion
    # ------------------------------------------------------------------

    def update(self, output: Any = None, metadata: dict | None = None) -> None:
        """Finalize the trace with its output value and any extra metadata."""
        if self._trace is None:
            return
        try:
            self._trace.update(
                output=_truncate(output, 4000),
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("langfuse_trace_update_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush buffered events to the Langfuse backend."""
        if self._langfuse is None:
            return
        try:
            self._langfuse.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LangfuseClient — the public facade
# ---------------------------------------------------------------------------

class LangfuseClient:
    """
    Singleton-style Langfuse client.

    Typical usage::

        client = LangfuseClient.from_settings()

        with client.trace("chat", session_id=sid, input=query) as trace:
            with trace.span("retrieval", input=query) as span:
                items = await retrieve(query)
                span.end(output=f"{len(items)} hits")
            trace.generation("llm", model="deepseek-chat",
                             input=prompt, output=answer,
                             usage={"promptTokens": 100})
            trace.update(output=answer)
    """

    def __init__(self, langfuse: Any | None = None) -> None:
        self._langfuse = langfuse
        self._enabled = langfuse is not None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "LangfuseClient":
        """
        Build from MedGraphia Settings.

        Returns a no-op client (tracing disabled) when:
          - TRACING_ENABLED=false (default)
          - langfuse is not installed
          - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are empty
        """
        from medgraphia.config import get_settings
        cfg = get_settings()

        if not cfg.tracing_enabled:
            logger.debug("langfuse_disabled", reason="TRACING_ENABLED=false")
            return cls(langfuse=None)

        public_key = cfg.langfuse_public_key
        secret_val = cfg.langfuse_secret_key
        secret_key = (
            secret_val.get_secret_value()
            if hasattr(secret_val, "get_secret_value")
            else str(secret_val)
        )

        if not public_key or not secret_key:
            logger.warning("langfuse_disabled", reason="missing LANGFUSE_PUBLIC_KEY or SECRET_KEY")
            return cls(langfuse=None)

        try:
            from langfuse import Langfuse  # type: ignore[import]

            lf = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=cfg.langfuse_host,
            )
            logger.info("langfuse_connected", host=cfg.langfuse_host)
            return cls(langfuse=lf)

        except ImportError:
            logger.warning("langfuse_not_installed", tip="pip install langfuse")
            return cls(langfuse=None)

        except Exception as exc:
            logger.warning("langfuse_init_failed", error=str(exc))
            return cls(langfuse=None)

    # ------------------------------------------------------------------
    # Trace context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def trace(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        input: Any = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> Generator[TraceContext, None, None]:
        """
        Context manager that opens a Langfuse trace and yields a TraceContext.
        The trace is flushed to the Langfuse backend on context exit.

        Usage::

            with client.trace("chat", session_id=sid, input=query) as trace:
                trace.generation("llm", model="gpt-4o", input=p, output=r)
                trace.update(output=final_answer)
        """
        if not self._enabled or self._langfuse is None:
            yield TraceContext()
            return

        try:
            trace_obj = self._langfuse.trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                input=_truncate(input, 2000),
                metadata=metadata or {},
                tags=tags or [],
            )
            trace_id = getattr(trace_obj, "id", "") or ""
        except Exception as exc:
            logger.debug("langfuse_trace_create_failed", error=str(exc))
            yield TraceContext()
            return

        # Bind trace_id to structlog context (snapshot & restore pattern)
        prev_ctx = structlog.contextvars.get_contextvars().copy()
        structlog.contextvars.bind_contextvars(langfuse_trace_id=trace_id)

        ctx = TraceContext(
            trace_id=trace_id,
            _trace=trace_obj,
            _langfuse=self._langfuse,
        )
        try:
            yield ctx
        except Exception as exc:
            # If an unhandled exception bubbles up, mark the trace as ERROR
            if trace_obj is not None:
                try:
                    trace_obj.update(level="ERROR", status_message=str(exc))
                except Exception:
                    pass
            raise
        finally:
            # Remove ctx.flush() to avoid blocking the event loop on every request.
            # Restore the structlog context to its exact previous state.
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(**prev_ctx)

    def flush(self) -> None:
        """Manually flush pending events (call this ONLY on application shutdown)."""
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when Langfuse is configured and tracing is active."""
        return self._enabled

    def __repr__(self) -> str:
        return f"LangfuseClient(enabled={self._enabled})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: LangfuseClient | None = None


def get_langfuse_client() -> LangfuseClient:
    """
    Return the process-wide LangfuseClient singleton.
    """
    global _client
    if _client is None:
        _client = LangfuseClient.from_settings()
    return _client


def reset_langfuse_client() -> None:
    """Force re-initialisation of the singleton (useful in tests)."""
    global _client
    _client = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(value: Any, max_len: int) -> str | None:
    """Stringify and truncate a value to avoid Langfuse payload limits."""
    if value is None:
        return None
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + " … [truncated]"
    return text
