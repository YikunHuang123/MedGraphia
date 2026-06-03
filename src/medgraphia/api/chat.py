"""
Chat endpoints.

POST /chat        — Synchronous Q&A: blocks until the full cited answer is ready.
POST /chat/stream — Streaming Q&A: Server-Sent Events, progressive token delivery.

Both run the complete MedGraphia pipeline:
  1. Session management  — create or resume a named conversation
  2. Retrieval           — NER → router → 3-path retrieval → RRF fusion → reranker
  3. Generation          — prompt selection → LLM call → citation injection
  4. History persistence — user + assistant messages appended to in-memory session
  5. Observability       — every stage traced to Langfuse (no-op when disabled)
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from medgraphia.api.auth import require_api_key
from medgraphia.api.deps import create_or_get_session, save_session
from medgraphia.api.schemas import ChatRequest, ChatResponse
from medgraphia.domain.base import Language, QueryType
from medgraphia.domain.chat import Message
from medgraphia.generation.pipeline import GenerationPipeline
from medgraphia.logger import get_logger
from medgraphia.observability import TraceContext, get_langfuse_client
from medgraphia.retrieval.pipeline import RetrievalPipeline

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# Lazy pipeline singletons — with async locks to prevent race conditions
# ---------------------------------------------------------------------------

_retrieval_pipeline: RetrievalPipeline | None = None
_generation_pipeline: GenerationPipeline | None = None

_retrieval_lock = asyncio.Lock()
_generation_lock = asyncio.Lock()


async def _get_retrieval() -> RetrievalPipeline:
    """Async-safe getter for the retrieval pipeline singleton."""
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        async with _retrieval_lock:
            # Double-check pattern
            if _retrieval_pipeline is None:
                # Use a blocking-to-async wrapper if initialization is heavy
                # but for now synchronous factory is called in the async context.
                _retrieval_pipeline = RetrievalPipeline.from_settings()
    return _retrieval_pipeline


async def _get_generation() -> GenerationPipeline:
    """Async-safe getter for the generation pipeline singleton."""
    global _generation_pipeline
    if _generation_pipeline is None:
        async with _generation_lock:
            if _generation_pipeline is None:
                _generation_pipeline = GenerationPipeline.from_settings()
    return _generation_pipeline


# ---------------------------------------------------------------------------
# POST /chat — synchronous
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatResponse, summary="Synchronous chat Q&A")
async def chat(
    body: ChatRequest,
    request: Request,
    principal: dict = Depends(require_api_key),
) -> ChatResponse:
    """
    Execute the full retrieval-augmented generation pipeline and return a
    complete, citation-annotated answer.
    """
    t0 = time.monotonic()
    session = create_or_get_session(body.session_id)
    request_id: str = request.state.request_id if hasattr(request.state, "request_id") else ""
    langfuse = get_langfuse_client()

    with langfuse.trace(
        "chat",
        session_id=session.session_id,
        user_id=principal.get("id", "anonymous"),
        input=body.message,
        metadata={
            "language": body.language.value,
            "domain": body.domain or "",
            "request_id": request_id,
        },
        tags=["sync"],
    ) as trace:
        try:
            result = await _run_full_pipeline(
                query=body.message,
                language=body.language,
                trace=trace,
            )
        except Exception as exc:
            logger.error("chat_pipeline_error", error=str(exc), request_id=request_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Pipeline execution failed. Please try again.",
            ) from exc

        # Persist conversation turn
        session.messages.append(
            Message(session_id=session.session_id, role="user", content=body.message)
        )
        session.messages.append(
            Message(
                session_id=session.session_id,
                role="assistant",
                content=result["answer"],
                citations=result["citations"],
                model_used=result["model_used"],
                retrieval_paths_used=result["retrieval_paths"],
            )
        )
        save_session(session)

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info("chat_ok", session_id=session.session_id, latency_ms=latency_ms)

        trace.update(
            output=result["answer"][:500],
            metadata={"latency_ms": latency_ms, "model_used": result["model_used"]},
        )

        return ChatResponse(
            session_id=session.session_id,
            content=result["answer"],
            citations=result["citations"],
            retrieval_paths_used=result["retrieval_paths"],
            model_used=result["model_used"],
            query_type=result["query_type"],
            disclaimer=result["disclaimer"],
        )


# ---------------------------------------------------------------------------
# POST /chat/stream — SSE streaming
# ---------------------------------------------------------------------------

@router.post("/stream", summary="Streaming chat Q&A (Server-Sent Events)")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    principal: dict = Depends(require_api_key),
) -> StreamingResponse:
    """
    Stream the answer token-by-token as Server-Sent Events.

    Refactored in Phase 8 to avoid fragile JSON streaming.
    1. Retrieval runs to completion.
    2. LLM streams raw text tokens (immediate UI feedback).
    3. Structured citations and disclaimer sent as final metadata events.
    """
    session = create_or_get_session(body.session_id)
    request_id: str = request.state.request_id if hasattr(request.state, "request_id") else ""
    langfuse = get_langfuse_client()

    async def _event_stream() -> AsyncIterator[str]:
        with langfuse.trace(
            "chat_stream",
            session_id=session.session_id,
            user_id=principal.get("id", "anonymous"),
            input=body.message,
            metadata={"language": body.language.value, "request_id": request_id},
            tags=["stream"],
        ) as trace:
            # ── Step 1: Retrieval ───────────────────────────────────────────
            retrieval = await _get_retrieval()
            generation = await _get_generation()

            with trace.span("retrieval", input=body.message) as span:
                try:
                    reranked = await retrieval.execute(
                        query=body.message,
                        language=body.language,
                    )
                except Exception as exc:
                    logger.error("stream_retrieval_failed", error=str(exc))
                    yield _sse({"type": "error", "detail": "Retrieval failed."})
                    return

                items = getattr(reranked, "items", [])
                query_type: QueryType = getattr(reranked, "query_type", QueryType.PATIENT_FAQ)
                retrieval_paths = list({item.source.value for item in items})
                span.end(output=f"{len(items)} items")

            # ── Step 2: Prepare Prompts ─────────────────────────────────────
            from medgraphia.generation.citation import build_numbered_context, inject_citations
            from medgraphia.generation.llm_router import LLMRouter
            from medgraphia.llm.gateway import CompletionRequest

            context_str = build_numbered_context(items)
            
            # Use the NEW public interface to get prompts (avoids internal API leakage)
            components = generation.get_streaming_components(query_type, body.language)
            system_prompt = components["system_prompt"]
            disclaimer = components["disclaimer"]

            user_prompt = (
                f"CONTEXT PASSAGES:\n{context_str}\n\n"
                f"QUESTION: {body.message}\n"
                f"RESPONSE LANGUAGE: {body.language.value.upper()}\n\n"
                "INSTRUCTIONS:\n"
                "1. Answer based ONLY on the provided context.\n"
                "2. Use [N] for inline citations.\n"
                "3. Provide a direct and clear medical explanation."
            )

            llm_router = LLMRouter.from_settings()
            gateway, routing = llm_router.route(query_type, body.language)

            # ── Step 3: Stream LLM tokens (Pure Text) ────────────────────────
            stream_req = CompletionRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stream=True,
                temperature=0.1,
            )

            accumulated: list[str] = []
            with trace.span("generation_stream", input=body.message):
                try:
                    async for token in gateway.astream(stream_req):
                        accumulated.append(token)
                        # Yield raw text chunk (Best for UX: no JSON overhead)
                        yield _sse({"type": "chunk", "content": token})
                except Exception as exc:
                    logger.error("stream_generation_failed", error=str(exc))
                    yield _sse({"type": "error", "detail": "Generation failed."})
                    return

            full_text = "".join(accumulated)

            # ── Step 4: Finalise (Citations & History) ──────────────────────
            # Inject citations from the final text
            citation_result = inject_citations(full_text, items)

            # Persist session history
            session.messages.append(
                Message(session_id=session.session_id, role="user", content=body.message)
            )
            session.messages.append(
                Message(
                    session_id=session.session_id,
                    role="assistant",
                    content=full_text,
                    citations=citation_result.citations,
                    model_used=routing.model_name,
                    retrieval_paths_used=retrieval_paths,
                )
            )
            save_session(session)

            # ── Step 5: Send Metadata Events ────────────────────────────────
            yield _sse({
                "type": "citations",
                "citations": [c.model_dump() for c in citation_result.citations],
            })
            yield _sse({
                "type": "done",
                "session_id": session.session_id,
                "model_used": routing.model_name,
                "query_type": query_type.value,
                "disclaimer": disclaimer,
            })

            trace.update(output=full_text[:500])

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _run_full_pipeline(
    query: str,
    language: Language,
    trace: TraceContext,
) -> dict:
    """Execute retrieve → generate and return a unified result dict."""
    retrieval = await _get_retrieval()
    generation = await _get_generation()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    with trace.span("retrieval", input=query) as span:
        reranked = await retrieval.execute(query=query, language=language)
        items = getattr(reranked, "items", [])
        query_type: QueryType = getattr(reranked, "query_type", QueryType.PATIENT_FAQ)
        retrieval_paths = list({item.source.value for item in items})
        span.end(output=f"{len(items)} items")

    # ── Generation ────────────────────────────────────────────────────────────
    with trace.span("generation", input=query) as span:
        gen_result = await generation.generate(
            question=query,
            query_type=query_type,
            retrieved_items=items,
            language=language,
        )
        model_used = gen_result.routing.model_name if gen_result.routing else ""
        span.end(output=gen_result.answer[:200])

    return {
        "answer": gen_result.answer,
        "citations": gen_result.citations,
        "retrieval_paths": retrieval_paths,
        "model_used": model_used,
        "query_type": query_type,
        "disclaimer": gen_result.disclaimer,
    }


def _sse(data: dict) -> str:
    """Encode a dict as a single SSE data frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
