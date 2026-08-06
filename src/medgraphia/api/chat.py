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
from collections.abc import AsyncIterator

# import Any
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from medgraphia.api.auth import require_api_key
from medgraphia.api.deps import create_or_get_session, save_session
from medgraphia.api.schemas import ChatRequest, ChatResponse
from medgraphia.domain.base import Language, QueryType
from medgraphia.domain.chat import Message, Session
from medgraphia.generation.citation import inject_citations
from medgraphia.generation.guard import LlamaGuard
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
_guard: LlamaGuard | None = None

_retrieval_lock = asyncio.Lock()
_generation_lock = asyncio.Lock()
_guard_lock = asyncio.Lock()


async def _get_retrieval() -> RetrievalPipeline:
    """Async-safe getter for the retrieval pipeline singleton."""
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        async with _retrieval_lock:
            if _retrieval_pipeline is None:
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


async def _get_guard() -> LlamaGuard:
    """Async-safe getter for the safety guard singleton."""
    global _guard
    if _guard is None:
        async with _guard_lock:
            if _guard is None:
                _guard = LlamaGuard()
    return _guard


# ---------------------------------------------------------------------------
# GET /chat/sessions — list history
# ---------------------------------------------------------------------------


@router.get("/sessions", summary="List all chat sessions for the current user")
async def list_sessions(
    principal: dict = Depends(require_api_key),
) -> list[dict[str, Any]]:
    """Return a summary of all stored chat sessions."""
    from medgraphia.graph.queries import list_chat_sessions

    user_id = principal.get("id", "anonymous")
    return await list_chat_sessions(user_id=user_id)


@router.get("/sessions/{session_id}", response_model=Session, summary="Retrieve a specific session")
async def get_session_history(
    session_id: str,
    principal: dict = Depends(require_api_key),
) -> Session:
    """Retrieve the full message history and metadata for a session."""
    from medgraphia.api.deps import get_session

    session = await get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found.",
        )
    return session


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
    session = await create_or_get_session(body.session_id)
    request_id: str = request.state.request_id if hasattr(request.state, "request_id") else ""
    langfuse = get_langfuse_client()

    # Auto-detect language if UNKNOWN
    language = body.language
    if language == Language.UNKNOWN:
        language = Language.detect(body.message)

    logger.info("query_received", language=language.value, stream=False)

    guard = await _get_guard()
    safety = await guard.check_input(body.message, history=session.messages)
    if not safety.is_safe:
        return ChatResponse(
            session_id=session.session_id,
            content=(
                "⚠️ [Blocked by Safety Guardrails] Your query was found to violate "
                f"our safety policies ({safety.category}). Please rephrase your question."
            ),
            citations=[],
            retrieval_paths_used=[],
            model_used="guardrail",
            query_type=QueryType.PATIENT_FAQ,
            disclaimer="This query was blocked for safety reasons.",
        )

    with langfuse.trace(
        "chat",
        session_id=session.session_id,
        user_id=principal.get("id", "anonymous"),
        input=body.message,
        metadata={
            "language": language.value,
            "request_id": request_id,
        },
        tags=["sync"],
    ) as trace:
        # ── Step 2: Generation ────────────────────────────────────────────
        try:
            result = await _run_full_pipeline(
                query=body.message,
                language=language,
                trace=trace,
                history=session.messages,
                user_id=principal.get("id", "anonymous"),
            )
        except Exception as exc:
            logger.error("chat_pipeline_error", error=str(exc), request_id=request_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Pipeline execution failed. Please try again.",
            ) from exc

        # ── Step 2a: Output Safety Check ────────────────────────────────
        output_safety = await guard.check_output(
            body.message, result["answer"], history=session.messages
        )
        if not output_safety.is_safe:
            return ChatResponse(
                session_id=session.session_id,
                content=(
                    "⚠️ [Blocked by Safety Guardrails] The generated response was found to "
                    "violate our safety policies. Please rephrase your question."
                ),
                citations=[],
                retrieval_paths_used=[],
                model_used="guardrail",
                query_type=QueryType.PATIENT_FAQ,
                disclaimer="This response was blocked for safety reasons.",
            )

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
        await save_session(session)

        # ── Step 5: Update Long-term Graph Memory ─────────────────────
        all_cuis = set(result.get("linked_cuis", []))
        if result.get("top_graph_cui"):
            all_cuis.add(result["top_graph_cui"])

        if all_cuis:
            from medgraphia.graph.queries import update_user_interests

            u_id = principal.get("id", "anonymous")
            cuis_list = list(all_cuis)
            logger.info("scheduling_interest_update", user_id=u_id, entities=cuis_list)
            asyncio.create_task(update_user_interests(user_id=u_id, cuis=cuis_list))

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
    Stream the answer token-by-token as Server-Sent Events."""
    session = await create_or_get_session(body.session_id)
    request_id: str = request.state.request_id if hasattr(request.state, "request_id") else ""
    langfuse = get_langfuse_client()

    # Auto-detect language if UNKNOWN
    language = body.language
    if language == Language.UNKNOWN:
        language = Language.detect(body.message)

    logger.info("query_received", language=language.value, stream=True)

    async def _event_stream() -> AsyncIterator[str]:
        with langfuse.trace(
            "chat_stream",
            session_id=session.session_id,
            user_id=principal.get("id", "anonymous"),
            input=body.message,
            metadata={"language": language.value, "request_id": request_id},
            tags=["stream"],
        ) as trace:
            # ── Step 0: Input Safety Check ──────────────────────────────────
            guard = await _get_guard()
            input_safety = await guard.check_input(body.message, history=session.messages)
            if not input_safety.is_safe:
                yield _sse(
                    {
                        "type": "error",
                        "detail": f"Blocked by safety guardrails: {input_safety.category}",
                    }
                )
                return

            # ── Step 1: Retrieval ───────────────────────────────────────────
            retrieval = await _get_retrieval()
            generation = await _get_generation()

            with trace.span("retrieval", input=body.message) as span:
                try:
                    reranked = await retrieval.execute(
                        query=body.message,
                        history=session.messages,
                        language=language,
                        user_id=principal.get("id", "anonymous"),
                    )
                except Exception as exc:
                    logger.error("stream_retrieval_failed", error=str(exc))
                    yield _sse({"type": "error", "detail": "Retrieval failed."})
                    return

                items = getattr(reranked, "items", [])
                query_type: QueryType = getattr(reranked, "query_type", QueryType.PATIENT_FAQ)
                complexity_tier = getattr(reranked, "complexity_tier", None)
                retrieval_paths = list({item.source.value for item in items})
                span.end(output=f"{len(items)} items")

            # ── Chitchat short-circuit ──────────────────────────────────────
            # No medical signal (e.g. a greeting) — skip the GEPA-tuned
            # generator and stream a static reply instead.
            if getattr(reranked, "is_chitchat", False):
                from medgraphia.prompts import get_chitchat_message

                chitchat_text = get_chitchat_message(language)
                yield _sse({"type": "chunk", "content": chitchat_text})

                session.messages.append(
                    Message(session_id=session.session_id, role="user", content=body.message)
                )
                session.messages.append(
                    Message(
                        session_id=session.session_id,
                        role="assistant",
                        content=chitchat_text,
                        citations=[],
                        model_used="static",
                        retrieval_paths_used=[],
                    )
                )
                await save_session(session)

                yield _sse({"type": "citations", "citations": []})
                yield _sse(
                    {
                        "type": "done",
                        "session_id": session.session_id,
                        "model_used": "static",
                        "query_type": query_type.value,
                        "disclaimer": "",
                    }
                )
                trace.update(output=chitchat_text)
                return

            # ── No-evidence short-circuit ────────────────────────────────────
            # Reranker fallback content scored below the noise floor — skip the
            # LLM call, the context is already known to be useless.
            if getattr(reranked, "no_evidence", False):
                from medgraphia.prompts import get_disclaimer, get_no_info_message

                no_info_text = get_no_info_message(language)
                yield _sse({"type": "chunk", "content": no_info_text})

                session.messages.append(
                    Message(session_id=session.session_id, role="user", content=body.message)
                )
                session.messages.append(
                    Message(
                        session_id=session.session_id,
                        role="assistant",
                        content=no_info_text,
                        citations=[],
                        model_used="static",
                        retrieval_paths_used=retrieval_paths,
                    )
                )
                await save_session(session)

                yield _sse({"type": "citations", "citations": []})
                yield _sse(
                    {
                        "type": "done",
                        "session_id": session.session_id,
                        "model_used": "static",
                        "query_type": query_type.value,
                        "disclaimer": get_disclaimer(language),
                    }
                )
                trace.update(output=no_info_text)
                return

            # ── Step 2 & 3: Stream LLM tokens via Unified Generation Pipeline ──
            accumulated: list[str] = []

            # Get disclaimer and routing info (pass tier so routing is consistent)
            disclaimer = generation.get_streaming_components(query_type, language)["disclaimer"]
            _, routing = (await _get_generation()).router.route(
                query_type, language, tier=complexity_tier
            )

            with trace.span("generation_stream", input=body.message):
                try:
                    async for token in generation.generate_stream(
                        question=body.message,
                        query_type=query_type,
                        retrieved_items=items,
                        history=session.messages,
                        language=language,
                        complexity_tier=complexity_tier,
                        entity_labels=getattr(reranked, "entity_labels", {}),
                        unlinked_mentions=getattr(reranked, "unlinked_mentions", []),
                    ):
                        accumulated.append(token)
                        yield _sse({"type": "chunk", "content": token})
                except Exception as exc:
                    logger.error("stream_generation_failed", error=str(exc))
                    yield _sse({"type": "error", "detail": "Generation failed."})
                    return

            full_text = "".join(accumulated)

            # ── Step 4: Finalise (Citations & History) ──────────────────────
            # ── Step 4a: Output Safety Check ────────────────────────────────
            output_safety = await guard.check_output(
                body.message, full_text, history=session.messages
            )
            if not output_safety.is_safe:
                yield _sse(
                    {
                        "type": "moderated",
                        "category": output_safety.category,
                        "detail": "Response redacted due to safety violation.",
                    }
                )
                return

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
            await save_session(session)

            # ── Step 5: Update Long-term Graph Memory ─────────────────────
            # 1. CUIs explicitly linked from the query (high signal)
            all_cuis = set(getattr(reranked, "linked_cuis", []))

            # 2. Implicit interest: Top-ranked graph node if it appears in top-10
            from medgraphia.retrieval.fusion import RetrievalSource

            top_items = getattr(reranked, "items", [])[:10]
            for item in top_items:
                if item.source == RetrievalSource.GRAPH:
                    cui = item.metadata.get("entity_cui")
                    if cui:
                        all_cuis.add(cui)
                        break

            if all_cuis:
                from medgraphia.graph.queries import update_user_interests

                u_id = principal.get("id", "anonymous")
                cuis_list = list(all_cuis)
                logger.info("scheduling_interest_update", user_id=u_id, entities=cuis_list)
                asyncio.create_task(update_user_interests(user_id=u_id, cuis=cuis_list))
            else:
                logger.info("no_linked_entities_for_memory")

            # ── Step 6: Send Metadata Events ────────────────────────────────
            yield _sse(
                {
                    "type": "citations",
                    "citations": [c.model_dump() for c in citation_result.citations],
                }
            )
            yield _sse(
                {
                    "type": "done",
                    "session_id": session.session_id,
                    "model_used": routing.model_name,
                    "query_type": query_type.value,
                    "disclaimer": disclaimer,
                }
            )

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
    history: list[Message] | None = None,
    user_id: str = "anonymous",
) -> dict:
    """Execute retrieve → generate and return a unified result dict."""
    retrieval = await _get_retrieval()
    generation = await _get_generation()

    # ── Retrieval ─────────────────────────────────────────────────────────────
    with trace.span("retrieval", input=query) as span:
        reranked = await retrieval.execute(
            query=query,
            history=history,
            language=language,
            user_id=user_id,
        )
        items = getattr(reranked, "items", [])
        query_type: QueryType = getattr(reranked, "query_type", QueryType.PATIENT_FAQ)
        complexity_tier = getattr(reranked, "complexity_tier", None)
        retrieval_paths = list({item.source.value for item in items})
        span.end(output=f"{len(items)} items")

    # ── Chitchat short-circuit ──────────────────────────────────────────────
    # No medical signal in the query (e.g. a greeting) — skip the GEPA-tuned
    # generator entirely rather than force a grounded answer from no context.
    if getattr(reranked, "is_chitchat", False):
        from medgraphia.prompts import get_chitchat_message

        return {
            "answer": get_chitchat_message(language),
            "citations": [],
            "retrieval_paths": [],
            "model_used": "static",
            "query_type": query_type,
            "disclaimer": "",
            "linked_cuis": [],
            "unlinked_mentions": [],
            "top_graph_cui": None,
        }

    # ── No-evidence short-circuit ────────────────────────────────────────────
    # Reranker fallback content scored below the noise floor — the retrieved
    # context is known-useless, so skip the LLM call instead of burning one on
    # a context the reranker already knows won't produce a grounded answer.
    if getattr(reranked, "no_evidence", False):
        from medgraphia.prompts import get_disclaimer, get_no_info_message

        return {
            "answer": get_no_info_message(language),
            "citations": [],
            "retrieval_paths": [],
            "model_used": "static",
            "query_type": query_type,
            "disclaimer": get_disclaimer(language),
            "linked_cuis": [],
            "unlinked_mentions": [],
            "top_graph_cui": None,
        }

    # ── Generation ────────────────────────────────────────────────────────────
    with trace.span("generation", input=query) as span:
        gen_result = await generation.generate(
            question=query,
            query_type=query_type,
            retrieved_items=items,
            history=history,
            language=language,
            complexity_tier=complexity_tier,
            entity_labels=getattr(reranked, "entity_labels", {}),
            unlinked_mentions=getattr(reranked, "unlinked_mentions", []),
        )
        model_used = gen_result.routing.model_name if gen_result.routing else ""
        span.end(output=gen_result.answer[:200])

    # ── Step 3: Extract implicit interest (Top graph node in top 10) ──────────
    from medgraphia.retrieval.fusion import RetrievalSource

    top_graph_cui = None
    for item in items[:10]:
        if item.source == RetrievalSource.GRAPH:
            top_graph_cui = item.metadata.get("entity_cui")
            if top_graph_cui:
                break

    return {
        "answer": gen_result.answer,
        "citations": gen_result.citations,
        "retrieval_paths": retrieval_paths,
        "model_used": model_used,
        "query_type": query_type,
        "disclaimer": gen_result.disclaimer,
        "linked_cuis": getattr(reranked, "linked_cuis", []),
        "unlinked_mentions": getattr(reranked, "unlinked_mentions", []),
        "top_graph_cui": top_graph_cui,
    }


def _sse(data: dict) -> str:
    """Encode a dict as a single SSE data frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
