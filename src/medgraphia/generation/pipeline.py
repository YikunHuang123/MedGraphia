"""
LLM generation pipeline.

Selects the best model and prompt based on query intent,
runs the generation, and handles post-processing (citations).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from medgraphia.domain.base import Language, QueryType
from medgraphia.domain.chat import Message
from medgraphia.generation.citation import inject_citations
from medgraphia.generation.llm_router import LLMRouter, ModelTier, RoutingDecision
from medgraphia.logger import get_logger
from medgraphia.prompts import (
    MedicalAnswer,
    get_disclaimer,
    get_no_info_message,
    get_system_prompt,
)

logger = get_logger(__name__)


def _build_memory_context(qa_memories: list[Any]) -> str:
    """Format QA memories with a relative-time label so the generator can reason about recency itself."""
    if not qa_memories:
        return ""

    from datetime import datetime

    lines = []
    now = datetime.now()
    for m in qa_memories:
        try:
            created = datetime.fromisoformat(m.created_at)
            days_ago = (now - created).days
            if days_ago < 1:
                when = "today"
            elif days_ago < 30:
                when = f"{days_ago}d ago"
            else:
                when = f"{days_ago // 30}mo ago"
        except (ValueError, TypeError):
            when = "previously"
        lines.append(f"[{when}] User asked: {m.question}\n[{when}] You answered: {m.answer}")

    return "\n\n[Relevant past conversation with this user]\n" + "\n\n".join(lines)


class GenerationPipeline:
    """
    Final stage of GraphRAG: Context + History + Query -> Cited Answer.
    """

    def __init__(
        self,
        router: LLMRouter | None = None,
    ) -> None:
        self.router = router or LLMRouter.from_settings()

    @classmethod
    def from_settings(cls) -> GenerationPipeline:
        return cls()

    async def generate(
        self,
        question: str,
        query_type: QueryType,
        retrieved_items: list[Any],
        history: list[Message] | None = None,
        language: Language = Language.EN,
        complexity_tier: ModelTier | None = None,
        entity_labels: dict[str, str] | None = None,
        unlinked_mentions: list[str] | None = None,
        qa_memories: list[Any] | None = None,
    ) -> GenerationResult:
        """
        Run the full generation flow:
          1. Build numbered context from retrieved items
          2. Route to the optimal model (DSPy tier preferred, static table as fallback)
          2.5. Agentic gap completion — may augment context with new evidence
          3. Predict the answer using the DSPy signature
          4. Post-process citations
        """
        from medgraphia.generation.citation import build_numbered_context

        # 1. Prepare context
        context_str = build_numbered_context(retrieved_items)

        # 2. Route — pass DSPy tier when available
        gateway, routing = self.router.route(query_type, language, tier=complexity_tier)

        # 3. Predict using DSPy
        import dspy

        from medgraphia.llm.dspy_setup import get_lm
        from medgraphia.programs.generator import get_generator

        # Get the DSPy LM using the EXACT provider and model selected by the router
        lm = get_lm(
            task="default",
            provider_override=routing.provider.value,
            model_override=routing.model_name,
        )

        # 2.5. Agentic gap completion — same lm as final generation, runs before it
        evidence_lines, new_chunks = await self._maybe_complete_gaps(
            question, context_str, entity_labels or {}, unlinked_mentions or [], lm
        )
        if new_chunks:
            # Rebuild the numbered context so newly-fetched passages get a
            # real [N] citation number the model can reference, instead of
            # being appended as unnumbered raw text.
            from medgraphia.retrieval.fusion import chunk_to_fused_item

            retrieved_items.extend(chunk_to_fused_item(c) for c in new_chunks)
            context_str = build_numbered_context(retrieved_items)
        if evidence_lines:
            context_str += "\n\n[Additional evidence found for this query]\n" + "\n".join(evidence_lines)
        context_str += _build_memory_context(qa_memories or [])

        target_lang = language.full_name if language else Language.EN.full_name

        from medgraphia.config import get_settings

        recent_turns = get_settings().qa_memory_recent_turns
        history_str = "No history."
        if history:
            history_str = ""
            for m in history[-recent_turns:]:
                role = "User" if m.role == "user" else "Assistant"
                history_str += f"{role}: {m.content}\n"

        # Log the context that will be passed to the LLM (stream path)
        truncated_context = []
        for i, item in enumerate(retrieved_items[:5]):
            text_preview = item.text.replace("\n", " ").strip()[:50]
            truncated_context.append(f"[{i + 1}] {text_preview}...")
        logger.info(
            "llm_context_preview",
            items_count=len(retrieved_items),
            preview=" | ".join(truncated_context) if truncated_context else "EMPTY",
        )

        import time

        start_time = time.perf_counter()
        try:
            # Get the program *before* entering the context
            program = get_generator()

            with dspy.context(lm=lm):
                prediction = await dspy.asyncify(program)(
                    system_instruction=get_system_prompt(query_type, language),
                    context=context_str,
                    history=history_str,
                    question=question,
                    target_language=target_lang,
                    no_info_message=get_no_info_message(language),
                )
                ans_data: MedicalAnswer = prediction.result

                # ── Auto-Stitching: Ensure JSON citations are reflected in text ──
                # This fixes "Format Drift" where LLMs omit [N] markers when a
                # separate citations field is present in the schema.
                if ans_data.citations:
                    missing = [c for c in ans_data.citations if f"[{c}]" not in ans_data.answer]
                    if missing:
                        ans_data.answer += " " + "".join([f"[{c}]" for c in missing])

        except Exception as exc:
            logger.error("generation_failed", error=str(exc))
            ans_data = MedicalAnswer(
                answer="I encountered an internal error generating the response.",
                citations=[],
                disclaimer=get_disclaimer(language),
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 4. Finalise citations (ensure they match the context)
        # Note: We pass ans_data.citations as explicit_citations for dual-layered resolution
        citation_result = inject_citations(
            ans_data.answer, retrieved_items, explicit_citations=ans_data.citations
        )

        return GenerationResult(
            answer=ans_data.answer,
            citations=citation_result.citations,
            disclaimer=ans_data.disclaimer or get_disclaimer(language),
            routing=routing,
            latency_ms=latency_ms,
        )

    async def generate_stream(
        self,
        question: str,
        query_type: QueryType,
        retrieved_items: list[Any],
        history: list[Message] | None = None,
        language: Language = Language.EN,
        complexity_tier: ModelTier | None = None,
        entity_labels: dict[str, str] | None = None,
        unlinked_mentions: list[str] | None = None,
        qa_memories: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming version of generate().
        NOTE: Since DSPy's compiled programs don't natively stream individual tokens
        while preserving CoT reasoning in a single call, we simulate streaming by
        first running the compiled logic and then streaming the result, OR by
        providing the compiled system prompt to the gateway.

        To preserve the exact clinical rigor of the compiled program, we run
        prediction and then stream the answer field.
        """
        from medgraphia.generation.citation import build_numbered_context

        context_str = build_numbered_context(retrieved_items)
        gateway, routing = self.router.route(query_type, language, tier=complexity_tier)

        import dspy

        from medgraphia.llm.dspy_setup import get_lm
        from medgraphia.programs.generator import get_generator

        lm = get_lm(
            task="default",
            provider_override=routing.provider.value,
            model_override=routing.model_name,
        )

        evidence_lines, new_chunks = await self._maybe_complete_gaps(
            question, context_str, entity_labels or {}, unlinked_mentions or [], lm
        )
        if new_chunks:
            from medgraphia.retrieval.fusion import chunk_to_fused_item

            retrieved_items.extend(chunk_to_fused_item(c) for c in new_chunks)
            context_str = build_numbered_context(retrieved_items)
        if evidence_lines:
            context_str += "\n\n[Additional evidence found for this query]\n" + "\n".join(evidence_lines)
        context_str += _build_memory_context(qa_memories or [])

        target_lang = language.full_name if language else Language.EN.full_name

        from medgraphia.config import get_settings

        recent_turns = get_settings().qa_memory_recent_turns
        history_str = "No history."
        if history:
            history_str = ""
            for m in history[-recent_turns:]:
                role = "User" if m.role == "user" else "Assistant"
                history_str += f"{role}: {m.content}\n"

        # Log the context that will be passed to the LLM
        truncated_context = []
        for i, item in enumerate(retrieved_items[:5]):
            text_preview = item.text.replace("\n", " ").strip()[:50]
            truncated_context.append(f"[{i + 1}] {text_preview}...")
        logger.info(
            "llm_context_preview",
            items_count=len(retrieved_items),
            preview=" | ".join(truncated_context) if truncated_context else "EMPTY",
        )

        try:
            program = get_generator()
            with dspy.context(lm=lm):
                # Use the compiled program (which includes the optimized few-shot demos)
                prediction = await dspy.asyncify(program)(
                    system_instruction=get_system_prompt(query_type, language),
                    context=context_str,
                    history=history_str,
                    question=question,
                    target_language=target_lang,
                    no_info_message=get_no_info_message(language),
                )
                ans_text = prediction.result.answer
                explicit_cits = prediction.result.citations

                # ── Auto-Stitching: Ensure JSON citations are reflected in text ──
                if explicit_cits:
                    missing = [c for c in explicit_cits if f"[{c}]" not in ans_text]
                    if missing:
                        ans_text += " " + "".join([f"[{c}]" for c in missing])

                chunk_size = 4
                for i in range(0, len(ans_text), chunk_size):
                    yield ans_text[i : i + chunk_size]
                    await asyncio.sleep(0.01)

        except Exception as exc:
            logger.error("stream_generation_failed", error=str(exc))
            yield "I encountered an error during streaming generation."

    async def _maybe_complete_gaps(
        self,
        question: str,
        context_str: str,
        entity_labels: dict[str, str],
        unlinked_mentions: list[str],
        lm: Any,
    ) -> tuple[list[str], list[Any]]:
        """
        Let an agent decide whether the context is missing a relation
        between two entities mentioned in the question, and if so fetch new
        evidence before the final answer is generated.

        Returns (evidence_status_lines, new_chunks). The caller is
        responsible for merging new_chunks into retrieved_items and
        rebuilding the numbered context so newly-fetched passages get a
        real [N] citation number instead of being appended as raw text.
        """
        from medgraphia.config import get_settings

        cfg = get_settings()
        if not cfg.gap_completion_enabled:
            return [], []

        candidate_labels = list(entity_labels.values()) + unlinked_mentions
        if len(candidate_labels) < 2:
            return [], []

        from medgraphia.generation.agentic_completion import run_gap_completion

        cui_map = {label: cui for cui, label in entity_labels.items()}
        evidence, new_chunks = await run_gap_completion(
            question=question,
            context=context_str,
            entity_labels=candidate_labels,
            entity_cui_map=cui_map,
            lm=lm,
            max_tool_calls=cfg.gap_completion_max_tool_calls,
        )
        if evidence:
            logger.info("gap_completion_applied", count=len(evidence))
        return evidence, new_chunks

    def get_streaming_components(self, query_type: QueryType, language: Language) -> dict[str, str]:
        """Return the system prompt and disclaimer for streaming."""
        return {
            "system_prompt": get_system_prompt(query_type, language),
            "disclaimer": get_disclaimer(language),
        }


class GenerationResult(BaseModel):
    """Unified result for the generation phase."""

    answer: str
    citations: list[Any]
    disclaimer: str
    routing: RoutingDecision | None = None
    latency_ms: float = 0.0
