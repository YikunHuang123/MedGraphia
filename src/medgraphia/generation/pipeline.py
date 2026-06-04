"""
LLM generation pipeline (architecture doc §3).

Selects the best model and prompt based on query intent,
runs the generation, and handles post-processing (citations).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from medgraphia.domain.base import Language, QueryType
from medgraphia.domain.chat import Message
from medgraphia.generation.citation import CitationResult, inject_citations
from medgraphia.generation.llm_router import LLMRouter, RoutingDecision
from medgraphia.logger import get_logger
from medgraphia.prompts import (
    GenerateClinicalAnswer, 
    MedicalAnswer, 
    get_disclaimer, 
    get_no_info_message, 
    get_system_prompt
)
from pydantic import BaseModel
from medgraphia.retrieval.reranker import RerankedResult

logger = get_logger(__name__)


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
    ) -> GenerationResult:
        """
        Run the full generation flow:
          1. Build numbered context from retrieved items
          2. Route to the optimal model
          3. Predict the answer using the DSPy signature
          4. Post-process citations
        """
        from medgraphia.generation.citation import build_numbered_context

        # 1. Prepare context
        context_str = build_numbered_context(retrieved_items)

        # 2. Route
        gateway, routing = self.router.route(query_type, language)

        # 3. Predict using DSPy
        import dspy
        from medgraphia.llm.dspy_setup import get_lm, get_program
        
        # Get the DSPy LM using the EXACT provider and model selected by the router
        lm = get_lm(
            task="default", 
            provider_override=routing.provider.value, 
            model_override=routing.model_name
        )
        
        lang_names = {Language.EN: "English", Language.ZH: "Chinese", Language.DE: "German"}
        target_lang = lang_names.get(language, "English")

        history_str = "No history."
        if history:
            history_str = ""
            for m in history[-5:]:
                role = "User" if m.role == "user" else "Assistant"
                history_str += f"{role}: {m.content}\n"

        # Log the context that will be passed to the LLM (stream path)
        truncated_context = []
        for i, item in enumerate(retrieved_items[:5]):
            text_preview = item.text.replace('\n', ' ').strip()[:50]
            truncated_context.append(f"[{i+1}] {text_preview}...")
        logger.info(
            "llm_context_preview", 
            items_count=len(retrieved_items), 
            preview=" | ".join(truncated_context) if truncated_context else "EMPTY"
        )

        try:
            # Get the program *before* entering the context, 
            # ensuring it's available even if the context block has issues
            program = get_program("generator")
            
            with dspy.context(lm=lm):
                prediction = program(
                    system_instruction=get_system_prompt(query_type, language),
                    context=context_str,
                    history=history_str,
                    question=question,
                    target_language=target_lang,
                    no_info_message=get_no_info_message(language)
                )
                ans_data: MedicalAnswer = prediction.result
        except Exception as exc:
            logger.error("generation_failed", error=str(exc))
            ans_data = MedicalAnswer(
                answer="I encountered an internal error generating the response.",
                citations=[],
                disclaimer=get_disclaimer(language)
            )

        # 4. Finalise citations (ensure they match the context)
        # Note: DSPy might have already cited, but we use inject_citations to be 100% sure
        citation_result = inject_citations(ans_data.answer, retrieved_items)
        
        return GenerationResult(
            answer=ans_data.answer,
            citations=citation_result.citations,
            disclaimer=ans_data.disclaimer or get_disclaimer(language),
            routing=routing,
        )

    async def generate_stream(
        self,
        question: str,
        query_type: QueryType,
        retrieved_items: list[Any],
        history: list[Message] | None = None,
        language: Language = Language.EN,
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
        gateway, routing = self.router.route(query_type, language)

        import dspy
        from medgraphia.llm.dspy_setup import get_lm, get_program
        
        lm = get_lm(
            task="default", 
            provider_override=routing.provider.value, 
            model_override=routing.model_name
        )
        
        lang_names = {Language.EN: "English", Language.ZH: "Chinese", Language.DE: "German"}
        target_lang = lang_names.get(language, "English")

        history_str = "No history."
        if history:
            history_str = ""
            for m in history[-5:]:
                role = "User" if m.role == "user" else "Assistant"
                history_str += f"{role}: {m.content}\n"

        # Log the context that will be passed to the LLM
        truncated_context = []
        for i, item in enumerate(retrieved_items[:5]):
            text_preview = item.text.replace('\n', ' ').strip()[:50]
            truncated_context.append(f"[{i+1}] {text_preview}...")
        logger.info(
            "llm_context_preview", 
            items_count=len(retrieved_items), 
            preview=" | ".join(truncated_context) if truncated_context else "EMPTY"
        )

        try:
            program = get_program("generator")
            with dspy.context(lm=lm):
                # Use the compiled program (which includes the optimized few-shot demos)
                prediction = program(
                    system_instruction=get_system_prompt(query_type, language),
                    context=context_str,
                    history=history_str,
                    question=question,
                    target_language=target_lang,
                    no_info_message=get_no_info_message(language)
                )
                ans_text = prediction.result.answer
                
                # Stream the answer character by character (or chunk) to satisfy UI expectations
                # In a real high-perf scenario, we'd use dspy's underlying LM for true streaming
                # but to guarantee Compiled Rigor, we follow the compiled path.
                chunk_size = 4
                for i in range(0, len(ans_text), chunk_size):
                    yield ans_text[i:i+chunk_size]
                    await asyncio.sleep(0.01) # Small delay for UI feel

        except Exception as exc:
            logger.error("stream_generation_failed", error=str(exc))
            yield "I encountered an error during streaming generation."

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

