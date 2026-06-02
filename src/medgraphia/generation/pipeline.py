"""
Generation Pipeline — the final stage of the MedGraphia Q&A flow.

Orchestrates:
  1. Context preparation (numbering retrieval hits)
  2. Model routing (tier-based selection)
  3. LLM prediction (structured output with citations)
  4. Citation resolution (mapping [N] to metadata)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from medgraphia.domain.base import Language, QueryType
from medgraphia.domain.chat import Citation
from medgraphia.generation.citation import build_numbered_context, inject_citations
from medgraphia.generation.llm_router import LLMRouter, RoutingDecision
from medgraphia.generation.prompts import PromptRegistry, MedicalAnswer
from medgraphia.logger import get_logger
from medgraphia.retrieval.fusion import FusedItem

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """The final payload delivered to the user or API endpoint."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    disclaimer: str = ""
    routing: RoutingDecision | None = None
    latency_ms: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    
    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0


# ---------------------------------------------------------------------------
# GenerationPipeline
# ---------------------------------------------------------------------------

class GenerationPipeline:
    """
    Orchestrates the transition from retrieved evidence to a final answer.
    """

    def __init__(
        self,
        router: LLMRouter | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        self._router = router or LLMRouter.from_settings()
        self._prompts = prompts or PromptRegistry()

    async def generate(
        self,
        question: str,
        query_type: QueryType,
        retrieved_items: list[FusedItem],
        language: Language = Language.EN,
    ) -> GenerationResult:
        """
        Produce a cited medical answer from a retrieved context.

        Args:
            question:         The user's original query.
            query_type:       Identified intent (from retrieval router).
            retrieved_items:  Ranked and fused items (from retrieval pipeline).
            language:         Desired response language.

        Returns:
            GenerationResult containing the cited answer and metadata.
        """
        t0 = time.monotonic()
        
        # 1. Prepare numbered context [1] text... [2] text...
        context_str = build_numbered_context(retrieved_items)
        
        # 2. Select the optimal model/gateway for this specific task
        gateway, routing = self._router.route(query_type, language)
        
        # 3. Predict the answer using the appropriate prompt template
        predictor = self._prompts.get(query_type)
        
        logger.info(
            "generation_started",
            query_type=query_type.value,
            language=language.value,
            context_items=len(retrieved_items),
            model=routing.model_name
        )
        
        try:
            # Call the LLM
            # MedicalPredictor handles the JSON parsing and fallback logic
            prediction: MedicalAnswer = await predictor.predict(
                context=context_str,
                question=question,
                language=language,
                gateway=gateway,
            )
            
            # 4. Resolve the [N] citations into full objects with metadata
            citation_result = inject_citations(
                answer_text=prediction.answer,
                context_items=retrieved_items,
                explicit_citations=prediction.citations
            )
            
            latency = (time.monotonic() - t0) * 1000.0
            
            logger.info(
                "generation_completed",
                latency_ms=f"{latency:.0f}",
                citations=len(citation_result.citations),
                unresolved=len(citation_result.unresolved)
            )
            
            return GenerationResult(
                answer=prediction.answer,
                citations=citation_result.citations,
                disclaimer=prediction.disclaimer,
                routing=routing,
                latency_ms=latency,
            )

        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000.0
            logger.error("generation_pipeline_failed", error=str(exc))
            return GenerationResult(
                answer=f"I encountered an error while generating your answer: {str(exc)}",
                latency_ms=latency
            )

    @classmethod
    def from_settings(cls) -> "GenerationPipeline":
        """Factory method to build a pipeline with default settings."""
        return cls()
