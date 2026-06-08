"""
Query Rewriter — condenses conversation history and the latest message into a
standalone query, and simultaneously returns a DSPy-assessed complexity tier.
"""

from __future__ import annotations

from medgraphia.domain.base import Language
from medgraphia.domain.chat import Message
from medgraphia.generation.llm_router import ModelTier
from medgraphia.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """
    Uses a fast LLM (configured via REWRITER_LLM_* env vars) to rewrite user
    queries and assess clinical complexity in a single DSPy ChainOfThought pass.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_settings(cls) -> QueryRewriter:
        return cls()

    async def rewrite(
        self,
        query: str,
        history: list[Message],
        language: Language = Language.EN,
    ) -> tuple[str, ModelTier]:
        """
        Rewrite the query and score its complexity tier.

        Returns
        -------
        (rewritten_query, complexity_tier)
            rewritten_query : standalone, coreference-resolved query string.
            complexity_tier : ModelTier enum (SMALL / MEDIUM / LARGE).
                              Falls back to LARGE on any DSPy error to ensure
                              medical safety is never compromised.

        Note: This method always runs — even when history is empty — because
        complexity assessment depends on the question itself, not the history.
        """
        # Format history (last 5 turns); empty string is valid for the first message
        history_str = ""
        for m in history[-5:]:
            role = "User" if m.role == "user" else "Assistant"
            history_str += f"{role}: {m.content}\n"

        logger.info("query_rewrite_started", has_history=bool(history_str.strip()))

        import dspy

        from medgraphia.llm.dspy_setup import get_lm
        from medgraphia.programs.rewriter import get_rewriter

        lm = get_lm("rewriter")

        try:
            with dspy.context(lm=lm):
                program = get_rewriter()
                prediction = program(history=history_str, latest_message=query)

            rewritten = prediction.rewritten_query
            tier: ModelTier = prediction.tier

            logger.info(
                "query_rewrite_completed",
                original=query,
                rewritten=rewritten,
                complexity_reasoning=getattr(prediction, "complexity_reasoning", "N/A"),
                complexity_tier=tier.value,
            )
            return rewritten, tier

        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            # Graceful degradation: original query + LARGE tier (medical safety first)
            return query, ModelTier.LARGE
