"""
Query Rewriter — condenses conversation history and the latest message into a standalone query.
"""
from __future__ import annotations

from typing import Any
from medgraphia.domain.base import Language
from medgraphia.domain.chat import Message
from medgraphia.logger import get_logger
from medgraphia.prompts import RewriteMedicalQuery, RewrittenQuery

logger = get_logger(__name__)

class QueryRewriter:
    """
    Uses a fast LLM to rewrite user queries based on conversation history.
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
    ) -> str:
        """
        Rewrite the query if it depends on history using DSPy.
        """
        if not history:
            return query

        logger.info("query_rewrite_started", history_len=len(history))

        # Format history for the prompt (last 3-5 turns)
        history_str = ""
        for m in history[-5:]:
            role = "User" if m.role == "user" else "Assistant"
            history_str += f"{role}: {m.content}\n"

        import dspy
        from medgraphia.llm.dspy_setup import get_lm, get_program
        lm = get_lm("rewriter")

        try:
            with dspy.context(lm=lm):
                # Use the compiled program (which might include few-shot demos)
                program = get_program("rewriter")
                prediction = program(history=history_str, latest_message=query)
            
            # The structure depends on how the program was wrapped in dspy_setup.py
            # If program is RewriterModule, it returns the result of self.prog
            rewritten = prediction.result.rewritten_query
            
            # Log the internal CoT reasoning
            logger.info(
                "query_rewrite_completed", 
                original=query, 
                rewritten=rewritten,
                internal_reasoning=getattr(prediction, "reasoning", "N/A"),
                is_standalone=prediction.result.is_standalone
            )
            return rewritten
        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            return query
