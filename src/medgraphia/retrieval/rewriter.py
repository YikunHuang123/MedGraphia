"""
Query Rewriter — condenses conversation history and the latest message into a standalone query.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from medgraphia.domain.base import Language
from medgraphia.domain.chat import Message
from medgraphia.logger import get_logger
from medgraphia.llm.gateway import LLMProvider

logger = get_logger(__name__)

class RewrittenQuery(BaseModel):
    """Structured output for the query rewriter."""
    is_standalone: bool = Field(
        ..., 
        description="Whether the original message was already standalone and didn't need rewriting."
    )
    rewritten_query: str = Field(
        ..., 
        description="The standalone, context-complete medical query."
    )

class QueryRewriter:
    """
    Uses a fast LLM to rewrite user queries based on conversation history.
    """

    def __init__(self) -> None:
        from medgraphia.llm.client import get_model
        from medgraphia.config import get_settings
        cfg = get_settings()
        
        # Use the configured "small" tier (most cost-effective)
        self._model = get_model(
            model_override=cfg.llm_small_model,
            provider_override=cfg.llm_small_provider
        )

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
        Rewrite the query if it depends on history.
        """
        if not history:
            return query

        logger.info("query_rewrite_started", history_len=len(history))

        # Format history for the prompt (last 3-5 turns)
        history_str = ""
        for m in history[-5:]:
            role = "User" if m.role == "user" else "Assistant"
            history_str += f"{role}: {m.content}\n"

        system_prompt = (
            "You are a medical query refiner. Your task is to analyze the conversation history "
            "and the latest user message. If the latest message refers to previous topics or "
            "uses pronouns (e.g., 'it', 'this drug', 'the symptoms'), rewrite it into a "
            "single, standalone medical question that can be understood without context.\n\n"
            "Rules:\n"
            "1. Maintain the original intent and medical terminology.\n"
            "2. If the message is already standalone, return it as is.\n"
            "3. Respond ONLY with the JSON matching the schema."
        )

        user_prompt = (
            f"Conversation History:\n{history_str}\n"
            f"Latest Message: {query}\n\n"
            f"Output JSON: {{\"is_standalone\": bool, \"rewritten_query\": \"...\"}}"
        )

        try:
            # Using pydantic-ai for structured output
            from pydantic_ai import Agent
            agent = Agent(self._model, output_type=RewrittenQuery, system_prompt=system_prompt)
            result = await agent.run(user_prompt)
            
            rewritten = result.data.rewritten_query
            logger.info(
                "query_rewrite_completed", 
                original=query, 
                rewritten=rewritten,
                is_standalone=result.data.is_standalone
            )
            return rewritten
        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            return query
