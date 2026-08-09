"""
DSPy program for medical query rewriting with complexity-tier assessment.
"""

from __future__ import annotations

from pathlib import Path

import dspy

from medgraphia.domain.base import Language
from medgraphia.generation.llm_router import ModelTier
from medgraphia.logger import get_logger
from medgraphia.prompts import ROUTING_RUBRIC, QueryRewritingWithTier

logger = get_logger(__name__)


class RewriterModule(dspy.Module):
    """
    Condenses conversation history and the latest message into a standalone query,
    and simultaneously scores the query's clinical complexity (SMALL / MEDIUM / LARGE).
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QueryRewritingWithTier)

    def forward(self, history: str, latest_message: str) -> dspy.Prediction:
        prediction = self.prog(
            rubric=ROUTING_RUBRIC,
            history=history,
            latest_message=latest_message,
        )

        tier = _parse_tier(prediction.complexity_tier)
        response_language = _parse_language(prediction.response_language)

        # Expose tier/language as top-level fields on the returned Prediction so
        # callers can access prediction.tier / prediction.response_language
        # without touching the raw strings.
        return dspy.Prediction(
            is_standalone=prediction.is_standalone,
            rewritten_query=prediction.rewritten_query,
            complexity_reasoning=prediction.complexity_reasoning,
            complexity_tier=prediction.complexity_tier,
            tier=tier,
            response_language=response_language,
        )


def _parse_tier(raw: str) -> ModelTier:
    """
    Convert the model's raw output string to a ModelTier enum.

    ModelTier values are lowercase ("small", "medium", "large") but the model
    is instructed to output uppercase ("SMALL", "MEDIUM", "LARGE"), so we
    normalise with .lower() before lookup.

    Any unrecognised value falls back to LARGE.
    """
    clean = raw.strip().lower()
    try:
        return ModelTier(clean)
    except ValueError:
        logger.warning(
            "complexity_tier_parse_failed",
            raw_value=raw,
            fallback=ModelTier.LARGE.value,
        )
        return ModelTier.LARGE


def _parse_language(raw: str) -> Language | None:
    """
    Convert the model's raw response_language output to a Language enum.

    Returns None on any unrecognised value so callers can fall back to the
    already-resolved request language instead of silently forcing English.
    """
    clean = raw.strip().lower()
    mapping = {"english": Language.EN, "chinese": Language.ZH, "german": Language.DE}
    lang = mapping.get(clean)
    if lang is None:
        logger.warning("response_language_parse_failed", raw_value=raw)
    return lang


def get_rewriter() -> RewriterModule:
    """
    Load the (potentially compiled) rewriter program.

    If a compiled JSON file exists it is loaded on top of the fresh module,
    injecting optimised few-shot demonstrations without changing the signature.
    """
    module = RewriterModule()
    path = Path("data/dspy/rewriter_compiled.json")

    if path.exists():
        try:
            module.load(str(path))
            logger.info("dspy_rewriter_loaded_compiled", path=str(path))
        except Exception as exc:
            logger.warning("dspy_rewriter_load_failed", error=str(exc))
    else:
        logger.info("dspy_rewriter_using_uncompiled")

    return module
