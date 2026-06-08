"""
DSPy program for medical community summarization.
"""

from __future__ import annotations

from pathlib import Path

import dspy

from medgraphia.logger import get_logger
from medgraphia.prompts import SummarizeMedicalCommunity

logger = get_logger(__name__)


class SummarizerModule(dspy.Module):
    """
    Generates clinical summaries for medical concept communities.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(SummarizeMedicalCommunity)

    def forward(self, concepts, relations):
        return self.prog(concepts=concepts, relations=relations)


def get_summarizer() -> SummarizerModule:
    """
    Load the (potentially compiled) summarizer program.
    """
    module = SummarizerModule()
    path = Path("data/dspy/summarizer_compiled.json")

    if path.exists():
        try:
            module.load(str(path))
            logger.info("dspy_summarizer_loaded_compiled", path=str(path))
        except Exception as exc:
            logger.warning("dspy_summarizer_load_failed", error=str(exc))
    else:
        logger.info("dspy_summarizer_using_uncompiled")

    return module
