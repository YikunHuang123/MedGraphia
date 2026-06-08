"""
DSPy program for medical relation extraction.
"""

from __future__ import annotations

from pathlib import Path

import dspy

from medgraphia.logger import get_logger
from medgraphia.prompts import ExtractMedicalRelations

logger = get_logger(__name__)


class ExtractorModule(dspy.Module):
    """
    Extracts medical relations between entities from raw text.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(ExtractMedicalRelations)

    def forward(self, text_content, entities, allowed_relations):
        return self.prog(
            text_content=text_content, entities=entities, allowed_relations=allowed_relations
        )


def get_extractor() -> ExtractorModule:
    """
    Load the (potentially compiled) extractor program.
    """
    module = ExtractorModule()
    path = Path("data/dspy/extractor_compiled.json")

    if path.exists():
        try:
            module.load(str(path))
            logger.info("dspy_extractor_loaded_compiled", path=str(path))
        except Exception as exc:
            logger.warning("dspy_extractor_load_failed", error=str(exc))
    else:
        logger.info("dspy_extractor_using_uncompiled")

    return module
