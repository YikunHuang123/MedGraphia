"""
DSPy program for clinical answer generation.
"""

from __future__ import annotations

from pathlib import Path

import dspy

from medgraphia.logger import get_logger
from medgraphia.prompts import GenerateClinicalAnswer

logger = get_logger(__name__)


class GeneratorModule(dspy.Module):
    """
    Synthesizes medical context into a structured clinical answer.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(GenerateClinicalAnswer)

    def forward(
        self, system_instruction, context, history, question, target_language, no_info_message
    ):
        return self.prog(
            system_instruction=system_instruction,
            context=context,
            history=history,
            question=question,
            target_language=target_language,
            no_info_message=no_info_message,
        )


def get_generator() -> GeneratorModule:
    """
    Load the (potentially compiled) generator program.
    """
    module = GeneratorModule()
    path = Path("data/dspy/generator_compiled.json")

    if path.exists():
        try:
            module.load(str(path))
            logger.info("dspy_generator_loaded_compiled", path=str(path))
        except Exception as exc:
            logger.warning("dspy_generator_load_failed", error=str(exc))
    else:
        logger.info("dspy_generator_using_uncompiled")

    return module
