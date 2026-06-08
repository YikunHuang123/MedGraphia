"""
DSPy program for medical query translation.
"""

from __future__ import annotations

import dspy

from medgraphia.logger import get_logger

logger = get_logger(__name__)


class TranslateMedicalQuery(dspy.Signature):
    """Translate a medical query from source_lang to target_lang.
    Preserve all clinical and pharmacological terminology precisely.
    Output ONLY the translated text — no explanation, no prefix."""

    source_text: str = dspy.InputField(desc="Medical query to translate")
    source_lang: str = dspy.InputField(desc="Language of source_text")
    target_lang: str = dspy.InputField(desc="Language to translate into")
    translated_text: str = dspy.OutputField(desc="Translation only, no additional text")


class TranslatorModule(dspy.Module):
    """
    Translates medical queries while preserving terminology.
    """

    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(TranslateMedicalQuery)

    def forward(self, source_text, source_lang, target_lang):
        return self.prog(source_text=source_text, source_lang=source_lang, target_lang=target_lang)


def get_translator() -> TranslatorModule:
    """
    Get the translator program.
    """
    return TranslatorModule()
