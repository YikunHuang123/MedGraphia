"""
DSPy Predictor prompt modules.
"""

from __future__ import annotations

import dspy
from pydantic import BaseModel, Field, field_validator


class MedicalAnswer(BaseModel):
    answer: str = Field(..., description="Response text in the requested language")
    citations: list[int] = Field(
        default_factory=list,
        description="1-indexed citation numbers corresponding to [N] markers in the answer",
    )
    disclaimer: str = Field(
        default="",
        description="Mandatory safety disclaimer; required for clinical/drug/multihop scenarios",
    )

    @field_validator("citations", mode="before")
    @classmethod
    def coerce_citations(cls, v: object) -> list[int]:
        # Small models sometimes output ["[1]", "[2]"] instead of [1, 2].
        # Strip brackets and convert to int, silently dropping unparseable values.
        if not isinstance(v, list):
            return []
        result: list[int] = []
        for item in v:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str):
                cleaned = item.strip().strip("[]")
                if cleaned.isdigit():
                    result.append(int(cleaned))
        return result


class GenerateClinicalAnswer(dspy.Signature):
    """Answer the medical question in the requested language using ONLY the provided database.
    Context paragraphs may be written in multiple languages (English, Chinese, German).
    You MUST read and synthesize relevant information from ALL context paragraphs regardless
    of the language they are written in — translate and integrate cross-language evidence.
    Respond exclusively in target_language.
    Cite sources using [N] inline.
    If context paragraphs discuss the queried topic through clinical notes, differential diagnoses,
    case reports, pharmacological data, or pathological descriptions — even without a formal
    encyclopedic definition — synthesize whatever IS available and cite sources; do not refuse.
    Use no_info_message ONLY when ALL context paragraphs are entirely off-topic (e.g., purely
    administrative records, or medical conditions completely unrelated to the question).
    Never refuse simply because context is clinical rather than definitional, or because it is
    written in a different language than the question."""

    system_instruction: str = dspy.InputField(desc="Core role and persona for the assistant")
    context: str = dspy.InputField(desc="Numbered medical context paragraphs")
    history: str = dspy.InputField(desc="Recent conversation history")
    question: str = dspy.InputField(desc="The current medical question")
    target_language: str = dspy.InputField(desc="The language to respond in")
    no_info_message: str = dspy.InputField(desc="The message to show if no info is found")

    result: MedicalAnswer = dspy.OutputField()
