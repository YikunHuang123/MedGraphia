"""
DSPy Predictor prompt modules.
"""
from __future__ import annotations

import dspy
from pydantic import BaseModel, Field
from medgraphia.domain.base import Language, QueryType

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

class GenerateClinicalAnswer(dspy.Signature):
    """Answer the medical question in the requested language using ONLY the provided database.
    Cite sources using [N]. If no info is found, state that clearly."""
    
    system_instruction: str = dspy.InputField(desc="Core role and persona for the assistant")
    context: str = dspy.InputField(desc="Numbered medical context paragraphs")
    history: str = dspy.InputField(desc="Recent conversation history")
    question: str = dspy.InputField(desc="The current medical question")
    target_language: str = dspy.InputField(desc="The language to respond in")
    no_info_message: str = dspy.InputField(desc="The message to show if no info is found")
    
    result: MedicalAnswer = dspy.OutputField()
