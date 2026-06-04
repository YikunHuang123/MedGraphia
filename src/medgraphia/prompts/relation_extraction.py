"""
Prompts for extracting medical relations from text.
"""
from __future__ import annotations
import dspy
from pydantic import BaseModel, Field
from medgraphia.domain import RelationType

class ExtractedRelation(BaseModel):
    """A single relation extracted by the LLM."""
    source_cui: str = Field(description="CUI of the source entity")
    target_cui: str = Field(description="CUI of the target entity")
    relation_type: str = Field(description="The semantic type of the relationship")
    evidence_text: str = Field(description="Verbatim excerpt from the text supporting this relation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")

class ExtractMedicalRelations(dspy.Signature):
    """You are a medical knowledge graph expert specialising in pharmacology and clinical medicine.
    Your task is to extract semantic relations between pairs of medical entities from the provided text.
    ONLY include relations explicitly supported by the text.
    NO SELF-LOOPS: source_cui must not equal target_cui."""
    
    text_content: str = dspy.InputField(desc="Raw medical text to analyze")
    entities: str = dspy.InputField(desc="Identified entities with CUIs and types")
    allowed_relations: str = dspy.InputField(desc="Comma-separated list of allowed relation types")
    
    relations: list[ExtractedRelation] = dspy.OutputField(desc="JSON list of extracted relations")
