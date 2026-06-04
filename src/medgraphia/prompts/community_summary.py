"""
Prompts for summarizing knowledge graph communities.
"""
from __future__ import annotations
import dspy
from pydantic import BaseModel, Field

class CommunitySummaryResult(BaseModel):
    """Structured clinical summary of a knowledge graph community."""
    summary: str = Field(description="A 2-3 sentence overview of the clinical focus of this community")
    explanation: str = Field(description="Brief explanation of why these concepts are grouped together")
    clinical_relevance: str = Field(description="Key clinical implications or takeaway")

class SummarizeMedicalCommunity(dspy.Signature):
    """You are a medical research assistant. You will be provided with a list of 
    medical concepts and their relationships within a specific knowledge community. 
    Your task is to provide a concise, clinically accurate summary of this community. 
    Focus on the primary medical theme connecting these entities."""
    
    concepts: str = dspy.InputField(desc="List of medical concepts in the cluster")
    relations: str = dspy.InputField(desc="Key relationships between these concepts")
    
    result: CommunitySummaryResult = dspy.OutputField()
