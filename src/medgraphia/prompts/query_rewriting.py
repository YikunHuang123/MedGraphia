"""
Prompts for query rewriting and coreference resolution.
"""
from __future__ import annotations
import dspy
from pydantic import BaseModel, Field

class RewrittenQuery(BaseModel):
    """Structured output for the query rewriter."""
    is_standalone: bool = Field(
        ..., 
        description="Whether the original message was already standalone and didn't need rewriting."
    )
    rewritten_query: str = Field(
        ..., 
        description="The standalone, context-complete medical query in the SAME language as the latest_message."
    )

class RewriteMedicalQuery(dspy.Signature):
    """Analyze the conversation history and the latest user message to produce a standalone medical query.
    
    CRITICAL INSTRUCTIONS:
    1. Resolve all pronouns (e.g., 'it', 'this', 'that', '他', '它', '这个') by looking at previous turns.
    2. If the user asks about 'treatment' or 'diagnosis' without naming the disease, explicitly include the disease name from the history.
    3. Maintain the ORIGINAL LANGUAGE of the latest message (e.g., if asked in Chinese, the result must be Chinese).
    4. Keep all specific medical terminology intact.
    5. If the message is already a standalone medical question, return it as is but ensure 'is_standalone' is True."""
    
    history: str = dspy.InputField(desc="Previous conversation turns between User and Assistant")
    latest_message: str = dspy.InputField(desc="The user's latest input needing potential rewriting")
    result: RewrittenQuery = dspy.OutputField()
