"""
Signatures and prompts for synthetic medical data generation.
"""

from __future__ import annotations

import dspy


class GenerateSyntheticMedicalQA(dspy.Signature):
    """Given real medical context chunks and a target complexity level, generate a realistic
    multi-turn clinical conversation with a grounded answer.

    STRICT REQUIREMENTS:
    1. requires_history control:
       If True: the latest_message MUST contain pronouns (e.g. "it", "this drug") needing resolution.
       If False: the latest_message MUST be a standalone, explicit medical question.
    2. synthetic_history: exactly 2-3 prior turns in "User: ...\\nAssistant: ..." format.
       If requires_history=False, set this to "No history."
    3. grounded_answer: clinical answer ONLY from the provided context. Every factual
       claim must be cited with [N] markers. No hallucinated content.
    4. Respect the target_tier complexity level:
       SMALL  → E+I ≤ 3 (single entity, fact lookup)
       MEDIUM → E+I = 4 (two entities or treatment decision for one condition)
       LARGE  → E+I ≥ 5 (three+ entities or complex clinical decision / ICU scenario)
    """

    real_context: str = dspy.InputField(
        desc="Numbered medical literature chunks [1]... [2]... (may mix EN/ZH/DE languages)"
    )
    target_tier: str = dspy.InputField(
        desc="Target complexity: SMALL / MEDIUM / LARGE (follow the E+I rubric strictly)"
    )
    simulated_persona: str = dspy.InputField(
        desc="Questioner persona (affects tone/vocabulary of the generated history)"
    )
    requires_history: bool = dspy.InputField(
        desc="Whether to force a dependency on conversation history (pronoun resolution)"
    )
    target_language: str = dspy.InputField(
        desc="The language the generated conversation (history, questions, answer) MUST be in (e.g. Chinese, English, German)"
    )

    synthetic_history: str = dspy.OutputField(
        desc="2-3 prior conversation turns formatted as 'User: ...\\nAssistant: ...'"
    )
    synthetic_latest_message: str = dspy.OutputField(
        desc="Follow-up question. Use pronouns if requires_history=True."
    )
    expected_rewritten_query: str = dspy.OutputField(
        desc="Standalone version of synthetic_latest_message with ALL pronouns resolved"
    )
    grounded_answer: str = dspy.OutputField(
        desc="Evidence-based answer using ONLY the provided context, with [N] citation markers"
    )
    disclaimer: str = dspy.OutputField(
        desc="One-sentence medical disclaimer in the same language as the answer"
    )
