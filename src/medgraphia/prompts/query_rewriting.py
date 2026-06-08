"""
Prompts for query rewriting and coreference resolution.
"""

from __future__ import annotations

import dspy

# ---------------------------------------------------------------------------
# Routing rubric — passed as a locked InputField to prevent MIPRO from
# modifying the scoring rules during optimisation.
# ---------------------------------------------------------------------------

ROUTING_RUBRIC = """\
[Complexity Scoring Rubric — follow step-by-step, do NOT deviate]

Step 1 — Entity Count (E):
  E = 1  →  single disease / drug / symptom / procedure
  E = 2  →  two distinct medical entities (drug-drug, disease-drug, …)
  E = 3  →  three or more entities, OR a vague population ("elderly diabetics with CKD")

Step 2 — Cognitive Depth (I):
  I = 1  →  fact lookup, definition, administrative (dose, schedule, hours, FAQ)
  I = 2  →  treatment plan for one condition, single drug interaction, symptom-to-diagnosis
  I = 3  →  complex clinical decision, multi-drug interaction, life-threatening / ICU scenario

Step 3 — Total = E + I → tier:
  Total ≤ 3  →  SMALL
  Total = 4  →  MEDIUM
  Total ≥ 5  →  LARGE
"""


# ---------------------------------------------------------------------------
# New combined signature: coreference resolution + complexity scoring
# ---------------------------------------------------------------------------


class QueryRewritingWithTier(dspy.Signature):
    """Resolve coreference in the user query using conversation history, then score clinical complexity.

    CRITICAL RULES:
    1. Resolve ALL pronouns / ellipsis using prior turns (e.g. "its" → the drug name from history).
    2. Keep the ORIGINAL LANGUAGE of latest_message in rewritten_query.
    3. Preserve every clinical qualifier (dosage, timing, patient specifics, comorbidities).
    4. Work through the rubric STEP-BY-STEP and write the scores in complexity_reasoning before
       deciding complexity_tier — never guess the tier without computing E and I first.
    5. complexity_tier MUST be exactly one of the three strings: SMALL, MEDIUM, LARGE.
    """

    rubric = dspy.InputField(
        desc="Read-only scoring rubric — compute E, I, Total as instructed; do not modify"
    )
    history = dspy.InputField(
        desc="Recent conversation turns formatted as 'User: …\\nAssistant: …'. Empty string if first message."
    )
    latest_message = dspy.InputField(
        desc="The user's current input, which may contain pronouns or implicit references to history"
    )

    is_standalone = dspy.OutputField(
        desc="'true' if the original message was already a self-contained question, 'false' otherwise"
    )
    rewritten_query = dspy.OutputField(
        desc="Standalone, context-complete query in the SAME language as latest_message"
    )
    complexity_reasoning = dspy.OutputField(
        desc="Scoring trace, e.g. 'E=2 (metformin + contrast agent), I=3 (contraindication in CKD → Total=5'"
    )
    complexity_tier = dspy.OutputField(desc="Must be exactly one of: SMALL, MEDIUM, LARGE")


class RewritingJudge(dspy.Signature):
    """
    You are a highly rigorous medical system evaluation expert. Your task is to evaluate whether the model-predicted 'rewritten query' is successful by comparing it to the 'expected' gold standard.

    CRITICAL Evaluation Criteria:
    1. SEMANTIC COMPLETENESS: The predicted query must preserve ALL nuances present in the 'expected' gold standard. This includes all temporal constraints, environmental conditions, patient specificities, and clinical qualifiers.
    2. NO OVERSIMPLIFICATION: If the 'expected' query contains any restrictive modifiers that change the scope of the clinical question, and the 'predicted' query omits them, it must be marked as a FAILURE.
    3. Intent Consistency: The core medical intent must remain identical to the original multi-turn conversation.
    4. Coreference Resolution: All pronouns must be resolved to their specific medical entities identified in the history.
    5. Scoring logic:
       - 1.0: Semantically equivalent to 'expected' in all clinical aspects.
       - 0.1-0.9: Partial success, but lost critical qualifying information or context.
       - 0.0: Failed intent, failed coreference, or hallucinated information.

    Provide a reasoning (your step-by-step analysis) and a score (a float between 0.0 and 1.0).
    """

    history = dspy.InputField(desc="Conversation history (context)")
    latest_message = dspy.InputField(desc="The user's original ambiguous question")
    expected = dspy.InputField(desc="The gold reference answer (standard rewriting by an expert)")
    predicted = dspy.InputField(desc="The model's actual predicted rewriting result")

    reasoning = dspy.OutputField(desc="Analysis of whether the predicted result meets the criteria")
    score = dspy.OutputField(desc="A float between 0.0 and 1.0 (e.g., 1.0, 0.8, 0.0)")
