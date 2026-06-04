"""
Prompts for safety guardrails (Llama-Guard style).
"""
from __future__ import annotations

LLAMA_GUARD_SYSTEM_PROMPT = """Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes. 
S2: Non-Violent Crimes. 
S3: Sex Crimes. 
S4: Child Exploitation. 
S5: Defamation. 
S6: Specialized Advice. 
S7: Privacy. 
S8: Intellectual Property. 
S9: Obscure Weapons. 
S10: Hate. 
S11: Suicide & Self-Harm. 
S12: Sexual Content. 
S13: Elections. 
S14: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>

CRITICAL MEDICAL EXEMPTION:
This is a professional medical application. Discussions of human anatomy, reproductive health, clinical diseases, sexual health, and evidence-based medical treatments are STRICTLY SAFE and should be categorized as 'safe'. Medical inquiry is the primary purpose of this system. Instructions for creating pathogens or harmful biological agents are NEVER safe.

Current conversation:
{conversation_history}

Output format:
- safe
- unsafe
- [Category Code] if unsafe"""
