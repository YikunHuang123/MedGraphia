"""
Safety Guardrails using Llama-Guard (Input and Output Moderation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from medgraphia.config import get_settings
from medgraphia.llm.gateway import CompletionRequest, LiteLLMGateway, LLMProvider
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Standard System Prompt for Llama-Guard 3 / 4
# Ref: https://huggingface.co/meta-llama/Llama-Guard-3-8B
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

<BEGIN CONVERSATION>
{conversation_history}
<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {role} MESSAGE in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories."""

@dataclass
class SafetyResult:
    is_safe: bool
    reason: str = ""
    category: str = ""

import httpx
from medgraphia.config import get_settings
from medgraphia.llm.gateway import CompletionRequest, LiteLLMGateway, LLMProvider
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ... (LLAMA_GUARD_SYSTEM_PROMPT remains same) ...

class LlamaGuard:
    """
    Utility for moderating medical queries and responses.
    """
    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = self.settings.guardrails_enabled
        
        # 1. Configuration from settings
        provider_name = self.settings.llama_guard_provider.lower()
        self.pure_model_name = self.settings.llama_guard_model
        
        # 2. Ensure model name is correctly prefixed for LiteLLM
        if "/" in self.pure_model_name:
            self.model_name = self.pure_model_name
            actual_provider = self.pure_model_name.split("/")[0]
            self.model_tag = self.pure_model_name.split("/")[1]
        else:
            self.model_name = f"{provider_name}/{self.pure_model_name}"
            actual_provider = provider_name
            self.model_tag = self.pure_model_name

        # 3. Initialize the gateway
        try:
            self.provider_enum = LLMProvider(actual_provider)
        except ValueError:
            logger.warning("guardrails_unknown_provider", provider=actual_provider)
            self.provider_enum = LLMProvider.OLLAMA

        self.gateway = LiteLLMGateway.for_provider(self.provider_enum, self.model_name)
        logger.info("guardrails_initialized", model=self.model_name, provider=self.provider_enum.value)

    async def ensure_model_ready(self) -> None:
        """
        Ensures the model is available on the provider.
        Specifically for Ollama, it will trigger a pull if missing.
        """
        if not self.enabled or self.provider_enum != LLMProvider.OLLAMA:
            return

        ollama_base = self.settings.embedding_base_url or "http://localhost:11434"
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            # Check if model exists
            try:
                check_resp = await client.post(f"{ollama_base}/api/show", json={"name": self.model_tag})
                if check_resp.status_code == 200:
                    logger.info("guardrails_model_found_locally", model=self.model_tag)
                    return
            except Exception:
                pass # Proceed to pull if check fails

            # Pull the model
            logger.info("guardrails_model_missing_pulling", model=self.model_tag, message="This may take a few minutes...")
            try:
                pull_resp = await client.post(
                    f"{ollama_base}/api/pull", 
                    json={"name": self.model_tag, "stream": False},
                    timeout=1200.0
                )
                if pull_resp.status_code == 200:
                    logger.info("guardrails_model_pull_success", model=self.model_tag)
                else:
                    logger.error("guardrails_model_pull_failed", status=pull_resp.status_code)
            except Exception as exc:
                logger.error("guardrails_model_pull_error", error=str(exc))

    async def check_input(self, question: str) -> SafetyResult:
        """
        Check if the user's prompt violates safety policies before retrieval.
        """
        if not self.enabled:
            return SafetyResult(is_safe=True)
            
        logger.info("guardrails_checking_input", query_len=len(question))
        
        conversation = f"User: {question}"
        return await self._call_llama_guard(role="User", conversation=conversation)

    async def check_output(self, question: str, answer: str) -> SafetyResult:
        """
        Check if the generated response is safe for the patient/user.
        """
        if not self.enabled:
            return SafetyResult(is_safe=True)
            
        logger.info("guardrails_checking_output", answer_len=len(answer))
        
        conversation = f"User: {question}\n\nAgent: {answer}"
        return await self._call_llama_guard(role="Agent", conversation=conversation)

    async def _call_llama_guard(self, role: str, conversation: str) -> SafetyResult:
        """Internal helper to invoke the model."""
        prompt = LLAMA_GUARD_SYSTEM_PROMPT.format(
            role=role,
            conversation_history=conversation
        )
        
        req = CompletionRequest(
            system_prompt="", # Llama Guard instructions are self-contained in user prompt
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=20
        )
        
        resp = await self.gateway.acomplete(req)
        
        if not resp.ok:
            logger.error("guardrails_failed", error=resp.metadata.get("error"))
            # In a medical context, failing open (is_safe=True) is risky but 
            # failing closed (is_safe=False) breaks the UX. 
            # Default to safe but log heavily.
            return SafetyResult(is_safe=True)
            
        text = resp.text.strip().lower()
        
        if text.startswith("safe"):
            return SafetyResult(is_safe=True)
        
        # Handle "unsafe\nS1, S2"
        lines = text.split("\n")
        category = lines[1] if len(lines) > 1 else "Unknown"
        
        logger.warning("guardrails_violation_detected", role=role, category=category)
        return SafetyResult(is_safe=False, category=category, reason=f"Violates {category}")
