"""
Safety Guardrails using Llama-Guard (Input and Output Moderation).
"""
from __future__ import annotations

import httpx
from dataclasses import dataclass

from medgraphia.config import get_settings
from medgraphia.llm.gateway import CompletionRequest, LiteLLMGateway, LLMProvider
from medgraphia.logger import get_logger
from medgraphia.prompts.safety import LLAMA_GUARD_SYSTEM_PROMPT
from medgraphia.domain.chat import Message

logger = get_logger(__name__)

@dataclass
class SafetyResult:
    is_safe: bool
    reason: str = ""
    category: str = ""

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

    async def check_input(self, question: str, history: list[Message] | None = None) -> SafetyResult:
        """
        Check if the user's prompt violates safety policies before retrieval.
        Includes history for multi-turn context.
        """
        if not self.enabled:
            return SafetyResult(is_safe=True)
            
        logger.info("guardrails_checking_input", query_len=len(question))
        
        conversation = ""
        if history:
            for m in history[-5:]: # Look at last 5 turns for context
                role = "User" if m.role == "user" else "Assistant"
                conversation += f"{role}: {m.content}\n\n"
        
        conversation += f"User: {question}"
        return await self._call_llama_guard(role="User", conversation=conversation)

    async def check_output(self, question: str, answer: str, history: list[Message] | None = None) -> SafetyResult:
        """
        Check if the generated response is safe for the patient/user.
        """
        if not self.enabled:
            return SafetyResult(is_safe=True)
            
        logger.info("guardrails_checking_output", answer_len=len(answer))
        
        conversation = ""
        if history:
            for m in history[-5:]:
                role = "User" if m.role == "user" else "Assistant"
                conversation += f"{role}: {m.content}\n\n"
        
        conversation += f"User: {question}\n\nAssistant: {answer}"
        return await self._call_llama_guard(role="Assistant", conversation=conversation)

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
        
        try:
            resp = await self.gateway.acomplete(req)
        except Exception as exc:
            logger.error("guardrails_exception", error=str(exc))
            return SafetyResult(is_safe=True)
        
        if not resp.ok:
            logger.error("guardrails_failed", error=resp.metadata.get("error"))
            return SafetyResult(is_safe=True)
            
        text = resp.text.strip().lower()
        lines = text.split("\n")
        first_line = lines[0].strip()
        
        # Robust parsing:
        # 1. If it starts with 'safe', it's safe.
        # 2. If it contains 'unsafe', it's unsafe.
        # 3. Otherwise, if it's ambiguous, we default to safe to avoid false positives (RAG context).
        
        if first_line.startswith("safe"):
            return SafetyResult(is_safe=True)
        
        if "unsafe" in first_line:
            category_str = lines[1].strip().upper() if len(lines) > 1 else "Unknown"
            violated = [c.strip() for c in category_str.split(",")]
            
            # Categories that must ALWAYS be blocked
            hard_danger = {"S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14"}
            
            # ── HARD OVERRIDE: Allow Medical/Specialized Advice ─────────────
            # Only allow if S6 is present AND no hard danger categories are present.
            if "S6" in violated and not any(c in hard_danger for c in violated):
                logger.info("guardrails_overridden_s6_allowed", role=role)
                return SafetyResult(is_safe=True)
                
            logger.warning("guardrails_violation_detected", role=role, category=category_str)
            return SafetyResult(is_safe=False, category=category_str, reason=f"Violates {category_str}")

        # If the model didn't clearly say 'unsafe', we default to safe.
        # This prevents small models (like 1B) from blocking benign content like "Hello" 
        # if they output something unexpected.
        logger.info("guardrails_ambiguous_output", output=text)
        return SafetyResult(is_safe=True)
