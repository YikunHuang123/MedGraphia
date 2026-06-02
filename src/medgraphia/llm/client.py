"""
Unified LLM client factory for MedGraphia.

Provides integration with pydantic-ai for structured output and
maintains a robust JSON parser for legacy/fallback use cases.
"""
from __future__ import annotations

import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel

from medgraphia.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

from medgraphia.llm.gateway import LLMProvider

# ---------------------------------------------------------------------------
# pydantic-ai Model Factory
# ---------------------------------------------------------------------------

def get_model(model_override: str | None = None) -> Model:
    """
    Create a pydantic-ai Model instance based on project settings.
    Supports OpenAI, DeepSeek, Anthropic, and Ollama (via OpenAI-compatible API).
    """
    from medgraphia.config import get_settings
    cfg = get_settings()
    
    # Parse provider into our standard enum
    try:
        provider = LLMProvider(cfg.llm_provider)
    except ValueError:
        provider = LLMProvider.OLLAMA

    model_name = model_override or cfg.llm_model
    base_url = cfg.llm_base_url

    def get_val(attr_name: str) -> str | None:
        obj = getattr(cfg, attr_name, None)
        if obj and hasattr(obj, "get_secret_value"):
            return obj.get_secret_value()
        return None

    # Handle Anthropic separately as pydantic-ai uses a specific class
    if provider == LLMProvider.ANTHROPIC:
        from pydantic_ai.models.anthropic import AnthropicModel
        key = get_val("anthropic_api_key")
        return AnthropicModel(model_name, api_key=key or "dummy")

    # For OpenAI-compatible endpoints (DeepSeek, Ollama, OpenAI)
    api_key = "dummy"
    match provider:
        case LLMProvider.OPENAI:
            api_key = get_val("openai_api_key") or "dummy"
            base_url = cfg.openai_base_url or "https://api.openai.com/v1"
        case LLMProvider.DEEPSEEK:
            api_key = get_val("deepseek_api_key") or "dummy"
            base_url = "https://api.deepseek.com"
        case LLMProvider.OLLAMA:
            base_url = base_url or "http://localhost:11434/v1"
            api_key = "ollama"

    # pydantic-ai v1.104+ reads these from environment variables or explicit config
    import os
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    return OpenAIModel(model_name=model_name)


# ---------------------------------------------------------------------------
# Legacy LLMClient (Updated to be more robust)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Wrapper for litellm calls, primarily used for plain-text completion
    or when pydantic-ai isn't used directly.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:3b",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._provider = provider
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Asynchronous JSON completion via litellm.
        """
        import litellm
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Prepare litellm call
        prefix = "openai" if self._provider in ["openai", "deepseek", "local"] else self._provider
        model_id = f"{prefix}/{self._model_name}"
        
        try:
            response = await litellm.acompletion(
                model=model_id,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                api_key=self._api_key,
                api_base=self._base_url,
                response_format={"type": "json_object"} if self._provider in ["openai", "deepseek"] else None
            )
            raw = response.choices[0].message.content or ""
            return _parse_json(raw)
        except Exception as exc:
            logger.warning("llm_call_failed", model=model_id, error=str(exc))
            return {}

    @classmethod
    def from_settings(cls, model_override: str | None = None) -> "LLMClient":
        from medgraphia.config import get_settings
        cfg = get_settings()
        
        api_key = ""
        if cfg.llm_provider == "deepseek":
            api_key = cfg.deepseek_api_key.get_secret_value()
        elif cfg.llm_provider == "openai":
            api_key = cfg.openai_api_key.get_secret_value()
            
        return cls(
            provider=cfg.llm_provider,
            model=model_override or cfg.llm_model,
            api_key=api_key,
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )


# ---------------------------------------------------------------------------
# Robust JSON Extraction
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict[str, Any]:
    """
    Extracts the first valid JSON object from a string.
    More robust than simple regex or bracket counting.
    """
    text = text.strip()
    
    # 1. Try to find content within markdown fences
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    content = match.group(1) if match else text
    
    # 2. Locate the first { and last }
    start = content.find('{')
    end = content.rfind('}')
    
    if start == -1 or end == -1:
        return {}
        
    json_str = content[start:end+1]
    
    try:
        # 3. Standard parse
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 4. Final attempt: basic cleanup for trailing commas
        try:
            # Simple heuristic: remove trailing commas before closing braces/brackets
            json_str_clean = re.sub(r",\s*([\]}])", r"\1", json_str)
            return json.loads(json_str_clean)
        except Exception:
            logger.warning("json_extraction_failed", snippet=text[:100])
            return {}
