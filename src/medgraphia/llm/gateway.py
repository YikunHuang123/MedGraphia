"""
Unified LiteLLM gateway for MedGraphia.

Routes all LLM calls through a single interface regardless of provider:
  - OpenAI    (gpt-4o, gpt-4o-mini, …)
  - Anthropic (claude-3-5-sonnet-*, claude-3-haiku-*, …)
  - DeepSeek  (deepseek-chat, deepseek-reasoner)
  - Gemini    (gemini-2.5-pro, gemini-2.5-flash, …)
  - Ollama    (any locally-hosted model via OpenAI-compatible endpoint)

LiteLLM handles:
  - Unified message format across all providers
  - Automatic retry / timeout management
  - Token counting per request
  - Async and sync call modes
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Request / response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompletionRequest:
    """A single-turn completion request."""
    system_prompt: str
    user_prompt: str
    model_id: str = ""          # explicit override; empty → use gateway default
    temperature: float = 0.1
    max_tokens: int = 2048
    json_mode: bool = False     # request JSON object output when provider supports it
    stream: bool = False
    mock_response: str | None = None


@dataclass
class CompletionResponse:
    """Result from a single-turn completion call."""
    text: str
    model_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        return bool(self.text) and "error" not in self.metadata


# ---------------------------------------------------------------------------
# Internal: provider → litellm model string
# ---------------------------------------------------------------------------

def _build_litellm_model_id(provider: LLMProvider, model_name: str) -> str:
    """
    Map (provider, model_name) → the model string litellm expects.

    LiteLLM routing table reference:
      https://docs.litellm.ai/docs/providers
    """
    if not model_name:
        return ""
        
    # If the model name already contains a slash, assume it's already prefixed
    if "/" in model_name:
        return model_name

    match provider:
        case LLMProvider.OPENAI:
            return model_name                       # "gpt-4o", "gpt-4o-mini", …
        case LLMProvider.ANTHROPIC:
            return f"anthropic/{model_name}"        # "anthropic/claude-3-5-sonnet-20241022"
        case LLMProvider.DEEPSEEK:
            return f"deepseek/{model_name}"         # "deepseek/deepseek-chat"
        case LLMProvider.GEMINI:
            return f"gemini/{model_name}"           # "gemini/gemini-1.5-pro"
        case LLMProvider.GROQ:
            return f"groq/{model_name}"             # "groq/llama-3.3-70b-versatile"
        case LLMProvider.OLLAMA:
            return f"ollama/{model_name}"           # "ollama/qwen2.5:7b"
        case _:
            return model_name


def _build_extra_kwargs(provider: LLMProvider, cfg: Any) -> dict[str, Any]:
    """Build litellm keyword arguments (api_key, api_base) for each provider."""
    kwargs: dict[str, Any] = {}

    def get_secret(attr_name: str) -> str | None:
        secret_obj = getattr(cfg, attr_name, None)
        if secret_obj and hasattr(secret_obj, "get_secret_value"):
            return secret_obj.get_secret_value()
        return None

    match provider:
        case LLMProvider.OPENAI:
            key = get_secret("openai_api_key")
            if key:
                kwargs["api_key"] = key
            base = getattr(cfg, "openai_base_url", "")
            if base:
                kwargs["api_base"] = base

        case LLMProvider.ANTHROPIC:
            key = get_secret("anthropic_api_key")
            if key:
                kwargs["api_key"] = key

        case LLMProvider.DEEPSEEK:
            key = get_secret("deepseek_api_key")
            if key:
                kwargs["api_key"] = key
            kwargs["api_base"] = "https://api.deepseek.com"

        case LLMProvider.GEMINI:
            key = get_secret("gemini_api_key")
            if key:
                kwargs["api_key"] = key

        case LLMProvider.GROQ:
            key = get_secret("groq_api_key")
            if key:
                kwargs["api_key"] = key

        case LLMProvider.OLLAMA:
            base = getattr(cfg, "llm_base_url", "") or "http://localhost:11434"
            kwargs["api_base"] = base
            kwargs["api_key"] = "ollama"

    return kwargs


# ---------------------------------------------------------------------------
# Providers that support JSON-mode response_format
# ---------------------------------------------------------------------------

_JSON_MODE_PROVIDERS = {LLMProvider.OPENAI, LLMProvider.DEEPSEEK, LLMProvider.GEMINI, LLMProvider.GROQ}


# ---------------------------------------------------------------------------
# LiteLLMGateway — public interface
# ---------------------------------------------------------------------------

class LiteLLMGateway:
    """
    Unified async/sync LLM gateway backed by litellm.

    Instantiate via factory methods::

        # Use global config (respects llm_provider / llm_model env vars)
        gw = LiteLLMGateway.from_settings()

        # Explicit provider + model
        gw = LiteLLMGateway.for_provider(LLMProvider.ANTHROPIC, "claude-3-5-sonnet-20241022")

        # Single async completion
        resp = await gw.acomplete(CompletionRequest(
            system_prompt="You are a medical assistant.",
            user_prompt="What is metformin?",
        ))
        print(resp.text)

        # Streaming
        async for chunk in gw.astream(req):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        provider: LLMProvider,
        model_name: str,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._model_id = _build_litellm_model_id(provider, model_name)
        self._extra_kwargs: dict[str, Any] = extra_kwargs or {}

    @property
    def model_id(self) -> str:
        """The fully-qualified LiteLLM model identifier (e.g. 'anthropic/claude-3-5-sonnet')."""
        return self._model_id

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> "LiteLLMGateway":
        """
        Build a gateway from Settings (reads .env / env vars).

        provider_override / model_override take precedence over config values.
        """
        from medgraphia.config import get_settings
        cfg = get_settings()

        raw_provider = provider_override or cfg.llm_provider
        try:
            provider = LLMProvider(raw_provider)
        except ValueError:
            logger.warning("unknown_llm_provider", value=raw_provider, fallback="ollama")
            provider = LLMProvider.OLLAMA

        model_name = model_override or cfg.llm_model
        extra = _build_extra_kwargs(provider, cfg)
        return cls(provider=provider, model_name=model_name, extra_kwargs=extra)

    @classmethod
    def for_provider(
        cls,
        provider: LLMProvider,
        model_name: str,
    ) -> "LiteLLMGateway":
        """Build a gateway for an explicit provider + model pair."""
        from medgraphia.config import get_settings
        cfg = get_settings()
        extra = _build_extra_kwargs(provider, cfg)
        return cls(provider=provider, model_name=model_name, extra_kwargs=extra)

    # ------------------------------------------------------------------
    # Async single-turn completion
    # ------------------------------------------------------------------

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Execute an async single-turn completion.

        Returns a CompletionResponse; never raises — errors are captured in
        response.metadata["error"] and response.ok will be False.
        """
        try:
            import litellm
        except ImportError:
            logger.error("litellm_not_installed")
            return CompletionResponse(
                text="",
                model_used=self._model_id,
                metadata={"error": "litellm is not installed"},
            )

        # Determine final model ID (re-prefix if request has an override)
        if request.model_id:
            model_id = _build_litellm_model_id(self._provider, request.model_id)
        else:
            model_id = self._model_id

        messages = _build_messages(request.system_prompt, request.user_prompt)

        call_kwargs: dict[str, Any] = {
            **self._extra_kwargs,
            "model": model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.mock_response:
            call_kwargs["mock_response"] = request.mock_response
            # Ensure dummy key so litellm doesn't fail on missing credentials during mock
            if "api_key" not in call_kwargs or not call_kwargs["api_key"]:
                call_kwargs["api_key"] = "mock-key"

        if request.json_mode and self._provider in _JSON_MODE_PROVIDERS:
            call_kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        try:
            response = await litellm.acompletion(**call_kwargs)
            latency = (time.monotonic() - t0) * 1000.0
            raw_text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)

            logger.info(
                "litellm_acomplete_ok",
                model=model_id,
                latency_ms=f"{latency:.0f}",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
            )
            return CompletionResponse(
                text=raw_text,
                model_used=model_id,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000.0
            logger.error("litellm_acomplete_failed", model=model_id, error=str(exc))
            return CompletionResponse(
                text="",
                model_used=model_id,
                latency_ms=latency,
                metadata={"error": str(exc)},
            )

    async def acomplete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Async completion that parses and returns the first JSON object found
        in the response.  Returns {} on parse failure.
        """
        req = CompletionRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            json_mode=True,
        )
        resp = await self.acomplete(req)
        return _parse_json_safe(resp.text)

    # ------------------------------------------------------------------
    # Async streaming
    # ------------------------------------------------------------------

    async def astream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """
        Async streaming completion.  Yields text delta strings as they arrive.

        Usage::

            async for chunk in gateway.astream(req):
                print(chunk, end="", flush=True)
        """
        try:
            import litellm
        except ImportError:
            logger.error("litellm_not_installed")
            return

        # Determine final model ID (re-prefix if request has an override)
        if request.model_id:
            model_id = _build_litellm_model_id(self._provider, request.model_id)
        else:
            model_id = self._model_id

        messages = _build_messages(request.system_prompt, request.user_prompt)

        call_kwargs: dict[str, Any] = {
            **self._extra_kwargs,
            "model": model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        if request.mock_response:
            call_kwargs["mock_response"] = request.mock_response

        try:
            stream = await litellm.acompletion(**call_kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("litellm_astream_failed", model=model_id, error=str(exc))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LiteLLMGateway(provider={self._provider.value!r}, model={self._model_name!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def _parse_json_safe(text: str) -> dict[str, Any]:
    """
    Extract the first valid JSON object from *text*.

    Handles:
      - Markdown code fences (```json … ```)
      - Trailing commas (common LLM mistake)
      - Leading/trailing whitespace
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    content = fence_match.group(1) if fence_match else text

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return {}

    json_str = content[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            return json.loads(cleaned)
        except Exception:
            logger.debug("json_parse_failed", snippet=text[:80])
            return {}
