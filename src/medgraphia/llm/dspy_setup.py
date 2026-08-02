"""
Central configuration for DSPy integration.
"""

from __future__ import annotations

from typing import Literal

import dspy

from medgraphia.config import get_settings
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Cache LMs to avoid re-initializing
_LM_CACHE: dict[str, dspy.LM] = {}


def get_lm(
    task: Literal["default", "rewriter", "summarizer", "judge"] = "default",
    provider_override: str | None = None,
    model_override: str | None = None,
    temperature: float | None = None,
) -> dspy.LM:
    """
    Get a dspy.LM instance for a specific task based on configuration.
    """
    cfg = get_settings()

    # 1. Determine provider and model
    # Priority: Direct Arguments > Task-specific Settings > Global Default
    provider = provider_override or cfg.default_llm_provider
    model = model_override or cfg.default_llm_model

    if not provider_override:
        if task == "rewriter" and cfg.rewriter_llm_provider:
            provider = cfg.rewriter_llm_provider
            model = cfg.rewriter_llm_model
        elif task == "summarizer" and cfg.summarizer_llm_provider:
            provider = cfg.summarizer_llm_provider
            model = cfg.summarizer_llm_model
        elif task == "judge" and cfg.judge_llm_provider:
            provider = cfg.judge_llm_provider
            model = cfg.judge_llm_model

    cache_key = f"{provider}/{model}/{task}/{temperature}"
    if cache_key in _LM_CACHE:
        return _LM_CACHE[cache_key]

    # Construct model_id ensuring provider prefix is present for LiteLLM
    # e.g., "deepseek/deepseek-chat" or "openai/gpt-4o"
    # vLLM is the odd one out: litellm's dedicated prefix for a self-hosted
    # OpenAI-compatible vLLM server is "hosted_vllm/", not "vllm/".
    litellm_prefix = "hosted_vllm" if provider == "vllm" else provider
    if model.startswith(f"{litellm_prefix}/"):
        model_id = model
    else:
        model_id = f"{litellm_prefix}/{model}"

    # 2. Determine API key and Base URL
    api_key = None
    api_base = None

    # Check for specialized task credentials first
    if task == "rewriter" and (cfg.rewriter_llm_api_key.get_secret_value() or cfg.rewriter_llm_base_url):
        api_key = cfg.rewriter_llm_api_key.get_secret_value()
        api_base = cfg.rewriter_llm_base_url or None
    elif task == "summarizer" and (cfg.summarizer_llm_api_key.get_secret_value() or cfg.summarizer_llm_base_url):
        api_key = cfg.summarizer_llm_api_key.get_secret_value()
        api_base = cfg.summarizer_llm_base_url or None
    elif task == "judge" and (cfg.judge_llm_api_key.get_secret_value() or cfg.judge_llm_base_url):
        api_key = cfg.judge_llm_api_key.get_secret_value()
        api_base = cfg.judge_llm_base_url or None
    else:
        # Fallback to standard provider credentials
        if provider == "openai":
            api_key = cfg.openai_api_key.get_secret_value()
            api_base = cfg.openai_base_url or None
        elif provider == "groq":
            api_key = cfg.groq_api_key.get_secret_value()
        elif provider == "deepseek":
            api_key = cfg.deepseek_api_key.get_secret_value()
        elif provider == "anthropic":
            api_key = cfg.anthropic_api_key.get_secret_value()
        elif provider == "gemini":
            api_key = cfg.gemini_api_key.get_secret_value()
        elif provider == "ollama":
            api_base = cfg.llm_base_url or "http://localhost:11434"
        elif provider == "vllm":
            api_key = "vllm"  # ignored unless the vLLM server was started with --api-key
            api_base = cfg.vllm_base_url or "http://localhost:8000/v1"

    try:
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Disable Qwen3's built-in thinking mode for ollama — the <think> tokens
        # cause DSPy's JSONAdapter to receive an empty string and fail to parse.
        if provider == "ollama":
            kwargs["extra_body"] = {"think": False}

        lm = dspy.LM(model=model_id, **kwargs)
        _LM_CACHE[cache_key] = lm
        logger.info(
            "dspy_lm_initialized", task=task, model=model_id, has_custom_base=bool(api_base)
        )
        return lm
    except Exception as exc:
        logger.error("dspy_lm_init_failed", task=task, model=model_id, error=str(exc))
        raise


def init_dspy() -> None:
    """
    Legacy global initialization. Sets the default LM in dspy.settings.
    """
    lm = get_lm("default")
    dspy.settings.configure(lm=lm)
