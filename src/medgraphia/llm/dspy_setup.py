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
    task: Literal["default", "rewriter", "summarizer", "judge", "translator"] = "default",
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
        elif task == "translator" and cfg.translator_llm_provider:
            provider = cfg.translator_llm_provider
            model = cfg.translator_llm_model

    if provider == "vllm":
        # LM objects are cached below, so this is the only per-call hook we
        # get — must run on every call (cache hit or miss), not just once at
        # construction, otherwise a tier that goes back to sleep never wakes
        # up again for subsequent requests.
        from medgraphia.llm.vllm_sleep_manager import get_sleep_manager

        base_url = cfg.vllm_small_base_url if model == cfg.llm_small_model else cfg.vllm_medium_base_url
        get_sleep_manager().ensure_awake_sync(base_url)

    cache_key = f"{provider}/{model}/{task}/{temperature}"
    if cache_key in _LM_CACHE:
        return _LM_CACHE[cache_key]

    # Providers that expose an OpenAI-compatible API: route through "openai/" prefix
    # so LiteLLM respects the api_key kwarg instead of calling their native SDK
    # (which only reads from environment variables and ignores kwarg).
    _OPENAI_COMPAT_PROVIDERS = {"cerebras", "siliconflow", "vllm"}
    if provider in _OPENAI_COMPAT_PROVIDERS:
        litellm_prefix = "hosted_vllm" if provider == "vllm" else "openai"
    else:
        litellm_prefix = provider
    if model.startswith(f"{litellm_prefix}/") or model.startswith("openai/"):
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
    elif task == "translator" and (cfg.translator_llm_api_key.get_secret_value() or cfg.translator_llm_base_url):
        api_key = cfg.translator_llm_api_key.get_secret_value()
        api_base = cfg.translator_llm_base_url or None
    else:
        # Dynamically resolve credentials by convention: `{provider}_api_key` and `{provider}_base_url`
        # Map fireworks_ai provider to fireworks_api_key config attribute
        attr_prefix = "fireworks" if provider == "fireworks_ai" else provider
        key_attr = f"{attr_prefix}_api_key"
        base_attr = f"{attr_prefix}_base_url"

        secret_obj = getattr(cfg, key_attr, None)
        if secret_obj and hasattr(secret_obj, "get_secret_value"):
            api_key = secret_obj.get_secret_value() or None

        if not api_base:
            api_base = getattr(cfg, base_attr, None) or None

        # Provider-specific base URL fallbacks for providers without a config field
        if not api_base:
            _BASE_URLS = {
                "cerebras": "https://api.cerebras.ai/v1",
                "siliconflow": "https://api.siliconflow.com/v1",
                "ollama": cfg.llm_base_url or "http://localhost:11434",
                "vllm": cfg.vllm_base_url or "http://localhost:8000/v1",
            }
            api_base = _BASE_URLS.get(provider)

        # vLLM needs a dummy key and per-model routing
        if provider == "vllm":
            api_key = api_key or "vllm"
            if model == cfg.llm_small_model:
                api_base = cfg.vllm_small_base_url
            elif model == cfg.llm_medium_model:
                api_base = cfg.vllm_medium_base_url

    # Providers that expose an OpenAI-compatible API but whose native SDK ignores
    # the api_key kwarg and only reads env vars — inject the key into the env before
    # creating the LM so LiteLLM's native provider code can find it.
    _PROVIDER_ENV_VARS: dict[str, str] = {
        "cerebras": "CEREBRAS_API_KEY",
        "siliconflow": "OPENAI_API_KEY",  # used when routed through openai/ prefix
        "groq": "GROQ_API_KEY",
        "fireworks_ai": "FIREWORKS_API_KEY",
    }
    if api_key and provider in _PROVIDER_ENV_VARS:
        import os
        os.environ.setdefault(_PROVIDER_ENV_VARS[provider], api_key)

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
