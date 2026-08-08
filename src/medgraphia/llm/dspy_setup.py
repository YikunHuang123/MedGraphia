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

    from medgraphia.llm.providers import build_litellm_model_id, resolve_credentials

    model_id = build_litellm_model_id(provider, model)

    # 2. Determine API key and Base URL — task-specific credentials take
    # priority over the provider's registry defaults (llm/providers.py).
    api_key = None
    api_base = None
    task_creds = {
        "rewriter": (cfg.rewriter_llm_api_key, cfg.rewriter_llm_base_url),
        "summarizer": (cfg.summarizer_llm_api_key, cfg.summarizer_llm_base_url),
        "judge": (cfg.judge_llm_api_key, cfg.judge_llm_base_url),
        "translator": (cfg.translator_llm_api_key, cfg.translator_llm_base_url),
    }
    if task in task_creds:
        secret, base = task_creds[task]
        if secret.get_secret_value() or base:
            api_key = secret.get_secret_value() or None
            api_base = base or None

    if api_key is None and api_base is None:
        creds = resolve_credentials(provider, cfg)
        api_key, api_base = creds.api_key, creds.base_url

    # vLLM per-tier routing overrides the registry's generic base_url.
    if provider == "vllm":
        api_key = api_key or "vllm"
        if model == cfg.llm_small_model:
            api_base = cfg.vllm_small_base_url
        elif model == cfg.llm_medium_model:
            api_base = cfg.vllm_medium_base_url

    try:
        kwargs = {"max_tokens": cfg.llm_max_tokens}
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
