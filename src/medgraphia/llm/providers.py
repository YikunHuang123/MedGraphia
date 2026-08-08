"""Central LLM/reranker provider registry — single place to add a new provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_UNSET = ""


@dataclass(frozen=True)
class ProviderSpec:
    api_key_field: str = _UNSET  # config.py SecretStr attribute name; "" if no key needed
    base_url_field: str = _UNSET  # config.py str attribute name for a user-overridable base_url
    default_base_url: str = _UNSET  # fallback when base_url_field is unset/empty
    litellm_prefix: str = _UNSET  # "" => use the provider name itself as the litellm prefix
    openai_compat: bool = False  # route through litellm's "openai/" prefix instead of the native one
    env_var: str = _UNSET  # some providers' SDK path ignores the api_key kwarg and only reads this


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(api_key_field="openai_api_key", base_url_field="openai_base_url"),
    "anthropic": ProviderSpec(api_key_field="anthropic_api_key"),
    "deepseek": ProviderSpec(api_key_field="deepseek_api_key", default_base_url="https://api.deepseek.com"),
    "gemini": ProviderSpec(api_key_field="gemini_api_key"),
    "groq": ProviderSpec(api_key_field="groq_api_key", env_var="GROQ_API_KEY"),
    "openrouter": ProviderSpec(api_key_field="openrouter_api_key", default_base_url="https://openrouter.ai/api/v1"),
    "fireworks_ai": ProviderSpec(
        api_key_field="fireworks_api_key",
        default_base_url="https://api.fireworks.ai/inference/v1",
        env_var="FIREWORKS_API_KEY",
    ),
    "together_ai": ProviderSpec(api_key_field="together_api_key"),
    "cerebras": ProviderSpec(
        api_key_field="cerebras_api_key",
        default_base_url="https://api.cerebras.ai/v1",
        openai_compat=True,
        env_var="CEREBRAS_API_KEY",
    ),
    "siliconflow": ProviderSpec(
        api_key_field="siliconflow_api_key",
        default_base_url="https://api.siliconflow.com/v1",
        openai_compat=True,
        env_var="OPENAI_API_KEY",
    ),
    "ollama": ProviderSpec(base_url_field="llm_base_url", default_base_url="http://localhost:11434"),
    "vllm": ProviderSpec(litellm_prefix="hosted_vllm"),
    # Reranker-only providers (no chat/DSPy usage, no litellm routing needed)
    "jina": ProviderSpec(api_key_field="jina_api_key", default_base_url="https://api.jina.ai/v1"),
    "cohere": ProviderSpec(api_key_field="cohere_api_key", default_base_url="https://api.cohere.ai/v1"),
}


@dataclass
class ProviderCredentials:
    api_key: str | None
    base_url: str | None
    litellm_prefix: str


def resolve_credentials(provider: str, cfg: Any) -> ProviderCredentials:
    """
    Resolve (api_key, base_url, litellm_prefix) for a provider from config,
    injecting into the process environment where a provider's own SDK path
    ignores an explicit api_key kwarg and only reads env vars.
    """
    spec = PROVIDER_REGISTRY.get(provider)
    if spec is None:
        return ProviderCredentials(api_key=None, base_url=None, litellm_prefix=provider)

    api_key: str | None = None
    if spec.api_key_field:
        secret = getattr(cfg, spec.api_key_field, None)
        if secret is not None and hasattr(secret, "get_secret_value"):
            api_key = secret.get_secret_value() or None

    base_url: str | None = None
    if spec.base_url_field:
        base_url = getattr(cfg, spec.base_url_field, None) or None
    base_url = base_url or (spec.default_base_url or None)

    if api_key and spec.env_var:
        import os

        os.environ.setdefault(spec.env_var, api_key)

    litellm_prefix = "openai" if spec.openai_compat else (spec.litellm_prefix or provider)
    return ProviderCredentials(api_key=api_key, base_url=base_url, litellm_prefix=litellm_prefix)


def build_litellm_model_id(provider: str, model_name: str) -> str:
    """Map (provider, model_name) to the model string litellm expects."""
    if not model_name:
        return ""
    if provider == "openai" and "/" not in model_name:
        return model_name

    spec = PROVIDER_REGISTRY.get(provider)
    if spec and spec.openai_compat:
        prefix = "openai"
    elif spec and spec.litellm_prefix:
        prefix = spec.litellm_prefix
    else:
        prefix = provider

    if model_name.startswith(f"{prefix}/") or model_name.startswith("openai/"):
        return model_name
    return f"{prefix}/{model_name}"
