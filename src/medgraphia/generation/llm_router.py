"""
LLM Router — selects provider + model based on query complexity and language.

Routing strategy:

Primary source (preferred): DSPy-assessed complexity_tier from the rewriter.
  The rewriter scores each query using a quantified rubric (E+I=Total) and
  returns SMALL / MEDIUM / LARGE directly.

Fallback (when tier is None): static QueryType → ModelTier table.
QueryType                 → ModelTier   Rationale
──────────────────────────────────────────────────────────────────────
PATIENT_FAQ               → SMALL       Low stakes, high volume; cheap fast model
DRUG_INTERACTION          → MEDIUM      Factual precision required; moderate model
LITERATURE_MULTIHOP       → LARGE       Multi-step reasoning; needs best model
CLINICAL_DECISION         → LARGE       High-risk clinical guidance; best model only
CROSS_CORPUS              → LARGE       Global synthesis; best model

Language override (LARGE tier only)
──────────────────────────────────────
zh  → prefer DeepSeek / Qwen  (strongest Chinese medical reasoning)
de  → prefer OpenAI GPT-4o    (best German coverage among accessible models)
en  → follow configured large-tier defaults

Model tiers are configured via Settings (with sensible defaults):
  LLM_SMALL_PROVIDER  / LLM_SMALL_MODEL
  LLM_MEDIUM_PROVIDER / LLM_MEDIUM_MODEL
  LLM_LARGE_PROVIDER  / LLM_LARGE_MODEL

If the preferred language-override provider lacks an API key, the router
falls back to the configured large-tier model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from medgraphia.domain.base import Language, QueryType
from medgraphia.llm.gateway import (
    LiteLLMGateway,
    LLMProvider,
)
from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model-tier enum
# ---------------------------------------------------------------------------


class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# ---------------------------------------------------------------------------
# Static routing tables
# ---------------------------------------------------------------------------

_QUERY_TYPE_TIER: dict[QueryType, ModelTier] = {
    QueryType.PATIENT_FAQ: ModelTier.SMALL,
    QueryType.DRUG_INTERACTION: ModelTier.MEDIUM,
    QueryType.LITERATURE_MULTIHOP: ModelTier.LARGE,
    QueryType.CLINICAL_DECISION: ModelTier.LARGE,
    QueryType.CROSS_CORPUS: ModelTier.LARGE,
}

# Preferred (provider, model) per language for the LARGE tier.
# Only used when the provider's API key is available.
# Note: English (EN) is omitted to respect the user's global LARGE-tier default.
_LANG_LARGE_PREF: dict[Language, tuple[LLMProvider, str]] = {
    Language.ZH: (LLMProvider.DEEPSEEK, "deepseek-v4-pro"),
    Language.DE: (LLMProvider.OPENAI, "gpt-4o"),
}


# ---------------------------------------------------------------------------
# Routing decision dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingDecision:
    """
    Describes which model was selected and why.

    Attributes:
        provider    — LLMProvider enum member
        model_name  — Exact model identifier string
        tier        — Which tier was selected
        query_type  — Originating query type
        language    — Language that influenced the decision
        reason      — Human-readable explanation for tracing / observability
    """

    provider: LLMProvider
    model_name: str
    tier: ModelTier
    query_type: QueryType
    language: Language
    reason: str = ""


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------


class LLMRouter:
    """
    Routes a (QueryType, Language) pair to a (LiteLLMGateway, RoutingDecision).

    Usage::

        router = LLMRouter.from_settings()
        gateway, decision = router.route(QueryType.CLINICAL_DECISION, Language.ZH)
        resp = await gateway.acomplete(CompletionRequest(
            system_prompt="...",
            user_prompt="...",
        ))
    """

    def __init__(
        self,
        tier_models: dict[ModelTier, tuple[LLMProvider, str]],
        respect_language_preference: bool = True,
    ) -> None:
        """
        Args:
            tier_models:                  {tier → (provider, model_name)} mapping
            respect_language_preference:  Whether to apply language-specific LARGE-tier overrides
        """
        self._tier_models = tier_models
        self._respect_lang = respect_language_preference

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> LLMRouter:
        """
        Construct a router from Settings.

        Reads optional tier-specific env vars, falling back to the global
        default_llm_provider / default_llm_model when not set.
        """
        from medgraphia.config import get_settings

        cfg = get_settings()

        default_provider = _parse_provider(cfg.default_llm_provider)
        default_model = cfg.default_llm_model

        def _tier(provider_attr: str, model_attr: str) -> tuple[LLMProvider, str]:
            p_str = getattr(cfg, provider_attr, None) or cfg.default_llm_provider
            m_str = getattr(cfg, model_attr, None) or default_model
            return _parse_provider(p_str), m_str

        tier_models: dict[ModelTier, tuple[LLMProvider, str]] = {
            ModelTier.SMALL: _tier("llm_small_provider", "llm_small_model"),
            ModelTier.MEDIUM: _tier("llm_medium_provider", "llm_medium_model"),
            ModelTier.LARGE: _tier("llm_large_provider", "llm_large_model"),
        }

        logger.info(
            "llm_router_init",
            small=f"{tier_models[ModelTier.SMALL][0].value}/{tier_models[ModelTier.SMALL][1]}",
            medium=f"{tier_models[ModelTier.MEDIUM][0].value}/{tier_models[ModelTier.MEDIUM][1]}",
            large=f"{tier_models[ModelTier.LARGE][0].value}/{tier_models[ModelTier.LARGE][1]}",
        )
        return cls(tier_models=tier_models)

    # ------------------------------------------------------------------
    # Route
    # ------------------------------------------------------------------

    def route(
        self,
        query_type: QueryType,
        language: Language = Language.EN,
        tier: ModelTier | None = None,
    ) -> tuple[LiteLLMGateway, RoutingDecision]:
        """
        Return (gateway, decision) for the given query type and language.

        Args:
            query_type: Intent class from the retrieval router.
            language:   Detected query language (influences LARGE-tier model choice).
            tier:       DSPy-assessed complexity tier from the rewriter.  When
                        provided it takes precedence over the static QueryType
                        mapping.  Pass None to fall back to the static table.
        """
        from medgraphia.config import get_settings

        cfg = get_settings()

        if tier is not None:
            # DSPy-assessed tier takes priority — more accurate than the static table
            reason = f"dspy_tier={tier.value} (query_type={query_type.value})"
        else:
            # Fall back to the static QueryType → ModelTier table
            tier = _QUERY_TYPE_TIER.get(query_type, ModelTier.MEDIUM)
            reason = f"static_table: query_type={query_type.value} → tier={tier.value}"

        provider, model_name = self._tier_models[tier]

        # ── Language Preference Override ─────────────────────────────────────
        # Only apply if:
        # 1. It's a LARGE tier query
        # 2. We are in 'respect_language' mode
        # 3. The user HAS NOT explicitly set a tier-specific provider

        is_explicit_config = False
        if tier == ModelTier.LARGE and cfg.llm_large_provider:
            is_explicit_config = True
        elif tier == ModelTier.MEDIUM and cfg.llm_medium_provider:
            is_explicit_config = True
        elif tier == ModelTier.SMALL and cfg.llm_small_provider:
            is_explicit_config = True

        if tier == ModelTier.LARGE and self._respect_lang and not is_explicit_config:
            lang_pref = _LANG_LARGE_PREF.get(language)
            if lang_pref is not None:
                pref_provider, pref_model = lang_pref
                if _api_key_available(pref_provider):
                    provider, model_name = pref_provider, pref_model
                    reason += f"; lang_override={language.value}→{provider.value}/{model_name}"
                else:
                    reason += f"; lang_override={language.value} skipped (no key)"
        elif is_explicit_config:
            reason += "; using explicit tier config"

        decision = RoutingDecision(
            provider=provider,
            model_name=model_name,
            tier=tier,
            query_type=query_type,
            language=language,
            reason=reason,
        )

        logger.info(
            "llm_routed",
            provider=provider.value,
            model=model_name,
            tier=tier.value,
            query_type=query_type.value,
            language=language.value,
        )

        gateway = LiteLLMGateway.for_provider(provider=provider, model_name=model_name)
        return gateway, decision

    # ------------------------------------------------------------------
    # Convenience: route and get gateway only
    # ------------------------------------------------------------------

    def get_gateway(
        self,
        query_type: QueryType,
        language: Language = Language.EN,
        tier: ModelTier | None = None,
    ) -> LiteLLMGateway:
        gateway, _ = self.route(query_type, language, tier=tier)
        return gateway


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_provider(value: str) -> LLMProvider:
    """Parse provider string, defaulting to OLLAMA for unknown values."""
    if value == "fireworks":
        value = "fireworks_ai"
    try:
        return LLMProvider(value)
    except ValueError:
        logger.warning("unknown_provider_in_config", value=value, fallback="ollama")
        return LLMProvider.OLLAMA


def _api_key_available(provider: LLMProvider) -> bool:
    """Return True if a non-empty API key is configured for the provider."""
    from medgraphia.config import get_settings

    cfg = get_settings()

    def get_val(attr_name: str) -> str | None:
        obj = getattr(cfg, attr_name, None)
        if obj and hasattr(obj, "get_secret_value"):
            return obj.get_secret_value()
        return None

    match provider:
        case LLMProvider.OPENAI:
            return bool(get_val("openai_api_key"))
        case LLMProvider.ANTHROPIC:
            return bool(get_val("anthropic_api_key"))
        case LLMProvider.DEEPSEEK:
            return bool(get_val("deepseek_api_key"))
        case LLMProvider.GROQ:
            return bool(get_val("groq_api_key"))
        case LLMProvider.GEMINI:
            return bool(get_val("gemini_api_key"))
        case LLMProvider.FIREWORKS_AI:
            return bool(get_val("fireworks_api_key"))
        case LLMProvider.SILICONFLOW:
            return bool(get_val("siliconflow_api_key"))
        case LLMProvider.CEREBRAS:
            return bool(get_val("cerebras_api_key"))
        case LLMProvider.TOGETHER_AI:
            return bool(get_val("together_api_key"))
        case LLMProvider.OLLAMA:
            return True  # Ollama is always "available" (runs locally, no key needed)
    return False
