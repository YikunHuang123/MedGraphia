"""LLM generation layer: providers, router, prompts, citation injection."""

from medgraphia.generation.citation import (
    CitationResult,
    build_numbered_context,
    inject_citations,
)
from medgraphia.generation.llm_router import (
    LLMRouter,
    ModelTier,
    RoutingDecision,
)
from medgraphia.prompts import (
    MedicalAnswer,
)
from medgraphia.llm.gateway import (
    CompletionRequest,
    CompletionResponse,
    LiteLLMGateway,
    LLMProvider,
)

__all__ = [
    # Gateway
    "LiteLLMGateway",
    "LLMProvider",
    "CompletionRequest",
    "CompletionResponse",
    # Router
    "LLMRouter",
    "ModelTier",
    "RoutingDecision",
    # Prompts
    "MedicalAnswer",
    # Citations
    "inject_citations",
    "build_numbered_context",
    "CitationResult",
]
