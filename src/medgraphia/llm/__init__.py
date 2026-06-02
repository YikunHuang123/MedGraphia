"""LLM provider adapters — unified gateway and legacy client."""

from medgraphia.llm.gateway import (
    CompletionRequest,
    CompletionResponse,
    LiteLLMGateway,
    LLMProvider,
)

# LLMClient uses pydantic-ai which is optional; guard the import
try:
    from medgraphia.llm.client import LLMClient
    __all__ = [
        "LiteLLMGateway",
        "LLMProvider",
        "CompletionRequest",
        "CompletionResponse",
        "LLMClient",
    ]
except ImportError:
    __all__ = [
        "LiteLLMGateway",
        "LLMProvider",
        "CompletionRequest",
        "CompletionResponse",
    ]
