"""Online query pipeline: three-path retrieval, RRF fusion, reranking."""

from medgraphia.retrieval.community_retriever import (
    CommunityHit,
    CommunityRetrievalResult,
    CommunityRetriever,
)
from medgraphia.retrieval.fusion import (
    FusedItem,
    FusionResult,
    RetrievalSource,
    RRFFusion,
)
from medgraphia.retrieval.graph_retriever import (
    ChunkHit,
    GraphRetrievalResult,
    GraphRetriever,
)
from medgraphia.retrieval.pipeline import RetrievalPipeline
from medgraphia.retrieval.query_ner import QueryEntities, QueryNERLinker
from medgraphia.retrieval.reranker import RerankedResult, Reranker
from medgraphia.retrieval.router import QueryRouter, RetrievalPlan, RouterState
from medgraphia.retrieval.vector_retriever import (
    VectorHit,
    VectorRetrievalResult,
    VectorRetriever,
)

__all__ = [
    # Pipeline
    "RetrievalPipeline",
    # Query NER + EL
    "QueryEntities",
    "QueryNERLinker",
    # Router
    "QueryRouter",
    "RetrievalPlan",
    "RouterState",
    # Graph retriever
    "GraphRetriever",
    "GraphRetrievalResult",
    "ChunkHit",
    # Vector retriever
    "VectorRetriever",
    "VectorRetrievalResult",
    "VectorHit",
    # Community retriever
    "CommunityRetriever",
    "CommunityRetrievalResult",
    "CommunityHit",
    # RRF fusion
    "RRFFusion",
    "FusionResult",
    "FusedItem",
    "RetrievalSource",
    # Reranker
    "Reranker",
    "RerankedResult",
]
