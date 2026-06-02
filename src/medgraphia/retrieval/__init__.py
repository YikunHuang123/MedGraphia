"""Online query pipeline: three-path retrieval, RRF fusion, reranking."""

from medgraphia.retrieval.query_ner import QueryEntities, QueryNERLinker
from medgraphia.retrieval.router import QueryRouter, RetrievalPlan, RouterState
from medgraphia.retrieval.graph_retriever import (
    GraphRetriever,
    GraphRetrievalResult,
    GraphTriple,
)
from medgraphia.retrieval.vector_retriever import (
    VectorRetriever,
    VectorRetrievalResult,
    VectorHit,
)
from medgraphia.retrieval.community_retriever import (
    CommunityRetriever,
    CommunityRetrievalResult,
    CommunityHit,
)
from medgraphia.retrieval.fusion import (
    RRFFusion,
    FusionResult,
    FusedItem,
    RetrievalSource,
)
from medgraphia.retrieval.reranker import Reranker, RerankedResult
from medgraphia.retrieval.pipeline import RetrievalPipeline

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
    "GraphTriple",
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
