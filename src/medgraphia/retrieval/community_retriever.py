"""
Community retriever.

Retrieves relevant Community summaries for global / cross-corpus queries.
Used when QueryType == CROSS_CORPUS ("overview of...", "what are common
comorbidities of T2DM across all guidelines?").

Two-stage retrieval strategy:
  Stage 1 — Neo4j lookup: pull all Community nodes (summary + member CUIs).
  Stage 2 — Semantic ranking: encode query + community summaries with BGE-M3
             dense embeddings, rank by cosine similarity, return top-K.

Community summaries are stored in Neo4j (written by community_builder.py) and
optionally also indexed in a dedicated Qdrant collection.

This module reads from Neo4j directly to avoid a separate embedding pipeline dependency;

if the Qdrant entity collection happens to contain community vectors in future, a
subclass can override _fetch_communities() to use hybrid search instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class CommunityHit:
    """A single community result."""
    community_id: str
    summary: str
    size: int = 0
    member_cuis: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class CommunityRetrievalResult:
    """Aggregated result from community retrieval."""
    hits: list[CommunityHit] = field(default_factory=list)
    query: str = ""

    @property
    def summaries(self) -> list[str]:
        return [h.summary for h in self.hits]

    def as_context_lines(self) -> list[str]:
        lines = []
        for h in self.hits:
            if h.summary:
                lines.append(f"[Community {h.community_id}] {h.summary}")
        return lines


# ---------------------------------------------------------------------------
# Cypher — fetch all community nodes from Neo4j
# ---------------------------------------------------------------------------

_CYPHER_FETCH_COMMUNITIES = """
MATCH (com:Community)
WHERE com.summary IS NOT NULL AND com.summary <> ''
OPTIONAL MATCH (e)-[:MEMBER_OF]->(com)
WITH com, collect(e.cui) AS member_cuis
RETURN
  com.community_id AS community_id,
  com.summary      AS summary,
  coalesce(com.size, 0) AS size,
  member_cuis
ORDER BY com.size DESC
LIMIT $limit
"""


# ---------------------------------------------------------------------------
# CommunityRetriever
# ---------------------------------------------------------------------------

class CommunityRetriever:
    """
    Retrieves the most relevant Community summaries for a given query.

    Ranking strategy:
      If FlagEmbedding is available: cosine similarity between BGE-M3
      embeddings of the query and each community summary.
      Otherwise: keyword overlap (Jaccard on word sets) as a fast fallback.

    Usage::

        retriever = CommunityRetriever.from_settings()
        result = await retriever.retrieve(
            "common comorbidities of type 2 diabetes", limit=5
        )
        for hit in result.hits:
            print(hit.score, hit.summary)
    """

    def __init__(
        self,
        max_communities: int = 500,
        model_name: str = "BAAI/bge-m3",
    ) -> None:
        """
        Args:
            max_communities: Upper bound on community nodes fetched from Neo4j
                             before re-ranking.  Keeps memory bounded.
            model_name:      BGE-M3 model for semantic similarity ranking.
        """
        self._max_communities = max_communities
        self._model_name = model_name
        self._model: Any = None

    @classmethod
    def from_settings(cls) -> "CommunityRetriever":
        return cls()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> CommunityRetrievalResult:
        """
        Return the top-K community hits most semantically relevant to the query.

        Args:
            query: Raw query text.
            limit: Number of communities to return.

        Returns:
            CommunityRetrievalResult sorted by score descending.
        """
        result = CommunityRetrievalResult(query=query)

        communities = await self._fetch_communities()
        if not communities:
            logger.warning("community_retriever_no_communities")
            return result

        scored = self._rank_communities(query, communities, limit=limit)
        result.hits = scored

        logger.info(
            "community_retriever_done",
            query_len=len(query),
            hits=len(result.hits),
            total_communities=len(communities),
        )
        return result

    # ------------------------------------------------------------------
    # Neo4j fetch
    # ------------------------------------------------------------------

    async def _fetch_communities(self) -> list[dict[str, Any]]:
        """Pull community nodes from Neo4j."""
        communities: list[dict[str, Any]] = []
        try:
            from medgraphia.graph.client import get_session
            async with get_session() as session:
                result = await session.run(
                    _CYPHER_FETCH_COMMUNITIES,
                    limit=self._max_communities,
                )
                async for record in result:
                    communities.append({
                        "community_id": record["community_id"] or "",
                        "summary": record["summary"] or "",
                        "size": int(record["size"] or 0),
                        "member_cuis": list(record["member_cuis"] or []),
                    })
        except Exception as exc:
            logger.warning("community_fetch_failed", error=str(exc))
        return communities

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank_communities(
        self,
        query: str,
        communities: list[dict[str, Any]],
        limit: int,
    ) -> list[CommunityHit]:
        """Rank communities by semantic similarity to the query."""
        try:
            return self._rank_by_embedding(query, communities, limit)
        except Exception as exc:
            logger.warning("community_embed_rank_failed", error=str(exc))
            return self._rank_by_keyword_overlap(query, communities, limit)

    def _rank_by_embedding(
        self,
        query: str,
        communities: list[dict[str, Any]],
        limit: int,
    ) -> list[CommunityHit]:
        """
        Rank communities using BGE-M3 dense cosine similarity.
        """
        self._load_model()

        summaries = [c["summary"] for c in communities]
        all_texts = [query] + summaries

        output = self._model.encode(
            all_texts,
            batch_size=min(32, len(all_texts)),
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        vecs = output["dense_vecs"]
        query_vec = vecs[0]
        community_vecs = vecs[1:]

        # Cosine similarity (vectors are already L2-normalised by BGE-M3)
        scores = (community_vecs @ query_vec).tolist()

        ranked = sorted(
            zip(scores, communities),
            key=lambda x: -x[0],
        )[:limit]

        return [
            CommunityHit(
                community_id=c["community_id"],
                summary=c["summary"],
                size=c["size"],
                member_cuis=c["member_cuis"],
                score=float(score),
            )
            for score, c in ranked
        ]

    def _rank_by_keyword_overlap(
        self,
        query: str,
        communities: list[dict[str, Any]],
        limit: int,
    ) -> list[CommunityHit]:
        """
        Fallback ranking: Jaccard similarity on word-level token sets.
        Fast but less accurate than embedding-based ranking.
        """
        query_tokens = set(query.lower().split())

        def jaccard(summary: str) -> float:
            if not summary:
                return 0.0
            summary_tokens = set(summary.lower().split())
            intersection = len(query_tokens & summary_tokens)
            union = len(query_tokens | summary_tokens)
            return intersection / union if union else 0.0

        ranked = sorted(communities, key=lambda c: -jaccard(c["summary"]))[:limit]
        return [
            CommunityHit(
                community_id=c["community_id"],
                summary=c["summary"],
                size=c["size"],
                member_cuis=c["member_cuis"],
                score=jaccard(c["summary"]),
            )
            for c in ranked
        ]

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is not installed. Run: pip install FlagEmbedding"
            ) from exc
        logger.info("community_retriever_loading_model", model=self._model_name)
        self._model = BGEM3FlagModel(self._model_name, use_fp16=True)
        logger.info("community_retriever_model_loaded")
