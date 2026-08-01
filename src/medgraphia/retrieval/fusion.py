"""
Reciprocal Rank Fusion (RRF).

Merges results from the three retrieval paths into a single ranked list.

  Path 1 — Graph retriever   : GraphRetrievalResult  (PPR-ranked chunks)
  Path 2 — Vector retriever  : VectorRetrievalResult (dense+sparse chunks)
  Path 3 — Community retriever: CommunityRetrievalResult (community summaries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)

_RRF_K: int = 60  # smoothing constant


# ---------------------------------------------------------------------------
# Unified item type
# ---------------------------------------------------------------------------


class RetrievalSource(str, Enum):
    GRAPH = "graph"
    VECTOR = "vector"
    COMMUNITY = "community"


@dataclass
class FusedItem:
    """
    A single candidate context piece after RRF fusion.

    Attributes:
        item_id     — Unique identifier within its source (chunk_id / CUI triple / community_id).
        text        — Text content for the LLM context window.
        source      — Which retrieval path produced this item.
        rrf_score   — Aggregated RRF score (higher = more relevant across paths).
        metadata    — Source-specific provenance (e.g. chunk_id, source_title, cuis).
    """

    item_id: str
    text: str
    source: RetrievalSource
    rrf_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_context_str(self) -> str:
        """Human-readable one-liner for LLM context."""
        source_tag = f"[{self.source.value}]"
        return f"{source_tag} {self.text}"


@dataclass
class FusionResult:
    """Final ranked list after RRF fusion (and before optional reranking)."""

    items: list[FusedItem] = field(default_factory=list)
    query: str = ""

    def top(self, n: int) -> FusionResult:
        return FusionResult(items=self.items[:n], query=self.query)

    def texts(self) -> list[str]:
        return [it.text for it in self.items]

    def as_context_lines(self) -> list[str]:
        return [it.as_context_str() for it in self.items]


# ---------------------------------------------------------------------------
# Adapters: each retrieval result → list[FusedItem]
# ---------------------------------------------------------------------------


def _graph_to_items(result: Any) -> list[FusedItem]:
    """Convert GraphRetrievalResult (PPR-ranked chunks) → FusedItem list."""
    from medgraphia.retrieval.graph_retriever import GraphRetrievalResult

    if not isinstance(result, GraphRetrievalResult):
        return []

    items: list[FusedItem] = []
    for hit in result.hits:
        # Using chunk_id as item_id lets a PPR-ranked chunk overlap with the
        # same chunk from vector search, boosting "graph-confirmed" chunks.
        items.append(
            FusedItem(
                item_id=hit.chunk_id,
                text=hit.as_text(),
                source=RetrievalSource.GRAPH,
                metadata={
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "source_id": hit.source_id,
                    "section_path": hit.section_path,
                    "ppr_score": hit.score,
                },
            )
        )
    return items


def _vector_to_items(result: Any) -> list[FusedItem]:
    """Convert VectorRetrievalResult → FusedItem list."""
    from medgraphia.retrieval.vector_retriever import VectorRetrievalResult

    if not isinstance(result, VectorRetrievalResult):
        return []

    items: list[FusedItem] = []
    for hit in result.hits:
        # Use full section text when available; fall back to the child chunk text.
        # This gives the reranker and LLM complete context while vector search
        # still matched on the focused child chunk.
        display_text = hit.parent_text or hit.text
        items.append(
            FusedItem(
                item_id=hit.chunk_id,
                text=display_text,
                source=RetrievalSource.VECTOR,
                metadata={
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "source_id": hit.source_id,
                    "source_title": hit.source_title,
                    "source_version": hit.source_version,
                    "section_path": hit.section_path,
                    "language": hit.language,
                    "page": hit.page,
                    "original_score": hit.score,
                    "chunk_text": hit.text,  # original child chunk for provenance
                },
            )
        )
    return items


def _community_to_items(result: Any) -> list[FusedItem]:
    """Convert CommunityRetrievalResult → FusedItem list."""
    from medgraphia.retrieval.community_retriever import CommunityRetrievalResult

    if not isinstance(result, CommunityRetrievalResult):
        return []

    items: list[FusedItem] = []
    for hit in result.hits:
        items.append(
            FusedItem(
                item_id=hit.community_id,
                text=hit.summary,
                source=RetrievalSource.COMMUNITY,
                metadata={
                    "community_id": hit.community_id,
                    "size": hit.size,
                    "member_cuis": hit.member_cuis,
                    "original_score": hit.score,
                },
            )
        )
    return items


def chunk_to_fused_item(chunk: Any) -> FusedItem:
    """Wrap a freshly-ingested Chunk (e.g. from agentic gap completion) as a
    FusedItem so it can be merged into an existing retrieved_items list and
    survive citation.py's numbering/lookup, which expects FusedItem shape."""
    return FusedItem(
        item_id=chunk.chunk_id,
        text=chunk.text,
        source=RetrievalSource.GRAPH,
        metadata={
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "source_id": chunk.source.source_id,
            "source_title": chunk.source.source_title,
            "section_path": chunk.section_path,
        },
    )


# ---------------------------------------------------------------------------
# Core RRF computation
# ---------------------------------------------------------------------------


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """1-indexed rank → RRF contribution."""
    return 1.0 / (k + rank)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[FusedItem]],
    k: int = _RRF_K,
) -> list[FusedItem]:
    """
    Fuse multiple ranked lists with RRF.

    Args:
        ranked_lists: Each inner list is a ranked list of FusedItem (best first).
        k:            Smoothing constant (default 60).

    Returns:
        Merged list sorted by RRF score descending, with rrf_score set.
    """
    scores: dict[str, float] = {}
    items_by_id: dict[str, FusedItem] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            uid = item.item_id
            scores[uid] = scores.get(uid, 0.0) + _rrf_score(rank, k)
            if uid not in items_by_id:
                items_by_id[uid] = item

    merged = []
    for uid, score in sorted(scores.items(), key=lambda x: -x[1]):
        fused = items_by_id[uid]
        fused.rrf_score = score
        merged.append(fused)

    return merged


# ---------------------------------------------------------------------------
# RRFFusion — public interface
# ---------------------------------------------------------------------------


class RRFFusion:
    """
    Fuses results from graph, vector, and community retrievers using RRF.

    Usage::

        fusion = RRFFusion()
        result = fusion.fuse(
            query="interaction between metformin and sitagliptin",
            graph_result=graph_result,
            vector_result=vector_result,
            community_result=None,
        )
        top5 = result.top(5)
    """

    def __init__(self, k: int = _RRF_K) -> None:
        """
        Args:
            k: RRF smoothing constant.  Default 60 is from the original paper.
               Increasing k reduces the relative importance of top-rank items.
        """
        self._k = k

    @classmethod
    def from_settings(cls) -> RRFFusion:
        return cls()

    def fuse(
        self,
        query: str,
        graph_result: Any | None = None,
        vector_result: Any | None = None,
        community_result: Any | None = None,
    ) -> FusionResult:
        """
        Merge up to three retrieval results into a single ranked FusionResult.

        Args:
            query:            Original query string (stored in FusionResult).
            graph_result:     GraphRetrievalResult or None.
            vector_result:    VectorRetrievalResult or None.
            community_result: CommunityRetrievalResult or None.

        Returns:
            FusionResult sorted by RRF score descending.
        """
        ranked_lists: list[list[FusedItem]] = []

        if graph_result is not None:
            items = _graph_to_items(graph_result)
            if items:
                ranked_lists.append(items)
                logger.debug("rrf_input_graph", items=len(items))

        if vector_result is not None:
            items = _vector_to_items(vector_result)
            if items:
                ranked_lists.append(items)
                logger.debug("rrf_input_vector", items=len(items))

        if community_result is not None:
            items = _community_to_items(community_result)
            if items:
                ranked_lists.append(items)
                logger.debug("rrf_input_community", items=len(items))

        if not ranked_lists:
            logger.warning("rrf_no_inputs")
            return FusionResult(query=query)

        merged = _reciprocal_rank_fusion(ranked_lists, k=self._k)

        logger.info(
            "rrf_done",
            sources=len(ranked_lists),
            merged=len(merged),
        )
        return FusionResult(items=merged, query=query)
