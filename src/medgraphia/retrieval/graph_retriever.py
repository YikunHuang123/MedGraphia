"""
Bipartite Entity-Chunk graph retriever.

Multi-hop signal comes from graph connectivity — entities co-occurring
through shared chunks — rather than typed relation edges (see project notes
"关系抽取阶段/4. 放弃类型化关系抽取..." for the design rationale). Seeds a
Neo4j GDS Personalized PageRank run at the query's linked entities over a
transient in-memory projection of Entity + Chunk nodes joined by
MENTIONED_IN, then returns the top-ranked chunks.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from medgraphia.domain import EntityType
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_PROJECTION_NODE_LABELS = [t.value for t in EntityType if t is not EntityType.UNKNOWN] + ["Chunk"]


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class ChunkHit:
    """A single PPR-ranked chunk."""

    chunk_id: str
    text: str
    doc_id: str = ""
    source_id: str = ""
    section_path: str = ""
    score: float = 0.0

    def as_text(self) -> str:
        return " ".join(self.text.split())


@dataclass
class GraphRetrievalResult:
    """Aggregated PPR result for a query."""

    seed_cuis: list[str] = field(default_factory=list)
    hits: list[ChunkHit] = field(default_factory=list)

    def as_context_lines(self) -> list[str]:
        return [h.as_text() for h in self.hits]


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_CYPHER_RESOLVE_SEEDS = "MATCH (n) WHERE n.cui IN $cuis RETURN id(n) AS node_id"

_CYPHER_PROJECT = """
CALL gds.graph.project(
  $graph_name,
  $node_labels,
  {MENTIONED_IN: {orientation: 'UNDIRECTED'}}
)
"""

_CYPHER_PPR_STREAM = """
CALL gds.pageRank.stream($graph_name, {
  sourceNodes: $source_node_ids,
  dampingFactor: $damping_factor,
  maxIterations: $max_iterations
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE node:Chunk
RETURN node.chunk_id AS chunk_id, node.text AS text, node.doc_id AS doc_id,
       node.source_id AS source_id, node.section_path AS section_path, score
ORDER BY score DESC
LIMIT $top_k
"""

_CYPHER_DROP_PROJECTION = "CALL gds.graph.drop($graph_name, false)"

# Shortest-path existence check between two seed CUIs. Only excludes FROM_DOC
# (Chunk -> Document, not meaningful for entity connectivity) — MENTIONED_IN
# is left in since chunk co-occurrence IS the connectivity signal now.
_CYPHER_PATH_EXISTS = """
MATCH p = shortestPath((a {{cui: $cui_a}})-[*..{max_hops}]-(b {{cui: $cui_b}}))
WHERE NONE(r IN relationships(p) WHERE type(r) = 'FROM_DOC')
RETURN p IS NOT NULL AS found
LIMIT 1
"""


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------


class GraphRetriever:
    """
    Async bipartite Entity-Chunk retriever using Neo4j GDS Personalized PageRank.

    Usage::

        retriever = GraphRetriever()
        result = await retriever.retrieve(["D001234", "D005014"])
        for hit in result.hits:
            print(hit.as_text())
    """

    def __init__(
        self,
        damping_factor: float = 0.85,
        max_iterations: int = 20,
        top_k_chunks: int = 15,
    ) -> None:
        self._damping_factor = damping_factor
        self._max_iterations = max_iterations
        self._top_k_chunks = top_k_chunks

    @classmethod
    def from_settings(cls) -> GraphRetriever:
        from medgraphia.config import get_settings

        cfg = get_settings()
        return cls(
            damping_factor=cfg.ppr_damping_factor,
            max_iterations=cfg.ppr_max_iterations,
            top_k_chunks=cfg.ppr_top_k_chunks,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        cuis: list[str],
        hops: int = 2,
        user_id: str | None = None,
    ) -> GraphRetrievalResult:
        """
        Run Personalized PageRank seeded at the query's linked entities and
        return the top-ranked chunks.

        Args:
            cuis:    Linked CUIs from query-side NER+EL (PPR seed entities).
            hops:    Wider search for complex queries: scales top_k_chunks
                     (1 = focused, 2 = broader). Values > 2 default back to 2.
            user_id: Optional user ID — historically interesting entities are
                     added as extra (equal-weight) seeds for personalization.
        """
        hops = min(max(hops, 1), 2)
        result = GraphRetrievalResult(seed_cuis=list(cuis))

        if not cuis:
            return result

        seed_cuis = list(cuis)
        if user_id:
            try:
                from medgraphia.graph.queries import get_user_top_interests

                seed_cuis += await get_user_top_interests(user_id, limit=5)
            except Exception as exc:
                logger.debug("ppr_user_interest_fetch_failed", error=str(exc))

        try:
            from medgraphia.graph.client import get_session

            async with get_session() as session:
                seed_result = await session.run(_CYPHER_RESOLVE_SEEDS, cuis=seed_cuis)
                source_node_ids = [r["node_id"] async for r in seed_result]
                if not source_node_ids:
                    return result

                graph_name = f"ppr_{uuid.uuid4().hex}"
                try:
                    await session.run(
                        _CYPHER_PROJECT,
                        graph_name=graph_name,
                        node_labels=_PROJECTION_NODE_LABELS,
                    )
                    ppr_result = await session.run(
                        _CYPHER_PPR_STREAM,
                        graph_name=graph_name,
                        source_node_ids=source_node_ids,
                        damping_factor=self._damping_factor,
                        max_iterations=self._max_iterations,
                        top_k=self._top_k_chunks * hops,
                    )
                    result.hits = [
                        ChunkHit(
                            chunk_id=r["chunk_id"] or "",
                            text=r["text"] or "",
                            doc_id=r["doc_id"] or "",
                            source_id=r["source_id"] or "",
                            section_path=r["section_path"] or "",
                            score=float(r["score"]),
                        )
                        async for r in ppr_result
                        if r["chunk_id"] and r["text"]
                    ]
                finally:
                    await session.run(_CYPHER_DROP_PROJECTION, graph_name=graph_name)

        except Exception as exc:
            logger.warning("ppr_retrieve_failed", cuis=cuis, error=str(exc))

        logger.info("ppr_retriever_done", seeds=len(cuis), hits=len(result.hits))
        return result

    # ------------------------------------------------------------------

    async def check_path_exists(self, cui_a: str, cui_b: str, max_hops: int = 3) -> bool:
        """
        Return True if two CUIs are connected (via chunk co-occurrence or
        community membership) within max_hops. Used as a cheap idempotency
        guard before firing a query-time completion fetch.
        """
        try:
            from medgraphia.graph.client import get_session

            cypher = _CYPHER_PATH_EXISTS.format(max_hops=max(1, max_hops))
            async with get_session() as session:
                result = await session.run(cypher, cui_a=cui_a, cui_b=cui_b)
                record = await result.single()
                return bool(record and record["found"])
        except Exception as exc:
            logger.warning("check_path_exists_failed", cui_a=cui_a, cui_b=cui_b, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Sync convenience wrapper (for use in sync contexts)
    # ------------------------------------------------------------------

    def retrieve_sync(
        self,
        cuis: list[str],
        hops: int = 2,
    ) -> GraphRetrievalResult:
        """Blocking wrapper around retrieve() for non-async callers."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "retrieve_sync() called inside a running event loop. "
                    "Use `await retrieve()` instead."
                )
            return loop.run_until_complete(self.retrieve(cuis, hops=hops))
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("graph_retrieve_sync_failed", error=str(exc))
            return GraphRetrievalResult(seed_cuis=list(cuis))
