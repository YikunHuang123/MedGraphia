"""
Graph retriever.

Given a list of seed CUIs (from query-side NER + EL), expands a 1- or 2-hop
subgraph in Neo4j and converts the result into a flat list of
(entity, relation_type, neighbor) triples suitable for downstream context
construction.

Output type:  list[GraphTriple]
  entity        — starting CUI node (always a seed or intermediate)
  relation_type — Cypher relationship type string (e.g. "TREATS")
  neighbor      — neighbor node dict {cui, label, entity_type, ...}
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class GraphTriple:
    """A single (entity, relation, neighbor) triple from graph traversal."""

    entity_cui: str
    entity_label: str
    entity_type: str
    relation_type: str
    neighbor_cui: str
    neighbor_label: str
    neighbor_type: str
    # Edge-level provenance
    evidence_text: str = ""
    confidence: float = 1.0
    chunk_id: str = ""

    def as_text(self) -> str:
        """Render into a natural language statement for the LLM context."""
        base = (
            f"{self.entity_label} ({self.entity_type}) "
            f"--[{self.relation_type}]--> "
            f"{self.neighbor_label} ({self.neighbor_type})"
        )
        if self.evidence_text:
            # Clean up text (remove excessive whitespace/newlines)
            clean_evidence = " ".join(self.evidence_text.split())
            return f'{base}. Evidence: "{clean_evidence}"'
        return base


@dataclass
class GraphRetrievalResult:
    """Aggregated result from all seed CUIs."""

    triples: list[GraphTriple] = field(default_factory=list)
    seed_cuis: list[str] = field(default_factory=list)
    hops: int = 2

    @property
    def unique_entity_cuis(self) -> set[str]:
        cuis: set[str] = set()
        for t in self.triples:
            cuis.add(t.entity_cui)
            cuis.add(t.neighbor_cui)
        return cuis

    def as_context_lines(self) -> list[str]:
        """Return deduplicated text lines for LLM context injection."""
        seen: set[str] = set()
        lines: list[str] = []
        for t in self.triples:
            line = t.as_text()
            if line not in seen:
                seen.add(line)
                lines.append(line)
        return lines


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

# 1-hop query: direct neighbours with optional user interest boost
_CYPHER_1_HOP = """
MATCH (start {cui: $cui})-[r]-(neighbor)
WHERE neighbor.cui IS NOT NULL
  AND NOT r:MENTIONED_IN
  AND NOT r:FROM_DOC
OPTIONAL MATCH (u:User {id: $user_id})-[interest:INTERESTED_IN]->(neighbor)
RETURN
  start.cui           AS entity_cui,
  start.label         AS entity_label,
  labels(start)[0]    AS entity_type,
  type(r)             AS relation_type,
  neighbor.cui        AS neighbor_cui,
  neighbor.label      AS neighbor_label,
  labels(neighbor)[0] AS neighbor_type,
  coalesce(r.evidence_text, '')  AS evidence_text,
  coalesce(r.confidence, 1.0)    AS confidence,
  coalesce(r.chunk_id, '')       AS chunk_id,
  coalesce(interest.weight, 0.0) AS interest_weight
ORDER BY interest_weight DESC, confidence DESC
LIMIT $limit
"""

# Node Summary query: Get the core information about a node and its most important neighbors
_CYPHER_NODE_SUMMARY = """
MATCH (n {cui: $cui})
OPTIONAL MATCH (n)-[r]-(m)
WHERE NOT type(r) IN ['MENTIONED_IN', 'FROM_DOC']
WITH n, m, r, labels(n)[0] AS n_type, labels(m)[0] AS m_type
ORDER BY r.confidence DESC
WITH n, n_type, collect({rel: type(r), target: m.label, target_type: m_type})[0..10] AS rels
RETURN 
  n.cui AS cui,
  n.label AS label,
  n_type AS type,
  rels AS top_relations
"""

# Shortest-path existence check between two seed CUIs, ignoring structural
# edges. Independent of the hop-limited retrieval above, so it can see
# further than whatever `hops` the caller's retrieval used.
# Cypher requires the variable-length range bound to be a literal, not a
# parameter, hence the {max_hops} placeholder filled in at call time.
_CYPHER_PATH_EXISTS = """
MATCH p = shortestPath((a {{cui: $cui_a}})-[*..{max_hops}]-(b {{cui: $cui_b}}))
WHERE NONE(r IN relationships(p) WHERE type(r) IN ['MENTIONED_IN', 'FROM_DOC'])
RETURN p IS NOT NULL AS found
LIMIT 1
"""

# 2-hop query: follows intermediaries with optional user interest boost
_CYPHER_2_HOP = """
MATCH (start {cui: $cui})-[r1]-(mid)-[r2]-(leaf)
WHERE mid.cui IS NOT NULL
  AND leaf.cui IS NOT NULL
  AND leaf.cui <> $cui
  AND NOT type(r1) IN ['MENTIONED_IN', 'FROM_DOC']
  AND NOT type(r2) IN ['MENTIONED_IN', 'FROM_DOC']
OPTIONAL MATCH (u:User {id: $user_id})-[interest:INTERESTED_IN]->(leaf)
RETURN
  start.cui           AS entity_cui,
  start.label         AS entity_label,
  labels(start)[0]    AS entity_type,
  type(r1) + '/' + type(r2)  AS relation_type,
  leaf.cui            AS neighbor_cui,
  leaf.label          AS neighbor_label,
  labels(leaf)[0]     AS neighbor_type,
  ''                  AS evidence_text,
  1.0                 AS confidence,
  ''                  AS chunk_id,
  coalesce(interest.weight, 0.0) AS interest_weight
ORDER BY interest_weight DESC, confidence DESC
LIMIT $limit
"""


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------


class GraphRetriever:
    """
    Async graph retriever using Neo4j.

    Usage::

        retriever = GraphRetriever()
        result = await retriever.retrieve(["D001234", "D005014"], hops=2)
        for triple in result.triples:
            print(triple.as_text())
    """

    def __init__(self, per_seed_limit: int = 100) -> None:
        """
        Args:
            per_seed_limit: Maximum Cypher rows returned per seed CUI per hop level.
                            Total triples = len(cuis) × hops × limit.
        """
        self._per_seed_limit = per_seed_limit

    @classmethod
    def from_settings(cls) -> GraphRetriever:
        return cls()

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
        Expand subgraph for all seed CUIs.

        Args:
            cuis:     Linked CUIs from query-side NER+EL.
            hops:     1 or 2.  Values > 2 default back to 2.
            user_id:  Optional user ID to boost historically relevant entities.

        Returns:
            GraphRetrievalResult with deduplicated triples.
        """
        hops = min(max(hops, 1), 2)
        result = GraphRetrievalResult(seed_cuis=list(cuis), hops=hops)

        if not cuis:
            return result

        # Expand all seeds concurrently
        tasks = [self._expand_one(cui, hops, user_id) for cui in cuis]
        per_seed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results from all seeds and deduplicate by relationship (A, REL, B).
        # We group by relationship and pick the triple with the highest confidence
        # to ensure we show the strongest evidence to the LLM.
        rel_map: dict[tuple[str, str, str], GraphTriple] = {}

        for item in per_seed_results:
            if isinstance(item, Exception):
                logger.warning("graph_retriever_task_failed", error=str(item))
                continue

            for triple in item:
                key = (triple.entity_cui, triple.relation_type, triple.neighbor_cui)

                if key not in rel_map:
                    rel_map[key] = triple
                else:
                    # Update if current triple has higher confidence or better evidence
                    existing = rel_map[key]
                    if triple.confidence > existing.confidence:
                        rel_map[key] = triple
                    elif not existing.evidence_text and triple.evidence_text:
                        rel_map[key] = triple

        result.triples = list(rel_map.values())

        logger.info(
            "graph_retriever_done",
            seeds=len(cuis),
            hops=hops,
            triples=len(result.triples),
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _expand_one(
        self, cui: str, hops: int, user_id: str | None = None
    ) -> list[GraphTriple]:
        """Run 1-hop and optionally 2-hop Cypher for a single seed CUI."""
        triples: list[GraphTriple] = []

        try:
            from medgraphia.graph.client import get_session

            async with get_session() as session:
                # 1. Get 1-hop Neighbors
                result = await session.run(
                    _CYPHER_1_HOP,
                    cui=cui,
                    user_id=user_id or "anonymous",
                    limit=self._per_seed_limit,
                )
                async for record in result:
                    triple = _record_to_triple(record)
                    if triple:
                        triples.append(triple)

                # Optionally run 2-hop
                if hops >= 2:
                    result2 = await session.run(
                        _CYPHER_2_HOP,
                        cui=cui,
                        user_id=user_id or "anonymous",
                        limit=self._per_seed_limit,
                    )
                    async for record in result2:
                        triple = _record_to_triple(record)
                        if triple:
                            triples.append(triple)

        except Exception as exc:
            logger.warning("graph_expand_failed", cui=cui, error=str(exc))

        return triples

    async def check_path_exists(self, cui_a: str, cui_b: str, max_hops: int = 3) -> bool:
        """
        Return True if a real (non-structural) path connects two CUIs within
        max_hops. Used as a cheap idempotency guard before firing a query-time
        completion fetch — no point re-searching PubMed for a pair that's
        already connected.
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
                # Inside an async context (e.g. FastAPI): caller must use await
                raise RuntimeError(
                    "retrieve_sync() called inside a running event loop. "
                    "Use `await retrieve()` instead."
                )
            return loop.run_until_complete(self.retrieve(cuis, hops=hops))
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("graph_retrieve_sync_failed", error=str(exc))
            return GraphRetrievalResult(seed_cuis=list(cuis), hops=hops)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_to_triple(record: Any) -> GraphTriple | None:
    """Convert a Neo4j record dict to a GraphTriple.  Returns None on missing fields."""
    try:
        return GraphTriple(
            entity_cui=record["entity_cui"] or "",
            entity_label=record["entity_label"] or "",
            entity_type=record["entity_type"] or "Unknown",
            relation_type=record["relation_type"] or "",
            neighbor_cui=record["neighbor_cui"] or "",
            neighbor_label=record["neighbor_label"] or "",
            neighbor_type=record["neighbor_type"] or "Unknown",
            evidence_text=record.get("evidence_text") or "",
            confidence=float(record.get("confidence") or 1.0),
            chunk_id=record.get("chunk_id") or "",
        )
    except (KeyError, TypeError) as exc:
        logger.debug("triple_parse_failed", error=str(exc))
        return None
