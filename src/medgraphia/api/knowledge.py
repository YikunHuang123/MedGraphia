"""
Knowledge-graph exploration endpoints.

GET  /graph/entity          — Expand a 1–3-hop subgraph from a named entity.
GET  /graph/entity/search   — Fuzzy-search entity names in Neo4j (for auto-complete).
GET  /graph/stats           — Node / relationship counts (public summary).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from medgraphia.api.auth import require_api_key
from medgraphia.api.schemas import EntityQueryResponse
from medgraphia.graph.client import get_session as neo4j_session
from medgraphia.graph.queries import get_graph_stats, get_subgraph
from medgraphia.logger import get_logger
from medgraphia.observability import get_langfuse_client

logger = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["knowledge"])


# ---------------------------------------------------------------------------
# GET /graph/entity — subgraph expansion from a named entity
# ---------------------------------------------------------------------------


@router.get(
    "/entity",
    response_model=EntityQueryResponse,
    summary="Expand a subgraph for a named entity",
)
async def get_entity_subgraph(
    request: Request,
    name: str = Query(..., min_length=1, description="Entity name, e.g. 'metformin'"),
    lang: str = Query("en", description="Preferred label language: en | zh | de"),
    hops: int = Query(1, ge=1, le=3, description="Graph expansion depth (1–3 hops)"),
    principal: dict = Depends(require_api_key),
) -> EntityQueryResponse:
    """
    Look up an entity by name label, then expand its neighbourhood up to
    *hops* hops in the knowledge graph.
    """
    request_id: str = request.state.request_id if hasattr(request.state, "request_id") else ""
    langfuse = get_langfuse_client()

    with langfuse.trace(
        "knowledge_subgraph",
        user_id=principal.get("id", "anonymous"),
        input=name,
        metadata={"hops": hops, "request_id": request_id},
    ) as trace:
        # ── Find the entity node by name ─────────────────────────────────────
        with trace.span("entity_lookup", input=name) as span:
            entity_node = await _find_entity_by_name(name)
            if entity_node is None:
                span.end(metadata={"found": False})
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No entity found with name '{name}'.",
                )
            span.end(metadata={"found": True, "cui": entity_node.get("cui")})

        cui: str = entity_node.get("cui", "")

        # ── Expand the subgraph ──────────────────────────────────────────────
        with trace.span("subgraph_expansion", input=cui) as span:
            try:
                subgraph = await get_subgraph(cui=cui, hops=hops)
                span.end(
                    metadata={"nodes": len(subgraph["nodes"]), "edges": len(subgraph["edges"])}
                )
            except Exception as exc:
                logger.error("subgraph_expansion_failed", cui=cui, error=str(exc))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Subgraph expansion failed.",
                ) from exc

        return EntityQueryResponse(
            entity=entity_node,
            subgraph=subgraph,
        )


# ---------------------------------------------------------------------------
# GET /graph/entity/search — name auto-complete
# ---------------------------------------------------------------------------


@router.get(
    "/entity/search",
    summary="Fuzzy-search entity names (auto-complete)",
    response_model=list[dict[str, Any]],
)
async def search_entities(
    request: Request,
    q: str = Query(..., min_length=2, description="Partial entity name"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    entity_type: str | None = Query(None, description="Filter: Disease|Drug|..."),
    principal: dict = Depends(require_api_key),
) -> list[dict[str, Any]]:
    """
    Return a list of entity names matching *q* (auto-complete).
    """
    request_id: str = request.state.request_id if hasattr(request.state, "request_id") else ""
    langfuse = get_langfuse_client()

    with langfuse.trace(
        "knowledge_search",
        user_id=principal.get("id", "anonymous"),
        input=q,
        metadata={"limit": limit, "type": entity_type, "request_id": request_id},
    ):
        results = await _fuzzy_search_entities(q=q, limit=limit, entity_type=entity_type)
        logger.info("knowledge_entity_search", q=q, results=len(results))
        return results


# ---------------------------------------------------------------------------
# GET /graph/stats — graph statistics
# ---------------------------------------------------------------------------


@router.get(
    "/stats",
    summary="Knowledge-graph node and relation counts",
    response_model=dict[str, int],
)
async def graph_stats(
    principal: dict = Depends(require_api_key),
) -> dict[str, int]:
    """Return high-level statistics about the knowledge graph."""
    try:
        stats = await get_graph_stats()
        return stats
    except Exception as exc:
        logger.error("graph_stats_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve graph statistics.",
        ) from exc


# ---------------------------------------------------------------------------
# Internal Cypher helpers
# ---------------------------------------------------------------------------


async def _find_entity_by_name(name: str) -> dict[str, Any] | None:
    """
    Lookup an entity by label across lang_zh/de.
    Optimised: Avoids toLower() on the property to enable index seek.
    """
    # Neo4j 5.x recommendation: Lowercase the parameter, not the property
    lower_name = name.lower()
    cypher = """
    MATCH (e)
    WHERE (e:Disease OR e:Drug OR e:Symptom OR e:Gene OR e:Procedure)
      AND (
        e.label_lower   = $q  // Optimized path: if we had a lower_label property
        OR e.label      = $name
        OR e.lang_zh    = $name
        OR e.lang_de    = $name
      )
    RETURN e.cui        AS cui,
           e.label      AS label,
           labels(e)[0] AS entity_type,
           coalesce(e.lang_zh, '') AS lang_zh,
           coalesce(e.lang_de, '') AS lang_de,
           coalesce(e.confidence, 1.0) AS confidence
    LIMIT 1
    """
    # Note: For now, we use exact matching to leverage the new indexes.
    # In a full-text scenario, we would use a full-text index.
    try:
        async with neo4j_session() as session:
            result = await session.run(cypher, name=name, q=lower_name)
            record = await result.single()
            if record is None:
                return None
            return {
                "cui": record["cui"],
                "label": record["label"],
                "entity_type": record["entity_type"],
                "lang_zh": record["lang_zh"],
                "lang_de": record["lang_de"],
                "confidence": record["confidence"],
            }
    except Exception as exc:
        logger.warning("entity_lookup_failed", name=name, error=str(exc))
        return None


async def _fuzzy_search_entities(
    q: str,
    limit: int,
    entity_type: str | None,
) -> list[dict[str, Any]]:
    """Substring search on entity labels."""
    if entity_type:
        _VALID_TYPES = {"Disease", "Drug", "Symptom", "Gene", "Procedure"}
        if entity_type not in _VALID_TYPES:
            return []
        type_filter = f"e:{entity_type} AND"
    else:
        type_filter = "(e:Disease OR e:Drug OR e:Symptom OR e:Gene OR e:Procedure) AND"

    # Optimization: Use case-insensitive regex (?i) or STARTS WITH
    # if we want to stay within standard indexes. CONTAINS is always a scan.
    cypher = f"""
    MATCH (e)
    WHERE {type_filter} (
        e.label   STARTS WITH $q
        OR e.lang_zh STARTS WITH $q
        OR e.lang_de STARTS WITH $q
    )
    RETURN e.cui        AS cui,
           e.label      AS label,
           labels(e)[0] AS entity_type
    ORDER BY size(e.label)
    LIMIT $limit
    """
    results: list[dict[str, Any]] = []
    try:
        async with neo4j_session() as session:
            result = await session.run(cypher, q=q, limit=limit)
            async for record in result:
                results.append(
                    {
                        "cui": record["cui"],
                        "label": record["label"],
                        "entity_type": record["entity_type"],
                    }
                )
    except Exception as exc:
        logger.warning("entity_search_failed", q=q, error=str(exc))
    return results
