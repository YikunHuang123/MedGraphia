"""
Query-time knowledge graph completion.

When a user's question involves two entities the graph can't connect, this
fetches a few PubMed abstracts about the pair, extracts any relation found,
and merges it into Neo4j so this answer and future queries both benefit.
"""

from __future__ import annotations

from medgraphia.logger import get_logger

logger = get_logger(__name__)


async def complete_gap(entity_a: str, entity_b: str, pubmed_limit: int = 5) -> str:
    """
    Search for evidence connecting two entities and merge any relation found
    into the graph.  Returns a short text summary for the calling LLM —
    either the evidence found, or an explicit "no evidence" message so the
    model does not invent a connection.

    Args:
        entity_a: Label or surface-form text of the first entity.
        entity_b: Label or surface-form text of the second entity.
        pubmed_limit: Max abstracts to fetch — kept small on purpose, this
            runs inline during a chat request.
    """
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
    from medgraphia.ingestion.lightweight_extract import docs_to_relations, get_relation_extractor

    query = f'"{entity_a}" AND "{entity_b}"'
    logger.info("gap_completion_started", entity_a=entity_a, entity_b=entity_b)

    try:
        async with PubMedConnector() as pubmed:
            docs = await pubmed.fetch(PubMedFetchConfig(query=query, max_results=pubmed_limit))
    except Exception as exc:
        logger.warning("gap_completion_fetch_failed", error=str(exc))
        return f"No published evidence found connecting {entity_a} and {entity_b}."

    if not docs:
        logger.info("gap_completion_no_results", entity_a=entity_a, entity_b=entity_b)
        return f"No published evidence found connecting {entity_a} and {entity_b}."

    _, relations = await docs_to_relations(docs, extracted_by="query_time_synthesis")
    if not relations:
        return f"Found {len(docs)} related article(s) but no explicit relation between {entity_a} and {entity_b} could be confirmed."

    await get_relation_extractor().write_relations_to_neo4j(relations)

    summaries = [
        f"{r.source_cui} --[{r.relation_type.value}]--> {r.target_cui}"
        f" (\"{r.evidence_text.strip()}\")" if r.evidence_text else f"{r.source_cui} --[{r.relation_type.value}]--> {r.target_cui}"
        for r in relations
    ]
    logger.info("gap_completion_relations_found", count=len(relations))
    return "New evidence found and added to the knowledge graph: " + "; ".join(summaries)
