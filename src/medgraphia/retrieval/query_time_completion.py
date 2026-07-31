"""
Query-time knowledge graph completion.

When a user's question involves two entities the graph can't connect, this
fetches a few PubMed abstracts about the pair and ingests them (chunk, NER,
link, co-occurrence edges) so future PPR retrieval can find the connection —
no relation extraction, the two entities simply join the same chunk graph.
"""

from __future__ import annotations

from medgraphia.logger import get_logger

logger = get_logger(__name__)


async def complete_gap(entity_a: str, entity_b: str, pubmed_limit: int = 5) -> str:
    """
    Fetch evidence connecting two entities and ingest it into the graph.
    Returns a short text summary for the calling LLM — either what was
    found, or an explicit "no evidence" message so the model does not
    invent a connection.

    Args:
        entity_a: Label or surface-form text of the first entity.
        entity_b: Label or surface-form text of the second entity.
        pubmed_limit: Max abstracts to fetch — kept small on purpose, this
            runs inline during a chat request.
    """
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
    from medgraphia.ingestion.lightweight_extract import docs_to_chunks, write_chunks_to_graph

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

    chunks = await docs_to_chunks(docs)
    if not chunks:
        return f"Found {len(docs)} related article(s) but could not extract usable passages."

    await write_chunks_to_graph(docs, chunks)

    logger.info("gap_completion_ingested", docs=len(docs), chunks=len(chunks))
    return (
        f"Found {len(docs)} new article(s) discussing both {entity_a} and {entity_b}; "
        f"added {len(chunks)} passage(s) to the knowledge base for future retrieval."
    )
