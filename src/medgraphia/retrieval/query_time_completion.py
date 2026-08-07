"""
Query-time knowledge graph completion.

When a user's question involves two entities the graph can't connect, this
fetches a few PubMed abstracts about the pair and ingests them (chunk, NER,
link, co-occurrence edges) so future PPR retrieval can find the connection —
no relation extraction, the two entities simply join the same chunk graph.
"""

from __future__ import annotations

import time
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)


async def complete_single_entity_gap(entity: str, pubmed_limit: int = 5) -> tuple[str, list[Any]]:
    """
    Fetch evidence about a single entity and ingest it — the single-entity
    counterpart to complete_gap(). Triggered only when the reranker's own
    no_evidence signal says local retrieval came up genuinely empty (not
    merely "results are mediocre"), so this fires far less often than a
    naive "any uncovered entity" gate would, and doesn't paper over
    corpus-quality bugs the same way a low-relevance-score trigger would.
    """
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
    from medgraphia.ingestion.lightweight_extract import docs_to_chunks, write_chunks_to_graph

    query = f'"{entity}"'
    logger.info("single_entity_gap_completion_started", entity=entity)

    try:
        async with PubMedConnector() as pubmed:
            docs = await pubmed.fetch(PubMedFetchConfig(query=query, max_results=pubmed_limit))
    except Exception as exc:
        logger.warning("single_entity_gap_completion_fetch_failed", error=str(exc))
        return f"No published evidence found for {entity}.", []

    if not docs:
        logger.info("single_entity_gap_completion_no_results", entity=entity)
        return f"No published evidence found for {entity}.", []

    chunks = await docs_to_chunks(docs)
    if not chunks:
        return f"Found {len(docs)} related article(s) but could not extract usable passages.", []

    await write_chunks_to_graph(docs, chunks)
    logger.info("single_entity_gap_completion_ingested", docs=len(docs), chunks=len(chunks))
    return (
        f"Found {len(docs)} new article(s) about {entity}; "
        f"added {len(chunks)} passage(s) to the knowledge base for future retrieval."
    ), chunks


async def complete_gap(entity_a: str, entity_b: str, pubmed_limit: int = 5) -> tuple[str, list[Any]]:
    """
    Fetch evidence connecting two entities and ingest it into the graph.
    Returns a tuple of (summary_message, list_of_new_chunks).
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
        return f"No published evidence found connecting {entity_a} and {entity_b}.", []

    if not docs:
        logger.info("gap_completion_no_results", entity_a=entity_a, entity_b=entity_b)
        return f"No published evidence found connecting {entity_a} and {entity_b}.", []

    t0 = time.monotonic()
    chunks = await docs_to_chunks(docs)
    t1 = time.monotonic()
    logger.info("gap_completion_ner_linking_done", docs=len(docs), chunks=len(chunks), seconds=round(t1 - t0, 2))
    if not chunks:
        return f"Found {len(docs)} related article(s) but could not extract usable passages.", []

    await write_chunks_to_graph(docs, chunks)
    t2 = time.monotonic()
    logger.info("gap_completion_graph_write_done", seconds=round(t2 - t1, 2))

    logger.info("gap_completion_ingested", docs=len(docs), chunks=len(chunks))
    return (
        f"Found {len(docs)} new article(s) discussing both {entity_a} and {entity_b}; "
        f"added {len(chunks)} passage(s) to the knowledge base for future retrieval."
    ), chunks
