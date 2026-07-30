"""
Tier 3 — query-time knowledge graph completion.

When a user's question involves two medical entities that the graph and the
retrieved context cannot connect, this module fetches a handful of PubMed
abstracts specifically about the pair, extracts any relation found, and
merges it into Neo4j so both this answer and future queries benefit.

Reuses the same ingestion building blocks as the offline pipeline (chunker,
NER, entity linker, relation extractor) — this is deliberately not a
separate code path, just a narrowly-scoped, on-demand run of the same stages.
"""

from __future__ import annotations

from functools import lru_cache

from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cached heavy components — building these fresh per call would reload
# GLiNER/BERT/SapBERT models and rebuild the whole MeSH dense index every time.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    from medgraphia.ingestion.ner import build_pipeline_from_settings

    return build_pipeline_from_settings()


@lru_cache(maxsize=1)
def _get_entity_linker():
    from medgraphia.ingestion.entity_linker import EntityLinker

    linker = EntityLinker.from_settings()
    linker.build_index()
    return linker


@lru_cache(maxsize=1)
def _get_relation_extractor():
    from medgraphia.ingestion.relation_extractor import RelationExtractor

    return RelationExtractor.from_settings()


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------


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

    relations = await _extract_relations_from_docs(docs)
    if not relations:
        return f"Found {len(docs)} related article(s) but no explicit relation between {entity_a} and {entity_b} could be confirmed."

    extractor = _get_relation_extractor()
    await extractor.write_relations_to_neo4j(relations)

    summaries = [
        f"{r.source_cui} --[{r.relation_type.value}]--> {r.target_cui}"
        f" (\"{r.evidence_text.strip()}\")" if r.evidence_text else f"{r.source_cui} --[{r.relation_type.value}]--> {r.target_cui}"
        for r in relations
    ]
    logger.info("gap_completion_relations_found", count=len(relations))
    return "New evidence found and added to the knowledge graph: " + "; ".join(summaries)


async def _extract_relations_from_docs(docs: list) -> list:
    """Run the fetched docs through chunk -> NER -> link -> extract, tagging
    any resulting relations as query-time-synthesized."""
    import asyncio

    from medgraphia.ingestion.chunker import MedicalChunker

    chunker = MedicalChunker()
    chunks = [c for doc in docs for c in chunker.chunk(doc)]
    if not chunks:
        return []

    # NER and entity linking run local GPU/CPU inference synchronously — offload
    # to a thread so a live chat request doesn't block the event loop.
    ner_pipeline = _get_ner_pipeline()
    chunks = await asyncio.to_thread(ner_pipeline.extract_batch, chunks)

    linker = _get_entity_linker()
    chunks = await asyncio.to_thread(linker.link_chunks_batch, chunks)

    extractor = _get_relation_extractor()
    relations = await extractor.extract_batch(chunks)
    for r in relations:
        r.extracted_by = "query_time_synthesis"
    return relations
