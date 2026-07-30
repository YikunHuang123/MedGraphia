"""
Shared small-scale ingestion path: chunk -> NER -> link -> extract for a
handful of freshly-fetched documents. Used wherever a small batch of new
docs needs to become linked chunks and relations, outside a full-corpus
build (see ingestion/pipeline.py for the batched build-time versions).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

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
def get_relation_extractor():
    from medgraphia.ingestion.relation_extractor import RelationExtractor

    return RelationExtractor.from_settings()


async def docs_to_relations(docs: list[Any], extracted_by: str) -> tuple[list[Any], list[Any]]:
    """
    Run freshly-fetched docs through chunk -> NER -> link -> extract.

    Returns (linked_chunks, relations). Relations are tagged with
    `extracted_by` but not written to Neo4j — callers decide whether/how
    to persist the chunks, entities, and relations.
    """
    from medgraphia.ingestion.chunker import MedicalChunker

    chunker = MedicalChunker()
    chunks = [c for doc in docs for c in chunker.chunk(doc)]
    if not chunks:
        return [], []

    # NER and entity linking run local GPU/CPU inference synchronously — offload
    # to a thread so an async caller (chat request or build pipeline) isn't blocked.
    ner_pipeline = _get_ner_pipeline()
    chunks = await asyncio.to_thread(ner_pipeline.extract_batch, chunks)

    linker = _get_entity_linker()
    chunks = await asyncio.to_thread(linker.link_chunks_batch, chunks)

    extractor = get_relation_extractor()
    relations = await extractor.extract_batch(chunks)
    for r in relations:
        r.extracted_by = extracted_by

    return chunks, relations
