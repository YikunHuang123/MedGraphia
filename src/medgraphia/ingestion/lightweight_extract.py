"""
Shared small-scale ingestion path: chunk -> NER -> link -> persist for a
handful of freshly-fetched documents. Used wherever a small batch of new
docs needs to join the graph outside a full-corpus build (see
ingestion/pipeline.py for the batched build-time versions). No relation
extraction — new entities join the graph via chunk co-occurrence, which the
PPR retriever picks up directly (see project notes "关系抽取阶段/4. ...").
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


def warm_up() -> None:
    """
    Force-populate the lru_caches above at server startup.

    This is a separate model instance from retrieval/query_ner.py's cache —
    they don't share GLiNER/SapBERT despite being the same weights, so without
    this, whichever request is the first to ever trigger query-time gap
    completion (agentic_completion.py) eats a ~15-20s cold load that every
    other query already paid for at startup.

    GLiNER/BertNER load their actual model weights lazily on the first
    extract_batch() call, not on construction — so a dummy call is required
    here, not just building the pipeline object.
    """
    from medgraphia.domain import Language, SourceMeta
    from medgraphia.domain.document import Chunk

    linker = _get_entity_linker()

    ner = _get_ner_pipeline()
    dummy_chunk = Chunk(
        doc_id="warmup",
        source=SourceMeta(source_id="warmup"),
        language=Language.EN,
        text="Metformin is used to treat type 2 diabetes.",
    )
    warmed = ner.extract_batch([dummy_chunk])
    linker.link_chunks_batch(warmed)


async def docs_to_chunks(docs: list[Any]) -> list[Any]:
    """Run freshly-fetched docs through chunk -> NER -> link, returning linked chunks."""
    from medgraphia.ingestion.chunker import MedicalChunker

    chunker = MedicalChunker()
    chunks = [c for doc in docs for c in chunker.chunk(doc)]
    if not chunks:
        return []

    # NER and entity linking run local GPU/CPU inference synchronously — offload
    # to a thread so an async caller (chat request or build pipeline) isn't blocked.
    ner_pipeline = _get_ner_pipeline()
    chunks = await asyncio.to_thread(ner_pipeline.extract_batch, chunks)

    linker = _get_entity_linker()
    chunks = await asyncio.to_thread(linker.link_chunks_batch, chunks)

    return chunks


async def write_chunks_to_graph(docs: list[Any], chunks: list[Any]) -> None:
    """Persist fresh Document/Chunk nodes plus their linked entities and
    MENTIONED_IN co-occurrence edges."""
    from medgraphia.graph.queries import (
        batch_upsert_entities_and_links,
        create_chunk,
        upsert_document,
    )

    for doc in docs:
        await upsert_document(doc)
    for chunk in chunks:
        await create_chunk(chunk)
    if chunks:
        await batch_upsert_entities_and_links(chunks)
