"""
Prefect-based offline knowledge-graph build pipeline (architecture doc §2).

Orchestrates all ingestion stages in sequence:
  fetch → parse → chunk → ner → link → extract → embed → community

Graceful degradation: if Prefect is not installed, @task and @flow decorators
are replaced with no-ops so the pipeline runs as plain async functions.

Usage (direct)::

    from medgraphia.ingestion.pipeline import build_graph_flow, BuildConfig
    await build_graph_flow(BuildConfig(domain="t2dm", pubmed_limit=200))

Usage (Prefect UI)::

    prefect deployment build medgraphia/ingestion/pipeline.py:build_graph_flow \
        --name medgraphia-build --apply
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional Prefect — graceful degradation
# ---------------------------------------------------------------------------

try:
    from prefect import flow, task  # type: ignore[import]
    from prefect.logging import get_run_logger  # type: ignore[import]
    _PREFECT_AVAILABLE = True
except ImportError:
    _PREFECT_AVAILABLE = False
    logger.warning(
        "prefect_not_installed",
        msg="pip install prefect for orchestration UI; running as plain async functions",
    )

    def task(fn: Any = None, **_kw: Any) -> Any:  # type: ignore[misc]
        """No-op @task decorator when Prefect is not installed."""
        if fn is None:
            return lambda f: f
        return fn

    def flow(fn: Any = None, **_kw: Any) -> Any:  # type: ignore[misc]
        """No-op @flow decorator when Prefect is not installed."""
        if fn is None:
            return lambda f: f
        return fn


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class BuildConfig:
    """
    Configuration for a single pipeline run.  Mirrors build_graph.py CLI args
    but is usable programmatically.
    """
    domain: str = "t2dm"
    pubmed_query: str | None = None
    pubmed_limit: int = 200
    drug_limit: int = 30
    include_ema_smpc: bool = False
    include_drugbank: bool = False
    drugbank_xml: str | None = None

    # Stage-skip flags
    skip_fetch: bool = False
    skip_parse: bool = False
    skip_chunk: bool = False
    skip_ner: bool = False
    skip_link: bool = False
    skip_extract: bool = False
    skip_embed: bool = False
    skip_community: bool = False

    # Extra metadata attached to Prefect run tags
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual stage tasks
# ---------------------------------------------------------------------------

@task(name="fetch", retries=2, retry_delay_seconds=30)
async def fetch_task(cfg: BuildConfig) -> list[Any]:
    """Download data from PubMed / FDA DailyMed / EMA SmPC / DrugBank."""
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
    from medgraphia.knowledge_base import DOMAIN_QUERIES, DOMAIN_DRUGS

    docs: list[Any] = []

    query = cfg.pubmed_query or DOMAIN_QUERIES.get(cfg.domain, cfg.domain)
    async with PubMedConnector() as pubmed:
        pubmed_docs = await pubmed.fetch(
            PubMedFetchConfig(query=query, max_results=cfg.pubmed_limit)
        )
    docs.extend(pubmed_docs)
    logger.info("fetch_pubmed_done", count=len(pubmed_docs))

    if cfg.drug_limit > 0:
        from medgraphia.data.fda_dailymed import FDADailyMedConnector

        drug_names = DOMAIN_DRUGS.get(cfg.domain, [])[:cfg.drug_limit]
        async with FDADailyMedConnector() as fda:
            for drug_name in drug_names:
                fda_docs = await fda.fetch_by_drug_name(drug_name, limit=2)
                docs.extend(fda_docs)
        logger.info("fetch_fda_done", extra=len(docs) - len(pubmed_docs))

    if cfg.include_drugbank and cfg.drugbank_xml:
        from medgraphia.data.drugbank import DrugBankConnector

        db = DrugBankConnector(xml_path=cfg.drugbank_xml)
        db_docs = db.fetch_all(limit=cfg.drug_limit)
        docs.extend(db_docs)
        logger.info("fetch_drugbank_done", count=len(db_docs))

    # 3. EMA SmPC (Local PDFs)
    if cfg.include_ema_smpc:
        from pathlib import Path
        from datetime import datetime
        from medgraphia.domain import SourceMeta, Language, RawDocument
        
        ema_dir = Path("data/raw/ema_smpc")
        if ema_dir.exists():
            ema_pdfs = list(ema_dir.glob("*.pdf"))
            for pdf_path in ema_pdfs:
                source = SourceMeta(
                    source_id=f"ema_local:{pdf_path.stem}",
                    source_title=pdf_path.stem.replace("_", " "),
                    retrieved_at=datetime.fromtimestamp(pdf_path.stat().st_mtime),
                    language=Language.EN
                )
                docs.append(RawDocument(source=source, file_path=str(pdf_path), format="pdf"))
            logger.info("fetch_ema_local_done", count=len(ema_pdfs))

    return docs


@task(name="parse")
def parse_task(raw_docs: list[Any]) -> list[Any]:
    """Parse PDFs / XML into structured documents."""
    from medgraphia.ingestion.parsers.docling_parser import DoclingParser

    parser = DoclingParser()
    parsed: list[Any] = []
    for doc in raw_docs:
        if doc.format in ("text", "xml"):
            parsed.append(doc)
        elif doc.format == "pdf" and doc.file_path:
            try:
                parsed.append(parser.parse(doc.file_path, source_meta=doc.source, language=doc.language))
            except Exception as exc:
                logger.warning("parse_pdf_failed", title=doc.title[:40], error=str(exc))
                parsed.append(doc)
    logger.info("parse_done", count=len(parsed))
    return parsed


@task(name="chunk")
async def chunk_task(docs: list[Any]) -> list[Any]:
    """Section-aware chunking + medical normalisation + Neo4j write."""
    from medgraphia.ingestion.chunker import MedicalChunker
    from medgraphia.ingestion.normalizer import MedicalNormalizer

    chunker = MedicalChunker()
    normalizer = MedicalNormalizer()
    all_chunks: list[Any] = []

    for doc in docs:
        chunks = chunker.chunk(doc)
        chunks = [normalizer.normalize_chunk(c) for c in chunks]
        all_chunks.extend(chunks)

        try:
            from medgraphia.graph.queries import create_chunk, upsert_document

            await upsert_document(doc)
            for chunk in chunks:
                await create_chunk(chunk)
        except Exception as exc:
            logger.warning("chunk_neo4j_failed", doc_id=doc.doc_id[:8], error=str(exc))

    logger.info("chunk_done", count=len(all_chunks))
    return all_chunks


@task(name="ner")
async def ner_task(chunks: list[Any]) -> list[Any]:
    """Multi-language NER: GLiNER + optional BERT precision pass."""
    from medgraphia.ingestion.ner import build_pipeline_from_settings

    pipeline = build_pipeline_from_settings()
    result: list[Any] = []
    for chunk in chunks:
        try:
            result.append(pipeline.extract(chunk))
        except Exception as exc:
            logger.warning("ner_chunk_failed", chunk_id=chunk.chunk_id[:8], error=str(exc))
            result.append(chunk)

    n_entities = sum(len(c.entities) for c in result)
    logger.info("ner_done", chunks=len(result), entities=n_entities)
    return result


@task(name="link")
async def link_task(chunks: list[Any]) -> list[Any]:
    """Entity linking: BM25 candidate retrieval + SapBERT re-ranking."""
    from medgraphia.config import get_settings
    from medgraphia.ingestion.entity_linker import EntityLinker

    cfg = get_settings()
    linker = EntityLinker.from_mesh(
        mesh_dir=cfg.mesh_dir,
        bm25_top_k=cfg.el_bm25_top_k,
        link_threshold=cfg.el_link_threshold,
        sapbert_model=cfg.el_sapbert_model,
        sapbert_threshold=cfg.el_sapbert_threshold,
    )
    linker.build_index()

    result: list[Any] = []
    for chunk in chunks:
        try:
            linked = linker.link_chunk(chunk)
            await linker.write_entities_to_neo4j(linked)
            result.append(linked)
        except Exception as exc:
            logger.warning("link_chunk_failed", chunk_id=chunk.chunk_id[:8], error=str(exc))
            result.append(chunk)

    linked_count = sum(
        1 for c in result for e in c.entities if not e.cui.startswith("MENTION:")
    )
    logger.info("link_done", linked=linked_count)
    return result


@task(name="extract")
async def extract_task(chunks: list[Any]) -> list[Any]:
    """LLM-based relation extraction (schema-guided)."""
    from medgraphia.ingestion.relation_extractor import RelationExtractor

    extractor = RelationExtractor.from_settings()
    relations = await extractor.extract_batch(chunks)
    await extractor.write_relations_to_neo4j(relations)
    logger.info("extract_done", relations=len(relations))
    return relations


@task(name="embed")
async def embed_task(chunks: list[Any]) -> list[Any]:
    """BGE-M3 chunk embedding (dense + sparse) → Qdrant collection."""
    from medgraphia.config import get_settings
    from medgraphia.ingestion.embedder import MedicalEmbedder
    from medgraphia.vector.qdrant_store import QdrantStore

    cfg = get_settings()
    embedder = MedicalEmbedder.from_settings()
    store = QdrantStore()

    try:
        await store.init_collection(
            collection_name=cfg.qdrant_collection_chunks,
            vector_size=embedder.dense_dim,
            sparse=True,
        )
        embedded = embedder.embed_chunks(chunks)
        written = await store.upsert_chunks(cfg.qdrant_collection_chunks, embedded)
        logger.info("embed_done", chunks=written, collection=cfg.qdrant_collection_chunks)
        return embedded
    except Exception as exc:
        logger.error("embed_task_failed", error=str(exc))
        return chunks


@task(name="community")
async def community_task(
    relations: list[Any],
    chunks: list[Any],
) -> list[Any]:
    """Leiden community detection + LLM community summaries."""
    from medgraphia.ingestion.community_builder import CommunityBuilder

    # Build entity_map for richer prompts
    entity_map = {
        e.cui: e
        for chunk in chunks
        for e in chunk.entities
        if not e.cui.startswith("MENTION:")
    }

    builder = CommunityBuilder.from_settings()
    communities = await builder.build_from_relations(relations, entity_map)
    await builder.write_communities_to_neo4j(communities)
    logger.info("community_done", count=len(communities))
    return communities


# ---------------------------------------------------------------------------
# Main orchestration flow
# ---------------------------------------------------------------------------

@flow(name="MedGraphia Build Pipeline", log_prints=True)
async def build_graph_flow(cfg: BuildConfig) -> dict[str, Any]:
    """
    Full offline knowledge-graph build pipeline.

    Returns a summary dict with counts for each stage.
    """
    summary: dict[str, Any] = {"domain": cfg.domain}

    # Stage 1: Fetch
    raw_docs: list[Any] = []
    if not cfg.skip_fetch:
        raw_docs = await fetch_task(cfg)
    summary["raw_docs"] = len(raw_docs)

    # Stage 2: Parse
    parsed_docs: list[Any] = raw_docs
    if not cfg.skip_parse and raw_docs:
        parsed_docs = parse_task(raw_docs)
    summary["parsed_docs"] = len(parsed_docs)

    # Stage 3: Chunk
    chunks: list[Any] = []
    if not cfg.skip_chunk and parsed_docs:
        chunks = await chunk_task(parsed_docs)
    summary["chunks"] = len(chunks)

    # RECOVERY: If chunks is empty but needed for downstream, load from DB
    if not chunks and not (cfg.skip_ner and cfg.skip_link and cfg.skip_extract and cfg.skip_embed and cfg.skip_community):
        from medgraphia.graph.queries import get_chunks_from_db
        chunks = await get_chunks_from_db(limit=5000)
        logger.info("pipeline_recovered_from_db", count=len(chunks))

    # Stage 4: NER
    if not cfg.skip_ner and chunks:
        chunks = await ner_task(chunks)
    summary["entities"] = sum(len(c.entities) for c in chunks)

    # Stage 5: Entity linking
    if not cfg.skip_link and chunks:
        chunks = await link_task(chunks)
    summary["linked"] = sum(
        1 for c in chunks for e in c.entities if not e.cui.startswith("MENTION:")
    )

    # Stage 6: Relation extraction
    relations: list[Any] = []
    if not cfg.skip_extract and chunks:
        relations = await extract_task(chunks)
    summary["relations"] = len(relations)

    # Stage 7: Embedding — BGE-M3 dense + sparse → Qdrant
    if not cfg.skip_embed:
        chunks = await embed_task(chunks)

    # Stage 8: Community detection
    communities: list[Any] = []
    if not cfg.skip_community and relations:
        communities = await community_task(relations, chunks)
    summary["communities"] = len(communities)

    logger.info("pipeline_complete", **summary)
    return summary


# ---------------------------------------------------------------------------
# CLI entrypoint (wraps build_graph_flow for direct invocation)
# ---------------------------------------------------------------------------

def run_pipeline(cfg: BuildConfig) -> dict[str, Any]:
    """Synchronous wrapper — useful for scripts and testing."""
    return asyncio.run(build_graph_flow(cfg))
