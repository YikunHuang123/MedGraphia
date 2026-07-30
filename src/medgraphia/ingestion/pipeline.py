"""
Prefect-based offline knowledge-graph build pipeline.

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

    # Direction-scoped online fetch for a user-specified topic
    domain: str | None = None
    pubmed_query: str | None = None
    pubmed_limit: int = 200
    drug_limit: int = 30
    include_ema_smpc: bool = False
    include_drugbank: bool = False
    drugbank_xml: str | None = None

    # Light supplementary fetch for entities under-covered by this build's own data
    frontier_min_mentions: int = 2  # entities mentioned at most this often are "frontier"
    frontier_max_entities: int = 8  # cap per build, keeps cost bounded

    # Stage-skip flags
    skip_fetch: bool = False
    skip_load: bool = False
    skip_parse: bool = False
    skip_chunk: bool = False
    skip_ner: bool = False
    skip_link: bool = False
    skip_extract: bool = False
    skip_frontier_expand: bool = False
    skip_embed: bool = False
    skip_community: bool = False

    # Max chunks to load from DB when recovering (None = unlimited)
    recovery_limit: int | None = None

    # Extra metadata attached to Prefect run tags
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual stage tasks
# ---------------------------------------------------------------------------


@task(name="fetch", retries=2, retry_delay_seconds=30)
async def fetch_task(cfg: BuildConfig) -> list[Any]:
    """Direction-scoped online fetch: PubMed / FDA DailyMed / DrugBank / EMA SmPC."""
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
    from medgraphia.knowledge_base import DOMAIN_DRUGS, DOMAIN_QUERIES

    docs: list[Any] = []

    # DOMAIN_QUERIES.get() falls back to the raw domain text, so an arbitrary
    # user-typed direction (e.g. "breast cancer") works as a free-text query.
    query = cfg.pubmed_query or DOMAIN_QUERIES.get(cfg.domain or "", cfg.domain or "")
    async with PubMedConnector() as pubmed:
        pubmed_docs = await pubmed.fetch(
            PubMedFetchConfig(query=query, max_results=cfg.pubmed_limit)
        )
    docs.extend(pubmed_docs)
    logger.info("fetch_pubmed_done", count=len(pubmed_docs))

    if cfg.drug_limit > 0:
        from medgraphia.data.fda_dailymed import FDADailyMedConnector

        drug_names = DOMAIN_DRUGS.get(cfg.domain or "", [])[: cfg.drug_limit]
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

    if cfg.include_ema_smpc:
        from datetime import datetime
        from pathlib import Path

        from medgraphia.domain import Language, RawDocument, SourceMeta

        ema_dir = Path("data/raw/ema_smpc")
        if ema_dir.exists():
            ema_pdfs = list(ema_dir.glob("*.pdf"))
            for pdf_path in ema_pdfs:
                source = SourceMeta(
                    source_id=f"ema_local:{pdf_path.stem}",
                    source_title=pdf_path.stem.replace("_", " "),
                    retrieved_at=datetime.fromtimestamp(pdf_path.stat().st_mtime),
                    language=Language.EN,
                )
                docs.append(RawDocument(source=source, file_path=str(pdf_path), format="pdf"))
            logger.info("fetch_ema_local_done", count=len(ema_pdfs))

    return docs


@task(name="load", retries=0)
async def load_task(cfg: BuildConfig) -> list[Any]:
    """Load pre-downloaded data from local data/raw directory."""
    from pathlib import Path
    from datetime import datetime
    from medgraphia.domain import RawDocument, SourceMeta, Language

    docs: list[Any] = []
    base_dir = Path("data/raw")
    if not base_dir.exists():
        logger.warning("data_raw_missing", msg="Directory data/raw/ does not exist. No files loaded.")
        return docs

    json_count = 0
    from medgraphia.ingestion.parsers.structured_parser import StructuredParser
    parser = StructuredParser()

    # 1. Load all pre-downloaded JSON documents (PubMed, FDA, DrugBank, GerMed, etc.)
    for path in base_dir.rglob("*.json"):
        if "germed" in path.parts:
            docs.extend(list(parser.parse_germed(path)))
            json_count += 1
            continue
        
        try:
            doc = RawDocument.model_validate_json(path.read_text(encoding="utf-8"))
            docs.append(doc)
            json_count += 1
        except Exception as exc:
            logger.warning("failed_to_parse_json", path=str(path), error=str(exc))

    for path in base_dir.rglob("*.jsonl"):
        if "huatuo" in path.parts:
            docs.extend(list(parser.parse_huatuo(path)))
            json_count += 1

    logger.info("load_local_json_done", count=json_count)

    # 2. Load EMA SmPC (Local PDFs)
    ema_dir = base_dir / "ema_smpc"
    if ema_dir.exists():
        ema_pdfs = list(ema_dir.glob("*.pdf"))
        for pdf_path in ema_pdfs:
            source = SourceMeta(
                source_id=f"ema_local:{pdf_path.stem}",
                source_title=pdf_path.stem.replace("_", " "),
                retrieved_at=datetime.fromtimestamp(pdf_path.stat().st_mtime),
                language=Language.EN,
            )
            docs.append(RawDocument(source=source, file_path=str(pdf_path), format="pdf"))
        logger.info("load_ema_local_done", count=len(ema_pdfs))

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
                parsed.append(
                    parser.parse(doc.file_path, source_meta=doc.source, language=doc.language)
                )
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

    import asyncio
    from concurrent.futures import ProcessPoolExecutor

    chunker = MedicalChunker()
    normalizer = MedicalNormalizer()
    all_chunks: list[Any] = []

    # and use asyncio.gather for concurrent DB writes (8 cores).

    db_sem = asyncio.Semaphore(50)  # Safe concurrent limit for Neo4j

    async def process_doc_db(doc, chunks):
        try:
            from medgraphia.graph.queries import create_chunk, upsert_document
            async with db_sem:
                await upsert_document(doc)
            
            async def write_chunk(chunk):
                async with db_sem:
                    await create_chunk(chunk)
                    
            await asyncio.gather(*[write_chunk(c) for c in chunks])
        except Exception as exc:
            logger.warning("chunk_neo4j_failed", doc_id=doc.doc_id[:8], error=str(exc))

    def cpu_bound_chunking(batch_docs):
        batch_results = []
        for d in batch_docs:
            c_list = chunker.chunk(d)
            c_norm = [normalizer.normalize_chunk(c) for c in c_list]
            batch_results.append((d, c_norm))
        return batch_results

    from tqdm import tqdm

    batch_size = 1000
    total_batches = (len(docs) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(docs), batch_size), total=total_batches, desc="Chunking & Ingesting", unit="batch"):
        batch = docs[i:i + batch_size]
        
        # 1. CPU-bound chunking in thread pool (frees event loop, allows concurrent C-extensions)
        processed_batch = await asyncio.to_thread(cpu_bound_chunking, batch)
        
        # 2. IO-bound concurrent DB writes
        db_tasks = []
        for d, c_list in processed_batch:
            all_chunks.extend(c_list)
            db_tasks.append(process_doc_db(d, c_list))
            
        await asyncio.gather(*db_tasks)
        
    logger.info("chunk_done", count=len(all_chunks))
    return all_chunks


@task(name="ner")
async def ner_task(chunks: list[Any]) -> list[Any]:
    """Multi-language NER: GLiNER + optional BERT precision pass."""
    from medgraphia.ingestion.ner import build_pipeline_from_settings
    from tqdm import tqdm

    pipeline = build_pipeline_from_settings()
    batch_size = 1000
    result: list[Any] = []
    
    try:
        for i in tqdm(range(0, len(chunks), batch_size), desc="Extracting Entities (NER)", unit="batch"):
            batch = chunks[i:i+batch_size]
            batch_res = await asyncio.to_thread(pipeline.extract_batch, batch)
            result.extend(batch_res)
    except Exception as exc:
        logger.warning("ner_batch_failed", error=str(exc))
        
        def fallback() -> list[Any]:
            res = []
            for chunk in chunks:
                try:
                    res.append(pipeline.extract(chunk))
                except Exception as chunk_exc:
                    logger.warning("ner_chunk_failed", chunk_id=chunk.chunk_id[:8], error=str(chunk_exc))
                    res.append(chunk)
            return res
            
        result = await asyncio.to_thread(fallback)

    n_entities = sum(len(c.entities) for c in result)
    logger.info("ner_done", chunks=len(result), entities=n_entities)
    return result


@task(name="link")
async def link_task(chunks: list[Any]) -> list[Any]:
    """Entity linking: BM25 candidate retrieval + SapBERT re-ranking."""
    from medgraphia.config import get_settings
    from medgraphia.ingestion.entity_linker import EntityLinker
    from tqdm import tqdm

    cfg = get_settings()
    linker = EntityLinker.from_mesh(
        mesh_dir=cfg.mesh_dir,
        link_threshold=cfg.el_link_threshold,
        sapbert_model=cfg.el_sapbert_model,
        sapbert_threshold=cfg.el_sapbert_threshold,
    )
    linker.build_index()

    batch_size = 500
    result: list[Any] = []
    
    try:
        for i in tqdm(range(0, len(chunks), batch_size), desc="Linking Entities to MeSH", unit="batch"):
            batch = chunks[i:i+batch_size]
            batch_res = await asyncio.to_thread(linker.link_chunks_batch, batch)
            result.extend(batch_res)
    except Exception as exc:
        logger.warning("link_batch_failed", error=str(exc))
        
        def fallback() -> list[Any]:
            return [linker.link_chunk(c) for c in chunks]
            
        result = await asyncio.to_thread(fallback)

    from medgraphia.graph.queries import batch_upsert_entities_and_links
    
    write_batch_size = 500
    for j in tqdm(range(0, len(result), write_batch_size), desc="Writing Entities to Neo4j", unit="batch"):
        await batch_upsert_entities_and_links(result[j:j+write_batch_size])

    linked_count = sum(1 for c in result for e in c.entities if not e.cui.startswith("MENTION:"))
    logger.info("link_done", linked=linked_count)
    return result


@task(name="extract")
async def extract_task(chunks: list[Any]) -> list[Any]:
    """LLM-based relation extraction (schema-guided)."""
    from medgraphia.ingestion.relation_extractor import RelationExtractor

    extractor = RelationExtractor.from_settings()
    relations = await extractor.extract_batch(chunks)
    await extractor.write_relations_to_neo4j(relations)
    
    from medgraphia.graph.queries import mark_chunks_extracted
    await mark_chunks_extracted([c.chunk_id for c in chunks])
    
    logger.info("extract_done", relations=len(relations))
    return relations


@task(name="frontier_expand", retries=1)
async def frontier_expand_task(
    cfg: BuildConfig, chunks: list[Any], relations: list[Any]
) -> tuple[list[Any], list[Any]]:
    """Fetch a few extra PubMed abstracts for entities barely covered in the graph so far."""
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
    from medgraphia.graph.queries import (
        batch_upsert_entities_and_links,
        get_entity_mention_counts,
        mark_chunks_extracted,
    )
    from medgraphia.ingestion.lightweight_extract import docs_to_relations, get_relation_extractor

    if not relations:
        return [], []

    label_map: dict[str, str] = {}
    for c in chunks:
        for e in c.entities:
            if not e.cui.startswith("MENTION:"):
                label_map[e.cui] = e.label

    involved_cuis = list({r.source_cui for r in relations} | {r.target_cui for r in relations})
    global_mention_counts = await get_entity_mention_counts(involved_cuis)
    frontier_cuis = [
        cui for cui in involved_cuis if global_mention_counts.get(cui, 0) <= cfg.frontier_min_mentions
    ][: cfg.frontier_max_entities]

    if not frontier_cuis:
        logger.info("frontier_expand_none")
        return [], []

    all_docs: list[Any] = []
    async with PubMedConnector() as pubmed:
        for cui in frontier_cuis:
            label = label_map.get(cui, cui)
            docs = await pubmed.fetch(PubMedFetchConfig(query=label, max_results=5))
            all_docs.extend(docs)

    if not all_docs:
        logger.info("frontier_expand_no_docs", entities=len(frontier_cuis))
        return [], []

    new_chunks, new_relations = await docs_to_relations(all_docs, extracted_by="frontier_expansion")
    if new_chunks:
        await batch_upsert_entities_and_links(new_chunks)
        await mark_chunks_extracted([c.chunk_id for c in new_chunks])
    if new_relations:
        await get_relation_extractor().write_relations_to_neo4j(new_relations)

    logger.info(
        "frontier_expand_done",
        entities=len(frontier_cuis),
        new_chunks=len(new_chunks),
        new_relations=len(new_relations),
    )
    return new_chunks, new_relations


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
        embedded = await embedder.embed_chunks(chunks)
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
        e.cui: e for chunk in chunks for e in chunk.entities if not e.cui.startswith("MENTION:")
    }

    builder = CommunityBuilder.from_settings()
    communities = await builder.build_from_relations(relations, entity_map)
    await builder.write_communities_to_neo4j(communities)
    logger.info("community_done", count=len(communities))
    return communities


# ---------------------------------------------------------------------------
# Main orchestration flow
# ---------------------------------------------------------------------------


@flow(name="MedGraphia Build Pipeline")
async def build_graph_flow(cfg: Any = None) -> dict[str, Any]:
    """
    Main orchestration flow.
    """
    if cfg is None:
        from medgraphia.config import get_settings
        cfg = get_settings()

    logger.info("flow_started", mode="build_graph")

    # Ensure Neo4j constraints and indexes exist before writing anything
    from medgraphia.graph.schema import apply_schema
    await apply_schema()

    summary: dict[str, Any] = {"scope": cfg.domain or "global"}

    # Stage 0: Fetch — only runs when a domain/query is set; global builds skip it.
    raw_docs: list[Any] = []
    if not cfg.skip_fetch and (cfg.domain or cfg.pubmed_query):
        raw_docs = await fetch_task(cfg)
    summary["fetched_docs"] = len(raw_docs)

    # Stage 1: Load — pre-downloaded local data/raw content, merged with
    # whatever Stage 0 just fetched.
    if not cfg.skip_load:
        raw_docs = raw_docs + await load_task(cfg)
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
    if not chunks and not (
        cfg.skip_ner
        and cfg.skip_link
        and cfg.skip_extract
        and cfg.skip_embed
        and cfg.skip_community
    ):
        # 1. Try loading fully-NER'd chunks from cache (Redis or Disk)
        if not cfg.skip_ner:
            cached_chunks = _load_ner_cache()
            if cached_chunks:
                chunks = cached_chunks
                cfg.skip_ner = True  # We successfully loaded NER results, skip the NER stage

        # 2. If no cache found, fall back to DB recovery (which provides chunks without entities)
        if not chunks:
            from medgraphia.graph.queries import get_chunks_from_db
            chunks = await get_chunks_from_db(limit=cfg.recovery_limit)
            logger.info("pipeline_recovered_from_db", count=len(chunks), limit=cfg.recovery_limit)

    # Stage 4: NER
    if not cfg.skip_ner and chunks:
        chunks = await ner_task(chunks)
        _save_ner_cache(chunks)
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

    # Stage 6.5: expand frontier entities for direction-scoped builds, folding
    # any new chunks/relations into the main lists before embedding.
    if not cfg.skip_frontier_expand and cfg.domain and relations:
        frontier_chunks, frontier_relations = await frontier_expand_task(cfg, chunks, relations)
        chunks = chunks + frontier_chunks
        relations = relations + frontier_relations
        summary["frontier_chunks"] = len(frontier_chunks)
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

# ---------------------------------------------------------------------------
# Caching Utilities
# ---------------------------------------------------------------------------

def _save_ner_cache(chunks_to_save: list[Any]) -> None:
    import pickle
    from pathlib import Path
    
    try:
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / "ner_backup.pkl"
        
        # Write massive payloads directly to high-speed NVMe SSD
        with open(cache_path, "wb") as f:
            pickle.dump(chunks_to_save, f)
        logger.info("disk_saved_ner_chunks", count=len(chunks_to_save), path=str(cache_path))
    except Exception as e:
        logger.warning("disk_save_ner_chunks_failed", error=str(e))

def _load_ner_cache() -> list[Any] | None:
    import pickle
    from pathlib import Path
    
    cache_path = Path(".cache/ner_backup.pkl")
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                chunks = pickle.load(f)
            logger.info("disk_loaded_ner_chunks", count=len(chunks), path=str(cache_path))
            return chunks
        except Exception as e:
            logger.warning("disk_load_failed", error=str(e))
            
    return None
