#!/usr/bin/env python3
"""
Master pipeline script: orchestrates the full offline knowledge-graph build.
Runs Phases 1–5 in sequence for a given domain.

Usage:
  # Lite mode — T2DM domain, 200 abstracts + 30 drug labels
  python scripts/build_graph.py --domain t2dm --pubmed-limit 200 --drug-limit 30

  # Enterprise mode — cardiovascular, with EMA SmPC and DrugBank
  python scripts/build_graph.py --domain cardiovascular --pubmed-limit 500 \
    --include-ema-smpc --drugbank-xml data/drugbank/full_database.xml

Stages executed (each can be skipped with --skip-<stage>):
  1. fetch      — Download data from PubMed / FDA DailyMed / EMA SmPC
  2. parse      — Parse PDFs / XML into RawDocument objects
  3. chunk      — Section-aware chunking with metadata
  4. ner        — Multi-language NER (Phase 3 — GLiNER + BioBERT)
  5. link       — Entity linking to UMLS CUI (Phase 3)
  6. extract    — LLM-based relation extraction (Phase 4)
  7. embed      — BGE-M3 embedding → Qdrant (Phase 5)
  8. community  — Leiden community detection + LLM summaries (Phase 4)

Phases 4–8 are stubs in this release (Phase 0/1). They log a "not yet implemented"
message and skip gracefully.  Fill them in during Phases 2–5 of development.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.logger import configure_logging, get_logger


@click.command()
@click.option("--domain", default=None, help="Domain key (t2dm, cardiovascular…)")
@click.option("--pubmed-query", default=None, help="Custom PubMed query")
@click.option("--pubmed-limit", default=200, show_default=True)
@click.option("--drug-limit", default=30, show_default=True)
@click.option("--include-ema-smpc", is_flag=True)
@click.option("--include-drugbank", is_flag=True)
@click.option("--drugbank-xml", default=None, type=click.Path())
@click.option("--skip-fetch", is_flag=True)
@click.option("--skip-parse", is_flag=True)
@click.option("--skip-chunk", is_flag=True)
@click.option("--skip-ner", is_flag=True)
@click.option("--skip-link", is_flag=True)
@click.option("--skip-extract", is_flag=True)
@click.option("--skip-embed", is_flag=True)
@click.option("--skip-community", is_flag=True)
def main(**kwargs: object) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    asyncio.run(_run(**kwargs))  # type: ignore[arg-type]


async def _run(
    domain: str | None,
    pubmed_query: str | None,
    pubmed_limit: int,
    drug_limit: int,
    include_ema_smpc: bool,
    include_drugbank: bool,
    drugbank_xml: str | None,
    skip_fetch: bool,
    skip_parse: bool,
    skip_chunk: bool,
    skip_ner: bool,
    skip_link: bool,
    skip_extract: bool,
    skip_embed: bool,
    skip_community: bool,
) -> None:
    cfg = get_settings()
    logger = get_logger("build_graph")
    domain_key = domain or cfg.default_domain

    click.echo(f"\n{'='*60}")
    click.echo(f"  MedGraphia — Build Graph Pipeline")
    click.echo(f"  Domain:  {domain_key}")
    click.echo(f"  Storage: {cfg.storage_backend}")
    click.echo(f"  Auth:    {cfg.auth_strategy}")
    click.echo(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Stage 1: Fetch
    # ------------------------------------------------------------------
    raw_docs = []
    if not skip_fetch:
        click.echo("[1/8] Fetching data sources…")
        raw_docs = await _stage_fetch(
            domain_key, pubmed_query, pubmed_limit, drug_limit,
            include_ema_smpc, include_drugbank, drugbank_xml,
        )
        click.echo(f"      → {len(raw_docs)} raw documents fetched.")
    else:
        click.echo("[1/8] Fetch skipped.")

    # ------------------------------------------------------------------
    # Stage 2: Parse
    # ------------------------------------------------------------------
    parsed_docs = []
    if not skip_parse:
        click.echo("[2/8] Parsing documents…")
        parsed_docs = _stage_parse(raw_docs)
        click.echo(f"      → {len(parsed_docs)} documents parsed.")
    else:
        click.echo("[2/8] Parse skipped.")

    # ------------------------------------------------------------------
    # Stage 3: Chunk + Normalise + Write Chunks to Neo4j  (Phase 2)
    # ------------------------------------------------------------------
    chunks = []
    if not skip_chunk:
        click.echo("[3/8] Chunking and normalising documents…")
        chunks = await _stage_chunk(parsed_docs)
        click.echo(f"      → {len(chunks)} chunks produced and written to Neo4j.")
    else:
        click.echo("[3/8] Chunk skipped.")

    # ------------------------------------------------------------------
    # Stage 4: NER  (Phase 3)
    # ------------------------------------------------------------------
    if not skip_ner:
        click.echo("[4/8] Running multi-language NER…")
        chunks = await _stage_ner(chunks)
        n_entities = sum(len(c.entities) for c in chunks)
        click.echo(f"      → {n_entities} entity mentions extracted across {len(chunks)} chunks.")
    else:
        click.echo("[4/8] NER skipped.")

    # ------------------------------------------------------------------
    # Stage 5: Entity linking  (Phase 3)
    # ------------------------------------------------------------------
    if not skip_link:
        click.echo("[5/8] Linking entities to UMLS CUIs…")
        chunks = await _stage_link(chunks)
        linked = sum(
            1 for c in chunks for e in c.entities
            if not e.cui.startswith("MENTION:")
        )
        unlinked = sum(
            1 for c in chunks for e in c.entities
            if e.cui.startswith("MENTION:")
        )
        click.echo(f"      → {linked} linked to UMLS CUI, {unlinked} kept as provisional mentions.")
    else:
        click.echo("[5/8] Entity linking skipped.")

    # ------------------------------------------------------------------
    # Stage 6: Relation extraction  (Phase 4)
    # ------------------------------------------------------------------
    relations = []
    if not skip_extract:
        click.echo("[6/8] Extracting relations (LLM schema-guided)…")
        relations = await _stage_extract(chunks)
        click.echo(f"      → {len(relations)} relations extracted.")
    else:
        click.echo("[6/8] Relation extraction skipped.")

    # ------------------------------------------------------------------
    # Stage 7: Embedding  (Phase 5 — stub)
    # ------------------------------------------------------------------
    if skip_embed:
        click.echo("[7/8] Embedding skipped.")
    else:
        click.echo("[7/8] Embedding — not yet implemented (Phase 5).")
        logger.info("stage_not_implemented", stage="Embedding")

    # ------------------------------------------------------------------
    # Stage 8: Community detection  (Phase 4)
    # ------------------------------------------------------------------
    if not skip_community:
        click.echo("[8/8] Running Leiden community detection + LLM summaries…")
        communities = await _stage_community(chunks, relations)
        click.echo(f"      → {len(communities)} communities detected.")
    else:
        click.echo("[8/8] Community detection skipped.")

    click.echo("\n✓ Pipeline complete.\n")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

_DOMAIN_QUERIES: dict[str, str] = {
    "t2dm": (
        "type 2 diabetes mellitus[MeSH] AND "
        "(drug therapy[MeSH] OR treatment[MeSH]) AND English[Language]"
    ),
    "cardiovascular": (
        "cardiovascular diseases[MeSH] AND drug therapy[MeSH] AND English[Language]"
    ),
    "oncology": (
        "neoplasms[MeSH] AND drug therapy[MeSH] AND English[Language]"
    ),
    "hypertension": (
        "hypertension[MeSH] AND antihypertensive agents[MeSH] AND English[Language]"
    ),
}


async def _stage_fetch(
    domain: str,
    pubmed_query: str | None,
    pubmed_limit: int,
    drug_limit: int,
    include_ema: bool,
    include_drugbank: bool,
    drugbank_xml: str | None,
) -> list:
    """Fetch all configured data sources and return combined RawDocument list."""
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig

    docs = []

    # PubMed
    query = pubmed_query or _DOMAIN_QUERIES.get(domain, domain)
    async with PubMedConnector() as pubmed:
        pubmed_docs = await pubmed.fetch(
            PubMedFetchConfig(query=query, max_results=pubmed_limit)
        )
    docs.extend(pubmed_docs)
    click.echo(f"      PubMed: {len(pubmed_docs)} abstracts")

    # FDA DailyMed
    if drug_limit > 0:
        # Fetch generic drug labels related to the domain
        from medgraphia.data.fda_dailymed import FDADailyMedConnector
        _DOMAIN_DRUGS: dict[str, list[str]] = {
            "t2dm": ["metformin", "insulin", "sitagliptin", "empagliflozin", "liraglutide"],
            "cardiovascular": ["warfarin", "aspirin", "atorvastatin", "lisinopril", "metoprolol"],
            "hypertension": ["amlodipine", "lisinopril", "losartan", "hydrochlorothiazide"],
        }
        drug_names = _DOMAIN_DRUGS.get(domain, [])[:drug_limit]
        async with FDADailyMedConnector() as fda:
            for drug_name in drug_names:
                fda_docs = await fda.fetch_by_drug_name(drug_name, limit=2)
                docs.extend(fda_docs)
        click.echo(f"      FDA DailyMed: {len(docs) - len(pubmed_docs)} labels")

    # EMA SmPC (Load locally downloaded PDFs)
    if include_ema:
        ema_dir = Path("data/raw/ema_smpc")
        if ema_dir.exists():
            ema_pdfs = list(ema_dir.glob("*.pdf"))
            for pdf_path in ema_pdfs:
                # Wrap local PDF in a RawDocument so the parser can see it
                from medgraphia.domain import SourceMeta, Language, RawDocument
                from datetime import datetime
                
                source = SourceMeta(
                    source_id=f"ema_local:{pdf_path.stem}",
                    source_title=pdf_path.stem.replace("_", " "),
                    retrieved_at=datetime.fromtimestamp(pdf_path.stat().st_mtime),
                    language=Language.EN
                )
                docs.append(RawDocument(source=source, file_path=str(pdf_path), format="pdf"))
            
            click.echo(f"      EMA SmPC: loaded {len(ema_pdfs)} local PDFs")
        else:
            click.echo("      EMA SmPC: No local PDFs found in data/raw/ema_smpc/")

    # DrugBank XML (local file, no network call)
    if include_drugbank and drugbank_xml:
        from medgraphia.data.drugbank import DrugBankConnector
        db = DrugBankConnector(xml_path=drugbank_xml)
        db_docs = db.fetch_all(limit=drug_limit)
        docs.extend(db_docs)
        click.echo(f"      DrugBank: {len(db_docs)} entries")

    return docs


def _stage_parse(raw_docs: list) -> list:
    """
    Parse documents.  PDF documents use Docling to extract structure (sections, tables).
    """
    click.echo(f"[{2}/8] Parsing documents…")
    from medgraphia.ingestion.parsers.docling_parser import DoclingParser
    
    parser = DoclingParser()
    parsed = []
    
    for doc in raw_docs:
        if doc.format in ("text", "xml"):
            parsed.append(doc)
        elif doc.format == "pdf" and doc.file_path:
            try:
                # Actual parsing of the PDF file downloaded in Stage 1
                parsed_doc = parser.parse(doc.file_path, source_meta=doc.source, language=doc.language)
                parsed.append(parsed_doc)
                click.echo(f"      ✓ Parsed PDF: {doc.title[:40]}...")
            except Exception as exc:
                click.echo(f"      ✗ Failed to parse PDF {doc.title}: {exc}")
                # Fallback to unparsed doc to keep it in the pipeline
                parsed.append(doc)
                
    click.echo(f"      → {len(parsed)} documents parsed.")
    return parsed


async def _stage_chunk(docs: list) -> list:
    """
    Phase 2: Section-aware chunking + medical normalisation + Neo4j write.

    For each document:
      1. Ensure the Document node exists in Neo4j (upsert).
      2. Chunk with MedicalChunker (section_path provenance preserved).
      3. Normalise each chunk's text (frequency / dosage unit normalisation).
      4. Write each Chunk node to Neo4j and link it to its parent Document.

    Neo4j connection is attempted but failures are logged and skipped so the
    script can still run when Neo4j is not available (e.g., in unit tests).
    """
    from medgraphia.ingestion.chunker import MedicalChunker
    from medgraphia.ingestion.normalizer import MedicalNormalizer

    chunker    = MedicalChunker()
    normalizer = MedicalNormalizer()
    all_chunks = []

    for doc in docs:
        chunks = chunker.chunk(doc)
        chunks = [normalizer.normalize_chunk(c) for c in chunks]
        all_chunks.extend(chunks)

        # Write to Neo4j — gracefully skip if Neo4j is unavailable
        try:
            from medgraphia.graph.queries import create_chunk, upsert_document
            await upsert_document(doc)
            for chunk in chunks:
                await create_chunk(chunk)
        except Exception as exc:
            click.echo(
                f"      ⚠ Neo4j write failed for {doc.doc_id[:8]}… "
                f"({type(exc).__name__}: {exc})"
            )

    return all_chunks


async def _stage_ner(chunks: list) -> list:
    """
    Phase 3 — Stage 4: Multi-language NER.

    Runs the two-stage NER pipeline (GLiNER + optional BERT) on every chunk and
    populates chunk.entities with provisional MENTION: CUIs.
    Falls back gracefully if neither model is available.
    """
    from medgraphia.ingestion.ner import build_pipeline_from_settings
    pipeline = build_pipeline_from_settings()

    result = []
    for chunk in chunks:
        try:
            result.append(pipeline.extract(chunk))
        except Exception as exc:
            click.echo(
                f"      ⚠ NER failed for chunk {chunk.chunk_id[:8]}… "
                f"({type(exc).__name__}: {exc})"
            )
            result.append(chunk)

    return result


async def _stage_link(chunks: list) -> list:
    """
    Phase 3 — Stage 5: Entity linking.

    Resolves MENTION: CUI placeholders to real MeSH IDs using BM25 + SapBERT.
    Writes linked entities and MENTIONED_IN edges to Neo4j (skips on unavailability).
    """
    from medgraphia.ingestion.entity_linker import EntityLinker
    cfg = get_settings()

    # Build linker — tries to load MeSH; downloads if missing.
    linker = EntityLinker.from_mesh(
        mesh_dir=cfg.mesh_dir,
        bm25_top_k=cfg.el_bm25_top_k,
        link_threshold=cfg.el_link_threshold,
        sapbert_model=cfg.el_sapbert_model,
        sapbert_threshold=cfg.el_sapbert_threshold,
    )
    linker.build_index()

    result = []
    for chunk in chunks:
        try:
            linked_chunk = linker.link_chunk(chunk)
            await linker.write_entities_to_neo4j(linked_chunk)
            result.append(linked_chunk)
        except Exception as exc:
            click.echo(
                f"      ⚠ EL failed for chunk {chunk.chunk_id[:8]}… "
                f"({type(exc).__name__}: {exc})"
            )
            result.append(chunk)

    return result


async def _stage_extract(chunks: list) -> list:
    """
    Phase 4 — Stage 6: LLM-based relation extraction.

    For each chunk with ≥ 2 linked entities, calls the configured LLM to identify
    typed semantic relations between entity pairs.  Falls back gracefully when the
    LLM is unavailable (returns []).
    """
    from medgraphia.ingestion.relation_extractor import RelationExtractor

    extractor = RelationExtractor.from_settings()
    relations = await extractor.extract_batch(chunks)

    try:
        await extractor.write_relations_to_neo4j(relations)
    except Exception as exc:
        click.echo(
            f"      ⚠ Neo4j relation write failed ({type(exc).__name__}: {exc})"
        )

    return relations


async def _stage_community(chunks: list, relations: list) -> list:
    """
    Phase 4 — Stage 8: Leiden community detection + LLM community summaries.

    Builds a graph from the extracted relations, runs Leiden (or a networkx
    fallback), calls the LLM to generate a clinical summary per community, and
    writes Community nodes + MEMBER_OF edges to Neo4j.
    """
    from medgraphia.ingestion.community_builder import CommunityBuilder

    entity_map = {
        e.cui: e
        for chunk in chunks
        for e in chunk.entities
        if not e.cui.startswith("MENTION:")
    }

    builder = CommunityBuilder.from_settings()
    communities = await builder.build_from_relations(relations, entity_map)

    try:
        await builder.write_communities_to_neo4j(communities)
    except Exception as exc:
        click.echo(
            f"      ⚠ Neo4j community write failed ({type(exc).__name__}: {exc})"
        )

    return communities


if __name__ == "__main__":
    main()
ain()
