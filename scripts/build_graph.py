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
    # Stages 3–8: Not yet implemented (Phase 2–5)
    # ------------------------------------------------------------------
    for stage_num, stage_name, skip_flag in [
        (3, "Chunking",             skip_chunk),
        (4, "NER",                  skip_ner),
        (5, "Entity linking",       skip_link),
        (6, "Relation extraction",  skip_extract),
        (7, "Embedding",            skip_embed),
        (8, "Community detection",  skip_community),
    ]:
        if skip_flag:
            click.echo(f"[{stage_num}/8] {stage_name} skipped.")
        else:
            click.echo(f"[{stage_num}/8] {stage_name} — not yet implemented (Phase 2+).")
            logger.info("stage_not_implemented", stage=stage_name)

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

    # EMA SmPC (stub — expensive, skip in lite mode by default)
    if include_ema:
        click.echo("      EMA SmPC: run scripts/fetch_ema_smpc.py separately for large downloads.")

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
    For text-format documents (PubMed abstracts, DailyMed XML) no additional
    parsing is needed — they are already RawDocument objects with full_text set.
    PDF documents require Docling / MinerU (handled in Phase 2).
    """
    parsed = []
    for doc in raw_docs:
        if doc.format in ("text", "xml"):
            parsed.append(doc)
        elif doc.format == "pdf" and doc.file_path:
            # PDF parsing will be wired up in Phase 2
            parsed.append(doc)
    return parsed


if __name__ == "__main__":
    main()
