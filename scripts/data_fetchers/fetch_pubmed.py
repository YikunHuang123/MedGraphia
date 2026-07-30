#!/usr/bin/env python3
"""
CLI tool: fetch PubMed abstracts and save them as JSON to data/raw/pubmed/.

Usage:
  python scripts/fetch_pubmed.py --domain t2dm --limit 200
  python scripts/data_fetchers/fetch_pubmed.py --query "Humans[MeSH] AND Drug Therapy[MeSH]" --from 2024-01-01 --to 2026-12-31 --limit 200000 --out data/raw/pubmed/clinical_general
Arguments:
  --domain   Predefined domain keyword set (t2dm, cardiovascular, oncology, …)
  --query    Override with an arbitrary PubMed search query
  --limit    Maximum number of abstracts to fetch (default: 200)
  --out      Output directory (default: data/raw/pubmed/<domain>)
  --from     Start date, YYYY-MM-DD (optional)
  --to       End date, YYYY-MM-DD (optional)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import click

# Ensure src/ is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
from medgraphia.knowledge_base import DOMAIN_QUERIES
from medgraphia.logger import configure_logging, get_logger


@click.command()
@click.option("--domain", default=None, help="Predefined domain key (e.g. t2dm)")
@click.option("--query", default=None, help="Custom PubMed query (overrides --domain)")
@click.option("--limit", default=200, show_default=True, help="Max abstracts to fetch")
@click.option("--out", default=None, help="Output directory")
@click.option("--from", "date_from", default=None, help="Start date YYYY-MM-DD")
@click.option("--to", "date_to", default=None, help="End date YYYY-MM-DD")
def main(
    domain: str | None,
    query: str | None,
    limit: int,
    out: str | None,
    date_from: str | None,
    date_to: str | None,
) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    logger = get_logger("fetch_pubmed")

    # Resolve query
    if query:
        search_query = query
        domain_key = "custom"
    elif domain:
        domain_key = domain.lower()
        search_query = DOMAIN_QUERIES.get(domain_key)
        if not search_query:
            click.echo(
                f"Unknown domain '{domain}'. Available: {', '.join(DOMAIN_QUERIES)}",
                err=True,
            )
            sys.exit(1)
    else:
        domain_key = cfg.default_domain
        search_query = DOMAIN_QUERIES.get(domain_key, domain_key)

    # Resolve output directory
    output_dir = Path(out) if out else Path("data/raw/pubmed") / domain_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse dates
    df = datetime.strptime(date_from, "%Y-%m-%d").date() if date_from else None
    dt = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else None

    fetch_cfg = PubMedFetchConfig(
        query=search_query,
        max_results=limit,
        date_from=df,
        date_to=dt,
    )

    logger.info("pubmed_fetch_start", query=search_query, limit=fetch_cfg.max_results)
    click.echo(f"Fetching up to {fetch_cfg.max_results} abstracts for domain '{domain_key}'…")

    asyncio.run(_fetch_and_save(fetch_cfg, output_dir))
    click.echo(f"Done. Results saved to {output_dir}/")


async def _fetch_and_save(config: PubMedFetchConfig, output_dir: Path) -> None:
    async with PubMedConnector() as connector:
        docs = await connector.fetch(config)

    saved = 0
    skipped = 0
    for doc in docs:
        doc_id = doc.source.source_id.replace(":", "_")
        out_path = output_dir / f"{doc_id}.json"
        if out_path.exists():
            skipped += 1
            continue
        out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        saved += 1

    msg = f"Saved {saved} documents to {output_dir}/"
    if skipped:
        msg += f" ({skipped} already existed, skipped)"
    click.echo(msg)


if __name__ == "__main__":
    main()
