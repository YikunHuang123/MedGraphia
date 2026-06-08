#!/usr/bin/env python3
"""
CLI tool: download FDA drug label XML from DailyMed.

Usage:
  python scripts/fetch_fda_dailymed.py --drugs "metformin,warfarin,aspirin" --limit 5
  python scripts/fetch_fda_dailymed.py --drug-file data/drug_list.txt --out data/raw/fda_dailymed

Arguments:
  --drugs      Comma-separated drug names
  --drug-file  Text file with one drug name per line
  --limit      Max labels per drug to download (default: 3)
  --out        Output directory (default: data/raw/fda_dailymed)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.data.fda_dailymed import FDADailyMedConnector
from medgraphia.logger import configure_logging


@click.command()
@click.option("--drugs", default=None, help="Comma-separated drug names")
@click.option("--drug-file", default=None, type=click.Path(exists=True))
@click.option("--limit", default=3, show_default=True, help="Max labels per drug")
@click.option("--out", default="data/raw/fda_dailymed", show_default=True)
def main(drugs: str | None, drug_file: str | None, limit: int, out: str) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)

    drug_names: list[str] = []
    if drugs:
        drug_names = [d.strip() for d in drugs.split(",") if d.strip()]
    elif drug_file:
        drug_names = [
            line.strip() for line in Path(drug_file).read_text().splitlines() if line.strip()
        ]

    if not drug_names:
        click.echo("No drug names provided. Use --drugs or --drug-file.", err=True)
        raise SystemExit(1)

    click.echo(f"Fetching FDA DailyMed labels for: {', '.join(drug_names)}")
    asyncio.run(_run(drug_names, limit, out))


async def _run(drug_names: list[str], limit: int, out: str) -> None:
    async with FDADailyMedConnector(output_dir=out) as connector:
        total = 0
        for drug_name in drug_names:
            docs = await connector.fetch_by_drug_name(drug_name, limit=limit)
            for doc in docs:
                click.echo(f"  ✓ {doc.title or drug_name}  [{doc.source.source_id}]")
                total += 1
    click.echo(f"Downloaded {total} label documents to {out}/")


if __name__ == "__main__":
    main()
