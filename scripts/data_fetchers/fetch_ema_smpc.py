#!/usr/bin/env python3
"""
CLI tool: download EMA SmPC PDFs for a given list of drug names.

Usage:
  python scripts/fetch_ema_smpc.py --drugs "metformin,insulin,warfarin" --out data/raw/ema_smpc
  python scripts/fetch_ema_smpc.py --drug-file data/drug_list.txt --limit 20

The script fetches the EMA EPAR product index, then downloads the SmPC PDF
for each matching drug name.

Arguments:
  --drugs      Comma-separated list of drug names
  --drug-file  Path to a text file with one drug name per line
  --limit      Maximum number of products to download (default: 50)
  --out        Output directory (default: data/raw/ema_smpc)
  --lang       Language filter: en (default) or de
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.data.ema_smpc import EMASmPCConnector
from medgraphia.domain import Language
from medgraphia.logger import configure_logging, get_logger


@click.command()
@click.option("--drugs", default=None, help="Comma-separated drug names")
@click.option(
    "--drug-file", default=None, type=click.Path(exists=True), help="File with one drug per line"
)
@click.option("--limit", default=50, show_default=True, help="Max products to download")
@click.option("--out", default="data/raw/ema_smpc", show_default=True, help="Output directory")
@click.option("--lang", default="en", type=click.Choice(["en", "de"]), help="Target language")
def main(
    drugs: str | None,
    drug_file: str | None,
    limit: int,
    out: str,
    lang: str,
) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    logger = get_logger("fetch_ema_smpc")

    # Resolve drug list
    drug_names: list[str] = []
    if drugs:
        drug_names = [d.strip() for d in drugs.split(",") if d.strip()]
    elif drug_file:
        drug_names = [
            line.strip() for line in Path(drug_file).read_text().splitlines() if line.strip()
        ]

    language = Language.DE if lang == "de" else Language.EN

    click.echo(
        f"Downloading EMA SmPC for {len(drug_names) if drug_names else 'all'} drugs (limit={limit})…"
    )
    asyncio.run(_run(drug_names, limit, out, language))


async def _run(drug_names: list[str], limit: int, out: str, language: Language) -> None:
    async with EMASmPCConnector(output_dir=out) as connector:
        products = await connector.fetch_product_list()

        if drug_names:
            # Filter to requested drug names (case-insensitive substring match)
            filtered = [
                p
                for p in products
                if any(
                    name.lower() in (p.get("Medicine name", "") or "").lower()
                    for name in drug_names
                )
            ]
        else:
            filtered = products

        filtered = filtered[:limit]
        click.echo(f"Found {len(filtered)} matching products in EMA index.")

        downloaded = 0
        skipped = 0
        for product in filtered:
            name = product.get("Medicine name", "")
            url = product.get("URL", "") or product.get("Product page", "")
            if not url:
                continue
            # Check if already downloaded before making a network request
            file_name = re.sub(r"[^\w\-]", "_", name) + ".pdf"
            if (Path(out) / file_name).exists():
                skipped += 1
                click.echo(f"  ~ {name} (already exists, skipped)")
                continue
            doc = await connector.download_smpc(url, name, language)
            if doc:
                downloaded += 1
                click.echo(f"  ✓ {name}")

    msg = f"Downloaded {downloaded} SmPC documents to {out}/"
    if skipped:
        msg += f" ({skipped} already existed, skipped)"
    click.echo(msg)


if __name__ == "__main__":
    main()
