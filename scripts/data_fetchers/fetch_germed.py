#!/usr/bin/env python3
"""
CLI tool: download the GERNERMED medical dataset from GitHub.

The script fetches the raw JSON dataset from the frankkramer-lab repository.

over 8000 QA-data
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import httpx

# Add src to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.logger import configure_logging, get_logger

GERNERMED_URL = (
    "https://raw.githubusercontent.com/frankkramer-lab/GERNERMED/main/data/GERNERMED_dataset.json"
)


@click.command()
@click.option("--out", default="data/raw/germed", show_default=True, help="Output directory")
def main(out: str) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    logger = get_logger("fetch_germed")

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    target_file = out_path / "GERNERMED_dataset.json"

    if target_file.exists():
        click.echo(f"Already exists: {target_file}, skipping download.")
        logger.info("germed_download_skipped", path=str(target_file))
        return

    click.echo(f"Downloading GERNERMED dataset from {GERNERMED_URL}...")

    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(GERNERMED_URL)
            response.raise_for_status()

            with open(target_file, "wb") as f:
                f.write(response.content)

            click.echo(f"Successfully downloaded to {target_file}")
            logger.info("germed_download_complete", path=str(target_file))

    except httpx.HTTPStatusError as e:
        click.echo(f"Error: Failed to download (Status {e.response.status_code})")
        logger.error("germed_download_failed", status_code=e.response.status_code)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        logger.error("germed_download_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
