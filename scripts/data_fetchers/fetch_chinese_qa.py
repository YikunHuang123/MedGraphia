#!/usr/bin/env python3
"""
CLI tool: download the Huatuo-Lite (Chinese medical QA) dataset from Hugging Face. （177703 total

Usage:
  python scripts/data_fetchers/fetch_chinese_qa.py --limit 177,703
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Add src to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.logger import configure_logging, get_logger

DATASET_NAME = "FreedomIntelligence/Huatuo26M-Lite"


@click.command()
@click.option("--limit", default=5000, show_default=True, help="Number of items to export")
@click.option("--out", default="data/raw/huatuo", show_default=True, help="Output directory")
def main(limit: int, out: str) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    logger = get_logger("fetch_chinese_qa")

    try:
        from datasets import load_dataset
    except ImportError:
        click.echo("Error: The 'datasets' library is required to download from Hugging Face.")
        click.echo("Please install it via: pip install datasets")
        sys.exit(1)

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    target_file = out_path / "huatuo_lite.jsonl"

    # Count already-downloaded records
    existing_count = 0
    if target_file.exists():
        with open(target_file, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)

    if existing_count >= limit:
        click.echo(f"Already have {existing_count} items (>= {limit}), skipping download.")
        logger.info("huatuo_download_skipped", path=str(target_file), existing=existing_count)
        return

    if existing_count > 0:
        click.echo(f"Found {existing_count} existing items, downloading {limit - existing_count} more...")
    else:
        click.echo(f"Downloading {DATASET_NAME} from Hugging Face...")

    try:
        # Load the dataset (Huatuo26M-Lite usually has a 'train' split)
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

        need = limit - existing_count
        click.echo(f"Exporting items {existing_count + 1}–{limit} to {target_file}...")

        count = 0
        with open(target_file, "a", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                if i < existing_count:
                    continue  # skip already-saved records
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if count >= need:
                    break

        total = existing_count + count
        click.echo(f"Successfully exported {count} new items ({total} total).")
        logger.info("huatuo_download_complete", path=str(target_file), new=count, total=total)

    except Exception as e:
        click.echo(f"Error: {str(e)}")
        logger.error("huatuo_download_failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
