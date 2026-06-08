#!/usr/bin/env python3
"""
CLI tool: download MeSH data and import into Neo4j.
Replaces the old UMLS-based import logic with a fully automated, open-source flow.

Usage:
  python scripts/import_mesh.py --limit 10000
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.data.mesh import MeSHLoader
from medgraphia.domain import Entity, EntityType
from medgraphia.graph.client import close_driver, get_driver
from medgraphia.graph.queries import merge_entity
from medgraphia.graph.schema import apply_schema
from medgraphia.logger import configure_logging, get_logger

BATCH_SIZE = 500


@click.command()
@click.option("--limit", default=0, show_default=True, help="0 = all concepts")
@click.option("--dry-run", is_flag=True, help="Parse only, no Neo4j writes")
def main(limit: int, dry_run: bool) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)

    click.echo(f"Starting MeSH import (limit={limit or 'all'}, dry_run={dry_run})")
    asyncio.run(_run(limit or None, dry_run))


async def _run(limit: int | None, dry_run: bool) -> None:
    logger = get_logger("import_mesh")
    cfg = get_settings()

    loader = MeSHLoader(storage_dir=cfg.mesh_dir)

    # Step 1: Download data if missing
    await loader.ensure_data()

    # Step 2: Parse ASCII file
    index = loader.load(limit=limit)
    click.echo(f"Parsed {len(index)} concepts from MeSH.")

    if dry_run:
        click.echo("Dry run — skipping Neo4j writes.")
        return

    # Step 3: Write to Neo4j
    await get_driver()
    await apply_schema()

    concepts = list(loader.iter_concepts())
    total = len(concepts)
    written = 0

    for i in range(0, total, BATCH_SIZE):
        batch = concepts[i : i + BATCH_SIZE]
        for concept in batch:
            try:
                entity = Entity(
                    cui=concept["cui"],  # MeSH UI
                    label=concept.get("label") or concept["cui"],
                    entity_type=EntityType(concept.get("entity_type", "Unknown")),
                    lang_labels=concept.get("lang_labels", {}),
                )
                await merge_entity(entity)
                written += 1
            except Exception as exc:
                logger.warning("mesh_write_error", cui=concept.get("cui"), error=str(exc))

        progress = int((i + len(batch)) / total * 100)
        click.echo(f"  Progress: {progress}%  ({i + len(batch)}/{total})", nl=False)
        click.echo("\r", nl=False)

    await close_driver()
    click.echo(f"\nSuccessfully imported {written}/{total} MeSH concepts into Neo4j.")


if __name__ == "__main__":
    main()
