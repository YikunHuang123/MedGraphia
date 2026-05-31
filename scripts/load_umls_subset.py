#!/usr/bin/env python3
"""
CLI tool: load a MetamorphoSys UMLS subset into Neo4j.

Prerequisites:
  1. Obtain a UMLS license at https://uts.nlm.nih.gov/uts/
  2. Download and run MetamorphoSys to export:
       Vocabularies: SNOMED CT (EN), RxNorm, MeSH
       Languages:    ENG, CHI (Simplified), GER
  3. Place the generated RRF files under data/umls/META/

Usage:
  python scripts/load_umls_subset.py --umls-dir data/umls/META --limit 500000

Arguments:
  --umls-dir   Path to the MetamorphoSys META directory containing MRCONSO.RRF
  --limit      Max concepts to load (0 = all; default: 0)
  --dry-run    Parse only — do not write to Neo4j
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.data.umls import UMLSLoader
from medgraphia.domain import Entity, EntityType
from medgraphia.graph.client import close_driver, get_driver
from medgraphia.graph.queries import merge_entity
from medgraphia.graph.schema import apply_schema
from medgraphia.logger import configure_logging, get_logger

BATCH_SIZE = 500  # Neo4j writes per transaction batch


@click.command()
@click.option("--umls-dir", default="data/umls/META", show_default=True)
@click.option("--limit", default=0, show_default=True, help="0 = all concepts")
@click.option("--dry-run", is_flag=True, help="Parse only, no Neo4j writes")
def main(umls_dir: str, limit: int, dry_run: bool) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    logger = get_logger("load_umls_subset")

    meta_path = Path(umls_dir)
    if not (meta_path / "MRCONSO.RRF").exists():
        click.echo(
            f"MRCONSO.RRF not found in {umls_dir}. "
            "Run MetamorphoSys first to generate the RRF export.",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Loading UMLS subset from {umls_dir} (limit={limit or 'all'}, dry_run={dry_run})")
    asyncio.run(_run(umls_dir, limit or None, dry_run))


async def _run(umls_dir: str, limit: int | None, dry_run: bool) -> None:
    logger = get_logger("load_umls_subset")

    loader = UMLSLoader(umls_meta_dir=umls_dir)
    index = loader.load(max_concepts=limit)
    click.echo(f"Parsed {len(index)} concepts from UMLS.")

    if dry_run:
        click.echo("Dry run — skipping Neo4j writes.")
        # Print a sample
        for i, concept in enumerate(loader.iter_concepts()):
            if i >= 5:
                break
            click.echo(f"  {concept['cui']:10s}  [{concept['entity_type']:10s}]  {concept['label']}")
        return

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
                    cui=concept["cui"],
                    label=concept.get("label") or concept["cui"],
                    entity_type=EntityType(concept.get("entity_type", "Unknown")),
                    lang_labels=concept.get("lang_labels", {}),
                )
                await merge_entity(entity)
                written += 1
            except Exception as exc:
                logger.warning("umls_write_error", cui=concept.get("cui"), error=str(exc))

        progress = int((i + len(batch)) / total * 100)
        click.echo(f"  Progress: {progress}%  ({i + len(batch)}/{total})", nl=False)
        click.echo("\r", nl=False)

    await close_driver()
    click.echo(f"\nLoaded {written}/{total} concepts into Neo4j.")


if __name__ == "__main__":
    main()
