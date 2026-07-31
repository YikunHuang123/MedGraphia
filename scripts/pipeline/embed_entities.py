#!/usr/bin/env python3
"""
Embed entity nodes from Neo4j using SapBERT and write to Qdrant.

Reads all entity nodes (any EntityType) from Neo4j,
encodes their canonical labels with SapBERT, and upserts dense vectors
into the entity Qdrant collection for entity-level similarity search.

Usage::

    # Use collection name from .env (qdrant_collection_entities)
    python scripts/embed_entities.py

    # Override collection
    python scripts/embed_entities.py --collection medgraphia_entities_v2

    # Dry-run: count entities in Neo4j without embedding
    python scripts/embed_entities.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed Neo4j entity nodes with SapBERT and store vectors in Qdrant."
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override the Qdrant collection name (default: from .env qdrant_collection_entities)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count entities in Neo4j and exit without embedding",
    )
    return parser.parse_args()


async def _run(collection: str | None, dry_run: bool) -> int:
    from medgraphia.logger import configure_logging

    configure_logging("INFO")

    from medgraphia.graph.queries import get_all_entities

    entities = await get_all_entities()
    print(f"Found {len(entities)} entity nodes in Neo4j")

    if dry_run or not entities:
        return 0

    from medgraphia.ingestion.embedder import EntityEmbedder

    embedder = EntityEmbedder.from_settings()
    count = await embedder.embed_and_store(collection_name=collection)
    print(f"Embedded and stored {count} entities → Qdrant")
    return count


def main() -> None:
    args = _parse_args()
    count = asyncio.run(_run(args.collection, args.dry_run))
    sys.exit(0 if count >= 0 else 1)


if __name__ == "__main__":
    main()
