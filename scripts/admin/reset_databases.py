#!/usr/bin/env python3
"""
Admin utility: wipe all data from Neo4j and Qdrant to start from a clean slate.
DANGER: This will delete everything in the configured databases.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.graph.client import get_session
from medgraphia.vector.qdrant_store import QdrantStore


async def reset_neo4j() -> None:
    click.echo("  → Wiping Neo4j data…")
    try:
        async with get_session() as session:
            # Delete all nodes and relationships
            await session.run("MATCH (n) DETACH DELETE n")
        click.echo("  ✓ Neo4j is now empty.")
    except Exception as exc:
        click.echo(f"  ❌ Neo4j reset failed: {exc}")


async def reset_qdrant() -> None:
    cfg = get_settings()
    store = QdrantStore()
    
    click.echo("  → Wiping Qdrant collections…")
    # We list potential collections from config
    collections = [cfg.qdrant_collection_chunks, cfg.qdrant_collection_entities]
    
    for coll in collections:
        try:
            await store._client.delete_collection(coll)
            click.echo(f"  ✓ Deleted collection: {coll}")
        except Exception as exc:
            # It's fine if it doesn't exist
            click.echo(f"  - Collection {coll} skipped (may not exist).")


@click.command()
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def main(force: bool) -> None:
    """Wipe all knowledge graph data from Neo4j and Qdrant."""
    click.echo("\n" + "!" * 60)
    click.echo("  WARNING: THIS WILL DELETE ALL DATA IN NEO4J AND QDRANT!")
    click.echo("!" * 60 + "\n")

    if not force:
        if not click.confirm("Are you absolutely sure you want to proceed?"):
            click.echo("Aborted.")
            sys.exit(0)

    async def _run_all() -> None:
        await reset_neo4j()
        await reset_qdrant()

    asyncio.run(_run_all())
    click.echo("\n✨ Databases reset successfully. You can now run build_graph.py from zero.\n")


if __name__ == "__main__":
    main()
