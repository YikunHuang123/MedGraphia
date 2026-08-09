#!/usr/bin/env python3
"""
Admin utility: clear the Redis-backed query NER/entity-linking cache
(cache/ner_cache.py, key pattern "ner:{lang}:{hash}") — this is what caches
each user question's extracted entities so repeat queries skip re-running
NER + entity linking. Does not touch rate-limit counters or the Arq task
queue, which live under separate Redis key prefixes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.cache.redis_client import get_redis

_PATTERN = "ner:*"


async def clear_qa_cache() -> int:
    redis = await get_redis()
    if redis is None:
        click.echo("  ❌ Redis is not available (check REDIS_URL).")
        return 0

    deleted = 0
    async for key in redis.scan_iter(match=_PATTERN, count=500):
        await redis.delete(key)
        deleted += 1
    return deleted


@click.command()
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def main(force: bool) -> None:
    """Clear all cached question → NER/entity-linking results from Redis."""
    if not force:
        if not click.confirm(f"Delete all Redis keys matching '{_PATTERN}'?"):
            click.echo("Aborted.")
            sys.exit(0)

    deleted = asyncio.run(clear_qa_cache())
    click.echo(f"✨ Cleared {deleted} cached question(s) from Redis.")


if __name__ == "__main__":
    main()
