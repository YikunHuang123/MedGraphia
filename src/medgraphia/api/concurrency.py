"""
Redis-backed global concurrency gate: caps simultaneous in-flight chat
pipeline executions, queuing (not rejecting) requests over the cap.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from medgraphia.cache.redis_client import get_redis
from medgraphia.config import get_settings
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_ACTIVE_KEY = "concurrency:active_requests"
_KEY_TTL_SECONDS = 300  # refreshed on every touch; only lapses once traffic is fully idle
_POLL_INTERVAL_SECONDS = 1.5
_MAX_QUEUE_WAIT_SECONDS = 90


class ConcurrencyQueueTimeout(Exception):
    """Raised when a request waited too long in the concurrency queue."""


async def _touch(redis, delta: int) -> int:
    value = await redis.incrby(_ACTIVE_KEY, delta)
    await redis.expire(_ACTIVE_KEY, _KEY_TTL_SECONDS)
    return value


async def wait_for_slot() -> AsyncIterator[str]:
    """
    Yields one friendly status message if the caller is queued behind
    `concurrency_max_active` other in-flight requests, then polls until a
    slot frees up. Completes immediately (no yields) if a slot is free.

    Callers MUST call release_slot() in a finally block after this generator
    is fully drained, even if it never yielded anything — release_slot() is
    a no-op when there was nothing to acquire (disabled / Redis down), so the
    pairing is always safe to call unconditionally.
    """
    cfg = get_settings()
    if not cfg.concurrency_limit_enabled:
        return

    redis = await get_redis()
    if redis is None:
        return  # fail open, matches rate_limit.py's behavior when Redis is down

    waited = 0.0
    announced = False
    while True:
        active = await _touch(redis, 1)
        if active <= cfg.concurrency_max_active:
            return

        # Over capacity — give back the slot we just claimed and wait.
        await _touch(redis, -1)
        if not announced:
            yield (
                "This demo is handling other requests right now — "
                "you're in the queue and will start shortly."
            )
            announced = True

        if waited >= _MAX_QUEUE_WAIT_SECONDS:
            logger.warning("concurrency_queue_timeout", waited=waited)
            raise ConcurrencyQueueTimeout()

        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS


async def release_slot() -> None:
    """Give back a slot acquired via a fully-drained wait_for_slot() call."""
    cfg = get_settings()
    if not cfg.concurrency_limit_enabled:
        return
    redis = await get_redis()
    if redis is None:
        return
    await _touch(redis, -1)
