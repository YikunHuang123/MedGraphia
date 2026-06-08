"""
Async Redis client singleton with graceful degradation.

When REDIS_URL is not configured (or Redis is unreachable) every public
function is a silent no-op — the rest of the application never needs to
check for None or catch exceptions from this module.
"""

from __future__ import annotations

import asyncio
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)

_client: Any | None = None
_init_lock: asyncio.Lock | None = None
_disabled: bool = False  # set True after first failed init to skip retries


async def get_redis() -> Any | None:
    """
    Return the shared async Redis client, or None if Redis is disabled/unavailable.

    Lazily initialises the connection on first call and caches it for the
    lifetime of the process.  Thread-safe via asyncio.Lock.
    """
    global _client, _init_lock, _disabled

    if _disabled:
        return None
    if _client is not None:
        return _client

    if _init_lock is None:
        _init_lock = asyncio.Lock()

    async with _init_lock:
        # Double-checked locking
        if _client is not None:
            return _client
        if _disabled:
            return None

        from medgraphia.config import get_settings

        cfg = get_settings()

        if not cfg.redis_url:
            _disabled = True
            return None

        try:
            import redis.asyncio as aioredis  # type: ignore[import]

            client = aioredis.from_url(
                cfg.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=False,
            )
            await client.ping()
            _client = client
            logger.info("redis_connected", url=cfg.redis_url)
        except ImportError:
            logger.warning("redis_not_installed", hint="pip install redis>=5.0")
            _disabled = True
        except Exception as exc:
            logger.warning("redis_unavailable", error=str(exc))
            _disabled = True

    return _client


async def close_redis() -> None:
    """Close the connection pool and reset module state."""
    global _client, _disabled
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None
    _disabled = False
    logger.info("redis_closed")


def reset_for_testing() -> None:
    """Reset all module-level state. Only call from test fixtures."""
    global _client, _init_lock, _disabled
    _client = None
    _init_lock = None
    _disabled = False
