"""
Arq connection pool singleton for the FastAPI process.

The API process uses this pool exclusively for *enqueueing* jobs.
Actual execution happens in the separate worker process

When REDIS_URL is not configured the pool is None and enqueueing is
silently skipped — the admin endpoint falls back to asyncio.create_task.
"""

from __future__ import annotations

import asyncio
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)

_pool: Any | None = None
_init_lock: asyncio.Lock | None = None
_disabled: bool = False


async def get_arq_pool() -> Any | None:
    """
    Return the shared Arq Redis pool, or None if Arq/Redis is unavailable.
    """
    global _pool, _init_lock, _disabled

    if _disabled:
        return None
    if _pool is not None:
        return _pool

    if _init_lock is None:
        _init_lock = asyncio.Lock()

    async with _init_lock:
        if _pool is not None:
            return _pool
        if _disabled:
            return None

        from medgraphia.config import get_settings

        cfg = get_settings()

        if not cfg.redis_url:
            _disabled = True
            return None

        try:
            from arq import create_pool  # type: ignore[import]
            from arq.connections import RedisSettings  # type: ignore[import]

            _pool = await create_pool(RedisSettings.from_dsn(cfg.redis_url))
            logger.info("arq_pool_ready", url=cfg.redis_url)
        except ImportError:
            logger.warning("arq_not_installed", hint="pip install arq>=0.25")
            _disabled = True
        except Exception as exc:
            logger.warning("arq_pool_init_failed", error=str(exc))
            _disabled = True

    return _pool


async def close_arq_pool() -> None:
    """Close the Arq pool and reset module state."""
    global _pool, _disabled
    if _pool is not None:
        try:
            await _pool.aclose()
        except Exception:
            pass
        _pool = None
    _disabled = False
    logger.info("arq_pool_closed")


def reset_for_testing() -> None:
    """Reset all module-level state. Only call from test fixtures."""
    global _pool, _init_lock, _disabled
    _pool = None
    _init_lock = None
    _disabled = False
