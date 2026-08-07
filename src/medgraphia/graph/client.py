"""
Neo4j async driver with connection-pool management.
Use get_driver() / close_driver() at application lifespan boundaries.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from medgraphia.config import get_settings
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_driver: AsyncDriver | None = None
_lock = asyncio.Lock()


async def get_driver() -> AsyncDriver:
    """Return the shared AsyncDriver, creating it on first call (thread-safe)."""
    global _driver
    if _driver is None:
        async with _lock:
            if _driver is None:
                cfg = get_settings()
                _driver = AsyncGraphDatabase.driver(
                    cfg.neo4j_uri,
                    auth=(cfg.neo4j_user, cfg.neo4j_password.get_secret_value()),
                    database=cfg.neo4j_database,
                    # Connection pool: fail fast, recycle idle connections before WSL/Docker drops them
                    max_connection_lifetime=200,
                    max_connection_pool_size=50,
                    connection_timeout=10,
                    keep_alive=True,
                )
                logger.info("neo4j_driver_created", uri=cfg.neo4j_uri)
    return _driver


async def close_driver() -> None:
    """Close the shared driver.  Call once during application shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("neo4j_driver_closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager that yields a Neo4j async session."""
    driver = await get_driver()
    cfg = get_settings()
    async with driver.session(database=cfg.neo4j_database) as session:
        yield session


async def ping() -> bool:
    """Return True if Neo4j is reachable and the driver can open a session."""
    try:
        driver = await get_driver()
        await driver.verify_connectivity()
        return True
    except Exception as exc:
        logger.warning("neo4j_ping_failed", error=str(exc))
        return False
