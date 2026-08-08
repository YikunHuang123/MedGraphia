"""
FastAPI application factory with lifespan management.

Startup sequence:
1. Configure structured logging
2. Load the bootstrap admin API key from config
3. Connect to Neo4j and apply the graph schema (idempotent DDL)
4. Verify Qdrant connectivity
5. Initialise the Langfuse tracing client (no-op when disabled)
6. Initialise Redis client (no-op when REDIS_URL is not set)
7. Initialise Arq task-queue pool (no-op when REDIS_URL is not set)
8. Background warm-up of ML models

Shutdown sequence:
1. Flush any pending Langfuse events
2. Close Arq connection pool
3. Close Redis connection pool
4. Close the Neo4j driver connection pool
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medgraphia.api.auth import _load_bootstrap_key
from medgraphia.api.health import router as health_router
from medgraphia.api.middleware import AuditMiddleware
from medgraphia.config import get_settings
from medgraphia.graph.client import close_driver, get_driver
from medgraphia.graph.schema import apply_schema
from medgraphia.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic wired into FastAPI lifespan."""
    cfg = get_settings()
    configure_logging(cfg.log_level)
    logger.info("medgraphia_starting", storage=cfg.storage_backend, auth=cfg.auth_strategy)

    # 1. Bootstrap API key
    await _load_bootstrap_key()

    # 2. Neo4j: establish connection pool and apply schema DDL
    await get_driver()
    await apply_schema()
    logger.info("neo4j_ready")

    # 3. Qdrant: verify connectivity at startup so we fail fast if it's down
    await _check_qdrant(cfg.qdrant_url)

    # 4. Langfuse: initialise singleton (no-op when tracing is disabled)
    from medgraphia.observability import get_langfuse_client

    lf = get_langfuse_client()
    logger.info(
        "observability_ready",
        langfuse=lf.enabled,
        tracing=cfg.tracing_enabled,
    )

    # 6. Redis: optional cache layer (skipped when REDIS_URL is unset)
    from medgraphia.cache.redis_client import close_redis, get_redis

    await get_redis()  # triggers lazy init + connectivity check

    # 7. Arq: optional task-queue pool for durable pipeline execution
    from medgraphia.worker.arq_client import close_arq_pool, get_arq_pool

    arq_pool = await get_arq_pool()
    logger.info(
        "task_queue_ready",
        arq_enabled=arq_pool is not None,
    )

    # 8. Background Warm-up: eagerly load ML models and prime GPU kernels
    import asyncio

    from medgraphia.api.health import _warm_up_models

    asyncio.create_task(_warm_up_models())

    logger.info("medgraphia_ready", port=cfg.api_port)
    yield  # ─── application running ─────────────────────────────────────────

    # Shutdown: flush Langfuse, close Arq + Redis, close Neo4j driver
    lf.flush()

    await close_arq_pool()
    await close_redis()
    await close_driver()
    logger.info("medgraphia_stopped")


def create_app() -> FastAPI:
    """
    Application factory — called by uvicorn (factory mode) and test fixtures.

    All routers are registered here so they appear in the OpenAPI schema.
    """
    cfg = get_settings()

    app = FastAPI(
        title="MedGraphia API",
        description=(
            "GraphRAG-powered multilingual medical knowledge Q&A.\n\n"
            "Supports English, Chinese (Simplified), and German.\n\n"
            "Educational/portfolio project — not for clinical or commercial use."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        contact={"name": "Yikun Huang", "url": "https://github.com/YikunHuang123/MedGraphia"},
        license_info={"name": "MIT", "url": "https://github.com/YikunHuang123/MedGraphia/blob/main/LICENSE"},
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Origins come from CORS_ALLOWED_ORIGINS (comma-separated); defaults to "*"
    # for local dev. Set it to your actual domain(s) for a public deployment.
    origins = [o.strip() for o in cfg.cors_allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request audit / tracing middleware ───────────────────────────────────
    app.add_middleware(AuditMiddleware)

    # ── Routers ──────────────────────────────────────────────────────────────
    # Health (public — no auth)
    app.include_router(health_router)

    # Chat — synchronous and streaming Q&A
    from medgraphia.api.chat import router as chat_router

    app.include_router(chat_router)

    # Knowledge-graph exploration
    from medgraphia.api.knowledge import router as knowledge_router

    app.include_router(knowledge_router)

    # Admin (requires admin role)
    from medgraphia.api.admin import router as admin_router

    app.include_router(admin_router)

    return app


def run() -> None:
    """Entry point used by the `medgraphia-api` CLI script."""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "medgraphia.api:create_app",
        factory=True,
        host=cfg.api_host,
        port=cfg.api_port,
        reload=False,
        log_level=cfg.log_level.lower(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _check_qdrant(qdrant_url: str) -> None:
    """
    Verify that Qdrant is reachable at startup.

    Uses the vector-store singleton's health() method so the same client
    that handles queries is used for the connectivity check.
    """
    try:
        from medgraphia.api.deps import get_vector_store

        store = get_vector_store()
        ok = await store.health()
        if ok:
            logger.info("qdrant_ready", url=qdrant_url)
        else:
            logger.warning("qdrant_unreachable", url=qdrant_url)
    except Exception as exc:
        logger.warning("qdrant_check_failed", url=qdrant_url, error=str(exc))
