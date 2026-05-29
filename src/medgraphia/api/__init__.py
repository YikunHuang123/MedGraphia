"""
FastAPI application factory with lifespan management.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

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

    # Load the admin API key from config into the in-memory key store
    _load_bootstrap_key()

    # Establish Neo4j connection and apply schema (idempotent)
    await get_driver()
    await apply_schema()

    logger.info("medgraphia_ready", port=cfg.api_port)
    yield

    # Graceful shutdown
    await close_driver()
    logger.info("medgraphia_stopped")


def create_app() -> FastAPI:
    """Application factory — called by uvicorn and tests."""
    cfg = get_settings()

    app = FastAPI(
        title="MedGraphia API",
        description="GraphRAG-powered multilingual medical knowledge Q&A",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS (restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # In production, configure this via settings
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AuditMiddleware)

    # Routers — more will be added in Phase 8
    app.include_router(health_router)

    return app


def run() -> None:
    """Entry point used by the medgraphia-api CLI script."""
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "medgraphia.api:create_app",
        factory=True,
        host=cfg.api_host,
        port=cfg.api_port,
        reload=False,  # Set to True for development if needed
        log_level=cfg.log_level.lower(),
    )
