"""
Arq task definitions for the MedGraphia worker process.

Each async function here is a first-class Arq task.  The worker picks tasks
off the Redis queue and executes them in a dedicated asyncio event loop,
completely isolated from the FastAPI process.

Task signatures follow the Arq convention:
    async def task_name(ctx: dict, arg1, arg2, ...) -> result

``ctx`` carries the worker context (arq pool, any objects added in on_startup).
"""

from __future__ import annotations

import time
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)


async def task_build_pipeline(
    ctx: dict,
    domain: str,
    pubmed_query: str,
    pubmed_limit: int,
    include_drugbank: bool,
    include_ema_smpc: bool,
) -> dict[str, Any]:
    """
    Full offline knowledge-graph build pipeline.

    Writes granular progress to Neo4j so the admin API can poll it via
    GET /admin/pipeline/status.  On failure the stage is set to "failed"
    with the error message; on success to "completed" with node/edge counts.

    Returns a summary dict that Arq stores as the job result (retrievable
    via arq JobResult for up to ``keep_result`` seconds).
    """
    from medgraphia.graph.queries import get_graph_stats, upsert_pipeline_status
    from medgraphia.ingestion.pipeline import BuildConfig, build_graph_flow

    async def _update(**kwargs: Any) -> None:
        await upsert_pipeline_status(domain, kwargs)

    logger.info("pipeline_task_started", domain=domain)

    try:
        await _update(stage="running", progress=0.05, started_at=time.time())

        cfg = BuildConfig(
            domain=domain,
            pubmed_query=pubmed_query or None,
            pubmed_limit=pubmed_limit,
            include_drugbank=include_drugbank,
            include_ema_smpc=include_ema_smpc,
        )
        await build_graph_flow(cfg)

        stats = await get_graph_stats()
        await _update(
            stage="completed",
            progress=1.0,
            finished_at=time.time(),
            nodes_created=stats.get("nodes", 0),
            edges_created=stats.get("relations", 0),
        )

        logger.info("pipeline_task_completed", domain=domain, stats=stats)
        return {"status": "completed", "domain": domain, **stats}

    except Exception as exc:
        await _update(stage="failed", error=str(exc), finished_at=time.time())
        logger.error("pipeline_task_failed", domain=domain, error=str(exc))
        raise
