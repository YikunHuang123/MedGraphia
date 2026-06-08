"""
Arq WorkerSettings — entry point for the standalone worker process.

Start the worker::

    arq medgraphia.worker.WorkerSettings

Or programmatically (useful for tests)::

    from arq import run_worker
    from medgraphia.worker import WorkerSettings
    run_worker(WorkerSettings)

The worker is optional: when REDIS_URL is not configured the FastAPI
process falls back to asyncio.create_task for pipeline execution.
"""

from __future__ import annotations

from medgraphia.worker.tasks import task_build_pipeline

__all__ = ["WorkerSettings", "task_build_pipeline"]


async def _on_startup(ctx: dict) -> None:
    """Initialise shared resources once per worker process."""
    from medgraphia.config import get_settings
    from medgraphia.graph.client import get_driver
    from medgraphia.graph.schema import apply_schema
    from medgraphia.logger import configure_logging

    cfg = get_settings()
    configure_logging(cfg.log_level)

    ctx["neo4j_driver"] = await get_driver()
    await apply_schema()


async def _on_shutdown(ctx: dict) -> None:
    """Clean up resources when the worker exits."""
    from medgraphia.graph.client import close_driver

    await close_driver()


def _make_worker_settings():  # type: ignore[return]
    """
    Build the WorkerSettings class dynamically so that importing this module
    does not hard-fail when arq is not installed (deployments without Redis).
    """
    try:
        from arq.connections import RedisSettings  # type: ignore[import]
    except ImportError:
        return None

    from medgraphia.config import get_settings

    cfg = get_settings()

    redis_url = cfg.redis_url or "redis://localhost:6379/0"

    class WorkerSettings:
        functions = [task_build_pipeline]
        on_startup = _on_startup
        on_shutdown = _on_shutdown
        redis_settings = RedisSettings.from_dsn(redis_url)
        # Allow up to 2 concurrent pipeline jobs (one per domain)
        max_jobs = 2
        # Pipelines can take up to 2 hours; kill after that to prevent stalls
        job_timeout = 7_200
        # Keep job results for 24 hours so the admin UI can query them
        keep_result = 86_400

    return WorkerSettings


WorkerSettings = _make_worker_settings()
