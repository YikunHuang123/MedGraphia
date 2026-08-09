"""
Admin endpoints — require the `admin` role.

POST /admin/pipeline/trigger  — Kick off the offline build pipeline
GET  /admin/pipeline/status   — Poll the latest pipeline run status
GET  /admin/graph/stats       — Detailed node + relation counts
POST /admin/keys              — Generate a new user API key
GET  /admin/keys              — List all active API key prefixes (redacted)
DELETE /admin/keys/{prefix}   — Revoke a key by its 8-char prefix

All endpoints are protected by `require_admin` (role == "admin").

Pipeline execution strategy:
• Redis available (REDIS_URL set): job is enqueued via Arq and executed by
  the dedicated worker process.  The response includes a ``job_id`` that can
  be used with arq's native job-result API.
• No Redis (REDIS_URL unset): falls back to asyncio.create_task running in
  the FastAPI process — identical behaviour, lower durability.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, status

from medgraphia.api.auth import generate_api_key, register_key, require_admin
from medgraphia.api.schemas import (
    PipelineTriggerRequest,
)
from medgraphia.graph.queries import (
    delete_api_keys_by_prefix,
    get_graph_stats,
    get_pipeline_status,
    list_api_keys,
    set_api_key_daily_limit,
    upsert_pipeline_status,
)
from medgraphia.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# Asyncio task handle — only used in no-Redis fallback mode
_pipeline_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# POST /admin/pipeline/trigger
# ---------------------------------------------------------------------------


@router.post(
    "/pipeline/trigger",
    response_model=dict[str, str],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger the offline knowledge-graph build pipeline",
)
async def trigger_pipeline(
    body: PipelineTriggerRequest,
    principal: dict = Depends(require_admin),
) -> dict[str, str]:
    """
    Start the offline build pipeline in the background.
    """
    global _pipeline_task

    # domain=None means "global build from whatever is in data/raw"; a real
    # direction scopes the status/concurrency key so different directions
    # can be tracked (and polled) independently.
    scope = body.domain or "global"

    # ── Concurrency Check (Distributed) ──────────────────────────────────
    # Check the database to see if a run is already active anywhere
    current = await get_pipeline_status(scope)
    if current and current.get("stage") == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A pipeline run for '{scope}' is already in progress.",
        )

    logger.info(
        "pipeline_trigger_requested",
        domain=scope,
        triggered_by=principal.get("id", "admin"),
    )

    # Reset status in database
    init_status = {
        "stage": "starting",
        "progress": 0.0,
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
    }
    await upsert_pipeline_status(scope, init_status)

    # ── Execution strategy ────────────────────────────────────────────────
    # Prefer Arq (durable, isolated worker process) when Redis is available;
    # otherwise fall back to asyncio.create_task (no-Redis mode).
    from medgraphia.worker.arq_client import get_arq_pool

    arq_pool = await get_arq_pool()
    if arq_pool is not None:
        job = await arq_pool.enqueue_job(
            "task_build_pipeline",
            domain=scope,
            pubmed_query=body.pubmed_query or "",
            pubmed_limit=body.pubmed_limit,
            include_drugbank=body.include_drugbank,
            include_ema_smpc=body.include_ema_smpc,
        )
        logger.info("pipeline_enqueued", job_id=job.job_id, domain=scope)
        return {
            "status": "accepted",
            "job_id": job.job_id,
            "message": f"Pipeline queued for '{scope}'.",
        }

    # Lite / no-Redis fallback
    global _pipeline_task
    _pipeline_task = asyncio.create_task(
        _run_pipeline_background(body),
        name=f"pipeline_{scope}",
    )
    logger.info("pipeline_task_created", domain=scope, mode="in_process_no_redis")
    return {
        "status": "accepted",
        "job_id": None,
        "message": f"Pipeline started for '{scope}' (in-process mode).",
    }


# ---------------------------------------------------------------------------
# GET /admin/pipeline/status
# ---------------------------------------------------------------------------


@router.get(
    "/pipeline/status",
    summary="Current pipeline run status",
)
async def pipeline_status(
    domain: str = "global",
    principal: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Return the persistent state of the most recent pipeline execution."""
    status_data = await get_pipeline_status(domain)
    if not status_data:
        return {"stage": "idle", "domain": domain}
    return status_data


# ---------------------------------------------------------------------------
# GET /admin/graph/stats
# ---------------------------------------------------------------------------


@router.get(
    "/graph/stats",
    summary="Knowledge-graph node and relationship counts",
)
async def admin_graph_stats(
    principal: dict = Depends(require_admin),
) -> dict[str, int]:
    return await get_graph_stats()


# ---------------------------------------------------------------------------
# POST /admin/keys — generate a new API key
# ---------------------------------------------------------------------------


@router.post(
    "/keys",
    status_code=status.HTTP_201_CREATED,
    summary="Generate a new user API key",
)
async def create_api_key(
    role: str = "user",
    daily_limit: int | None = None,
    custom_key: str | None = None,
    principal: dict = Depends(require_admin),
) -> dict[str, Any]:
    if custom_key:
        if len(custom_key) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="custom_key must be at least 6 characters.",
            )
        new_key = custom_key
    else:
        new_key = generate_api_key()
    await register_key(new_key, role=role, daily_limit=daily_limit)
    logger.info(
        "api_key_created", prefix=new_key[:8], by=principal.get("id", "admin"), daily_limit=daily_limit
    )
    return {
        "api_key": new_key,
        "role": role,
        "prefix": new_key[:8],
        "daily_limit": daily_limit,
        "note": "Store this key securely — it cannot be retrieved again.",
    }


# ---------------------------------------------------------------------------
# GET /admin/keys — list active keys
# ---------------------------------------------------------------------------


@router.get(
    "/keys",
    summary="List active API key prefixes",
)
async def list_api_keys_endpoint(
    principal: dict = Depends(require_admin),
) -> list[dict[str, Any]]:
    keys = await list_api_keys()
    return [
        {"prefix": k["prefix"] + "…", "role": k["role"], "daily_limit": k.get("daily_limit")}
        for k in keys
    ]


# ---------------------------------------------------------------------------
# PATCH /admin/keys/{prefix}/limit — set or clear a key's daily request cap
# ---------------------------------------------------------------------------


@router.patch(
    "/keys/{prefix}/limit",
    summary="Set (or clear, with daily_limit omitted) a key's daily request cap",
)
async def update_api_key_limit(
    prefix: str = Path(..., min_length=8, max_length=8),
    daily_limit: int | None = None,
    principal: dict = Depends(require_admin),
) -> dict[str, Any]:
    count = await set_api_key_daily_limit(prefix, daily_limit)
    if count == 0:
        raise HTTPException(status_code=404, detail="Key prefix not found.")
    logger.info(
        "api_key_limit_updated", prefix=prefix, daily_limit=daily_limit, by=principal.get("id", "admin")
    )
    return {"status": "updated", "prefix": prefix, "daily_limit": daily_limit, "count": count}


# ---------------------------------------------------------------------------
# DELETE /admin/keys/{prefix} — revoke a key
# ---------------------------------------------------------------------------


@router.delete(
    "/keys/{prefix}",
    summary="Revoke an API key by its 8-character prefix",
)
async def delete_api_key(
    prefix: str = Path(..., min_length=8, max_length=8),
    principal: dict = Depends(require_admin),
) -> dict[str, Any]:
    count = await delete_api_keys_by_prefix(prefix)
    if count == 0:
        raise HTTPException(status_code=404, detail="Key prefix not found.")
    return {"status": "revoked", "prefix": prefix, "count": count}


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline_background(body: PipelineTriggerRequest) -> None:
    """Execute the build pipeline and update persistent status."""

    scope = body.domain or "global"

    async def _update_db(**kwargs: Any) -> None:
        await upsert_pipeline_status(scope, kwargs)

    try:
        from medgraphia.ingestion.pipeline import BuildConfig, build_graph_flow

        await _update_db(stage="running", progress=0.05)

        cfg = BuildConfig(
            domain=body.domain,
            pubmed_query=body.pubmed_query,
            pubmed_limit=body.pubmed_limit,
            include_drugbank=body.include_drugbank,
            include_ema_smpc=body.include_ema_smpc,
        )
        await build_graph_flow(cfg)

        stats = await get_graph_stats()
        await _update_db(
            stage="completed",
            progress=1.0,
            finished_at=time.time(),
            nodes_created=stats.get("nodes", 0),
            edges_created=stats.get("relations", 0),
        )

    except Exception as exc:
        await _update_db(stage="failed", error=str(exc), finished_at=time.time())
        logger.error("pipeline_background_failed", error=str(exc))
