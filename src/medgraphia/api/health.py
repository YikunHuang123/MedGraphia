"""
Health check endpoints.
GET /health/live  — liveness probe (always 200 if process is running)
GET /health/ready — readiness probe (checks Neo4j, vector store, and model pre-warming)
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Response, status

from medgraphia.api.deps import get_vector_store
from medgraphia.api.schemas import LivenessResponse, ReadinessResponse
from medgraphia.graph.client import ping as neo4j_ping
from medgraphia.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

# ── Global warming state ───────────────────────────────────────────────────
_IS_WARM = False


@router.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Kubernetes liveness probe — returns 200 if the process is alive."""
    return LivenessResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(response: Response) -> ReadinessResponse:
    """
    Readiness probe — checks backing services AND model pre-warming status.
    Returns 503 if the system is still warming up or services are down.
    """
    neo4j_ok = await neo4j_ping()
    vector_ok = await get_vector_store().health()

    neo4j_status = "ok" if neo4j_ok else "unavailable"
    vector_status = "ok" if vector_ok else "unavailable"
    warm_status = "ok" if _IS_WARM else "warming_up"

    is_ready = neo4j_ok and vector_ok and _IS_WARM
    overall = "ready" if is_ready else "not_ready"

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    logger.debug("health_ready", neo4j=neo4j_status, vector=vector_status, warming=warm_status)
    
    # We add 'warming' to the response (note: requires schema update or using extra)
    return ReadinessResponse(
        neo4j=neo4j_status,
        qdrant=vector_status,
        overall=f"{overall} ({warm_status})",
    )


async def _warm_up_models() -> None:
    """
    Background task: eagerly load models and perform dummy inference.
    Prevents the first user request from hitting a massive 'cold start' penalty.
    """
    global _IS_WARM
    logger.info("warmup_started", message="Eagerly loading models and priming GPU/MPS kernels...")

    try:
        from medgraphia.api.chat import _get_retrieval
        
        # 1. Load & Prime Retrieval Pipeline
        # This triggers loading of GLiNER, BERT-NER, SapBERT, BGE-M3, and Reranker
        retrieval = await _get_retrieval()
        
        logger.info("warmup_retrieval_loading_done")
        
        # 2. Perform a dummy 'synthetic' request to compile GPU kernels
        # We use a short medical term to exercise all paths
        await retrieval.execute(query="Metformin", top_k=1)
        
        logger.info("warmup_synthetic_inference_done")
        
        # 3. Mark as warm
        _IS_WARM = True
        logger.info("warmup_complete", message="All models are hot and ready for traffic.")

    except Exception as exc:
        logger.error("warmup_failed", error=str(exc))
        # Note: we don't set _IS_WARM=True here, so readiness will continue to fail.
        # This is the correct behaviour for enterprise high-availability.
