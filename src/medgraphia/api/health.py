"""
Health check endpoints.
GET /health/live  — liveness probe (always 200 if process is running)
GET /health/ready — readiness probe (checks Neo4j and vector store connectivity)
"""
from __future__ import annotations

from fastapi import APIRouter

from medgraphia.api.schemas import LivenessResponse, ReadinessResponse
from medgraphia.graph.client import ping as neo4j_ping
from medgraphia.api.deps import get_vector_store
from medgraphia.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Kubernetes liveness probe — returns 200 if the process is alive."""
    return LivenessResponse(status="ok")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    """
    Readiness probe — checks that all required backing services are reachable.
    Returns 200 only when both Neo4j and the vector store respond.
    """
    neo4j_ok = await neo4j_ping()
    vector_ok = await get_vector_store().health()

    neo4j_status = "ok" if neo4j_ok else "unavailable"
    vector_status = "ok" if vector_ok else "unavailable"
    overall = "ready" if (neo4j_ok and vector_ok) else "degraded"

    logger.debug("health_ready", neo4j=neo4j_status, vector=vector_status)
    return ReadinessResponse(
        neo4j=neo4j_status,
        qdrant=vector_status,
        overall=overall,
    )
