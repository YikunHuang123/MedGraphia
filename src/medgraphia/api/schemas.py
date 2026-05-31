"""
Pydantic request / response DTOs for the FastAPI application.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from medgraphia.domain import Citation, Language, QueryType


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = None
    language: Language = Language.EN
    domain: str | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_paths_used: list[str] = Field(default_factory=list)
    model_used: str = ""
    faithfulness_score: float | None = None
    query_type: QueryType | None = None
    disclaimer: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class EntityQueryRequest(BaseModel):
    name: str = Field(..., min_length=1)
    lang: Language = Language.EN
    hops: int = Field(default=1, ge=1, le=3)


class EntityQueryResponse(BaseModel):
    entity: dict[str, Any]
    subgraph: dict[str, Any]


# ---------------------------------------------------------------------------
# Admin / pipeline
# ---------------------------------------------------------------------------

class PipelineStatusResponse(BaseModel):
    stage: str
    documents_processed: int
    documents_total: int
    nodes_created: int
    edges_created: int
    progress: float


class PipelineTriggerRequest(BaseModel):
    domain: str
    pubmed_query: str | None = None
    pubmed_limit: int = 200
    include_drugbank: bool = False
    include_ema_smpc: bool = False


class GraphStatsResponse(BaseModel):
    nodes: int
    relations: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class LivenessResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    neo4j: str
    qdrant: str
    overall: str