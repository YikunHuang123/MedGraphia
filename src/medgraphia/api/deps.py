"""
FastAPI dependency injectors: session store, vector store factory, rate limiting.
"""
from __future__ import annotations

from functools import lru_cache

from medgraphia.config import get_settings
from medgraphia.domain import Session
from medgraphia.vector.base import VectorStoreBase

# ---------------------------------------------------------------------------
# In-memory session store (Phase 8 will add Neo4j-backed persistence)
# ---------------------------------------------------------------------------
_sessions: dict[str, Session] = {}


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def create_or_get_session(session_id: str | None) -> Session:
    from uuid import uuid4
    sid = session_id or str(uuid4())
    if sid not in _sessions:
        _sessions[sid] = Session(session_id=sid)
    return _sessions[sid]


def save_session(session: Session) -> None:
    _sessions[session.session_id] = session


# ---------------------------------------------------------------------------
# Vector store factory (singleton per backend type)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreBase:
    """Return the Qdrant vector store backend (cached singleton)."""
    from medgraphia.vector.qdrant_store import QdrantStore
    return QdrantStore()
