"""
FastAPI dependency injectors: session store, vector store factory, rate limiting.
"""
from __future__ import annotations

from functools import lru_cache

from medgraphia.config import get_settings
from medgraphia.domain import Session
from medgraphia.vector.base import VectorStoreBase

# ---------------------------------------------------------------------------
# Persistent session store (Neo4j-backed)
# ---------------------------------------------------------------------------

async def get_session(session_id: str) -> Session | None:
    """Retrieve a session from Neo4j."""
    from medgraphia.graph.queries import get_chat_session
    return await get_chat_session(session_id)


async def create_or_get_session(session_id: str | None) -> Session:
    """
    Retrieve an existing session or create a new one in Neo4j.
    """
    from uuid import uuid4
    from medgraphia.graph.queries import get_chat_session, upsert_chat_session

    sid = session_id or str(uuid4())
    session = await get_chat_session(sid)
    
    if session is None:
        session = Session(session_id=sid)
        await upsert_chat_session(session)
    
    return session


async def save_session(session: Session) -> None:
    """
    Persist the session state and its latest messages to Neo4j.
    """
    from medgraphia.graph.queries import upsert_chat_session, save_chat_message

    # 1. Update session metadata (updated_at)
    await upsert_chat_session(session)

    # 2. Save only the last message (assuming others are already saved)
    # In a more robust implementation, we might diff or save all.
    # For now, we save all messages in the session to ensure persistence.
    for msg in session.messages:
        await save_chat_message(msg)


# ---------------------------------------------------------------------------
# Vector store factory (singleton per backend type)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreBase:
    """Return the Qdrant vector store backend (cached singleton)."""
    from medgraphia.vector.qdrant_store import QdrantStore
    return QdrantStore()
