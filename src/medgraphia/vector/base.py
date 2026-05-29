"""
Abstract interface for vector store backends.
Rest of the codebase uses this interface for Qdrant storage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from medgraphia.domain import Chunk


class VectorStoreBase(ABC):
    """Pluggable vector store contract."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def init_collection(self, collection_name: str, vector_size: int) -> None:
        """Create the collection if it does not exist."""

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the backend is reachable."""

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abstractmethod
    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
    ) -> int:
        """
        Upsert a batch of chunks.  Each Chunk must have .embedding set.
        Returns the number of points successfully written.
        """

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dense vector search.
        Returns a list of dicts: {"chunk_id": ..., "score": ..., "payload": {...}}
        """

    @abstractmethod
    async def hybrid_search(
        self,
        collection_name: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        limit: int = 10,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid dense + sparse search (Qdrant native; Chroma falls back to dense).
        Returns same format as search().
        """

    @abstractmethod
    async def delete_by_doc_id(self, collection_name: str, doc_id: str) -> None:
        """Remove all vectors whose payload.doc_id matches doc_id."""
