"""
Qdrant vector store implementation.
Supports dense-only and dense+sparse hybrid search (native Qdrant feature).
"""
from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from medgraphia.config import get_settings
from medgraphia.domain import Chunk
from medgraphia.logger import get_logger
from medgraphia.vector.base import VectorStoreBase

logger = get_logger(__name__)


class QdrantStore(VectorStoreBase):
    """AsyncQdrantClient-backed vector store with dense + sparse hybrid support."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._client = AsyncQdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key or None,
            timeout=30,
        )
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init_collection(
        self,
        collection_name: str,
        vector_size: int,
        sparse: bool = True,
    ) -> None:
        """
        Create collection with dense vectors and (optionally) sparse vectors.
        BGE-M3 produces 1024-dim dense vectors.
        """
        existing = {c.name for c in (await self._client.get_collections()).collections}
        if collection_name in existing:
            logger.debug("qdrant_collection_exists", name=collection_name)
            return

        vectors_config: dict[str, qmodels.VectorParams] = {
            "dense": qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            )
        }
        sparse_vectors_config: dict[str, qmodels.SparseVectorParams] | None = None
        if sparse:
            sparse_vectors_config = {
                "sparse": qmodels.SparseVectorParams(
                    index=qmodels.SparseIndexParams(on_disk=False)
                )
            }

        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            optimizers_config=qmodels.OptimizersConfigDiff(memmap_threshold=10_000),
        )
        logger.info("qdrant_collection_created", name=collection_name, vector_size=vector_size)

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception as exc:
            logger.warning("qdrant_health_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
    ) -> int:
        """Upsert chunks.  Each Chunk must have .embedding set."""
        points: list[qmodels.PointStruct] = []

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("chunk_missing_embedding", chunk_id=chunk.chunk_id)
                continue

            vectors: dict[str, Any] = {"dense": chunk.embedding}
            if chunk.sparse_embedding:
                # Convert {token_id: weight} → SparseVector
                indices = list(chunk.sparse_embedding.keys())
                values = [chunk.sparse_embedding[i] for i in indices]
                vectors["sparse"] = qmodels.SparseVector(indices=indices, values=values)

            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section_path": chunk.section_path,
                "language": chunk.language.value,
                "source_id": chunk.source.source_id,
                "source_title": chunk.source.source_title,
                "source_version": chunk.source.source_version,
                "text": chunk.text[:1000],  # truncate for storage
                "page": chunk.page,
            }
            points.append(
                qmodels.PointStruct(
                    id=chunk.chunk_id,
                    vector=vectors,
                    payload=payload,
                )
            )

        if not points:
            return 0

        await self._client.upsert(collection_name=collection_name, points=points, wait=True)
        logger.debug("qdrant_upserted", collection=collection_name, count=len(points))
        return len(points)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        qdrant_filter = _build_filter(filter_payload)
        results = await self._client.search(
            collection_name=collection_name,
            query_vector=qmodels.NamedVector(name="dense", vector=query_vector),
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {"chunk_id": r.payload.get("chunk_id", ""), "score": r.score, "payload": r.payload}
            for r in results
        ]

    async def hybrid_search(
        self,
        collection_name: str,
        query_dense: list[float],
        query_sparse: dict[int, float],
        limit: int = 10,
        filter_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Qdrant native hybrid search: dense COSINE + sparse BM25-style, fused with RRF.
        Falls back to dense-only if sparse vector is empty.
        """
        if not query_sparse:
            return await self.search(collection_name, query_dense, limit=limit,
                                     filter_payload=filter_payload)

        qdrant_filter = _build_filter(filter_payload)
        sparse_indices = list(query_sparse.keys())
        sparse_values = [query_sparse[i] for i in sparse_indices]

        results = await self._client.query_points(
            collection_name=collection_name,
            prefetch=[
                qmodels.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit * 3,
                ),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=limit * 3,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),  # type: ignore
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {"chunk_id": r.payload.get("chunk_id", ""), "score": r.score, "payload": r.payload}
            for r in results.points
        ]

    async def delete_by_doc_id(self, collection_name: str, doc_id: str) -> None:
        await self._client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
                )
            ),
        )
        logger.debug("qdrant_deleted_by_doc_id", collection=collection_name, doc_id=doc_id)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_filter(filter_payload: dict[str, Any] | None) -> qmodels.Filter | None:
    if not filter_payload:
        return None
    conditions = [
        qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
        for k, v in filter_payload.items()
    ]
    return qmodels.Filter(must=conditions)
