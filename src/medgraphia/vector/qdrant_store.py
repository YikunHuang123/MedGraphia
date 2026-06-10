"""
Qdrant vector store implementation.
Supports dense-only and dense+sparse hybrid search.
"""

from __future__ import annotations

import uuid
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
                "sparse": qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=False))
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
        batch_size: int = 100,
    ) -> int:
        """Upsert chunks in batches.  Each Chunk must have .embedding set."""
        total_written = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points: list[qmodels.PointStruct] = []

            for chunk in batch:
                if chunk.embedding is None:
                    logger.warning("chunk_missing_embedding", chunk_id=chunk.chunk_id)
                    continue

                vectors: dict[str, Any] = {"dense": chunk.embedding}
                if chunk.sparse_embedding:
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
                    "text": chunk.text,
                    "parent_text": chunk.parent_text,
                    "page": chunk.page,
                }
                points.append(
                    qmodels.PointStruct(
                        id=chunk.chunk_id,
                        vector=vectors,
                        payload=payload,
                    )
                )

            if points:
                await self._client.upsert(collection_name=collection_name, points=points, wait=True)
                total_written += len(points)
                logger.debug("qdrant_batch_upserted", collection=collection_name, count=len(points))

        return total_written

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
            return await self.search(
                collection_name, query_dense, limit=limit, filter_payload=filter_payload
            )

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

    async def set_payload_batch(
        self,
        collection_name: str,
        updates: list[tuple[str, dict[str, Any]]],
        batch_size: int = 100,
    ) -> int:
        """
        Patch payload fields on existing points without touching vectors.

        Args:
            updates:    List of (point_id, payload_patch) pairs.
            batch_size: Points per Qdrant request.

        Returns:
            Total number of points updated.
        """
        total = 0
        for i in range(0, len(updates), batch_size):
            batch = updates[i : i + batch_size]
            for point_id, patch in batch:
                await self._client.set_payload(
                    collection_name=collection_name,
                    payload=patch,
                    points=[point_id],
                    wait=True,
                )
            total += len(batch)
            logger.debug("qdrant_payload_patched", collection=collection_name, count=len(batch))
        return total

    async def delete_by_doc_id(self, collection_name: str, doc_id: str) -> None:
        await self._client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))
                    ]
                )
            ),
        )
        logger.debug("qdrant_deleted_by_doc_id", collection=collection_name, doc_id=doc_id)

    # ------------------------------------------------------------------
    # Entity vectors (second collection, dense-only)
    # ------------------------------------------------------------------

    async def upsert_entities(
        self,
        collection_name: str,
        entity_points: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert entity dense embeddings into the entity collection in batches.
        """
        total_written = 0

        for i in range(0, len(entity_points), batch_size):
            batch = entity_points[i : i + batch_size]
            points: list[qmodels.PointStruct] = []

            for ep in batch:
                if not ep.get("embedding"):
                    logger.warning("entity_missing_embedding", cui=ep.get("cui", "?"))
                    continue
                points.append(
                    qmodels.PointStruct(
                        id=_entity_uuid(ep["cui"]),
                        vector={"dense": ep["embedding"]},
                        payload={
                            "cui": ep["cui"],
                            "label": ep["label"],
                            "entity_type": ep.get("entity_type", "Unknown"),
                            "lang_zh": ep.get("lang_zh", ""),
                            "lang_de": ep.get("lang_de", ""),
                        },
                    )
                )

            if points:
                await self._client.upsert(collection_name=collection_name, points=points, wait=True)
                total_written += len(points)
                logger.debug(
                    "qdrant_entities_batch_upserted", collection=collection_name, count=len(points)
                )

        return total_written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity_uuid(cui: str) -> str:
    """Deterministic UUID for an entity CUI used as the Qdrant point ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, cui))


def _build_filter(filter_payload: dict[str, Any] | None) -> qmodels.Filter | None:
    if not filter_payload:
        return None
    conditions = [
        qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
        for k, v in filter_payload.items()
    ]
    return qmodels.Filter(must=conditions)
