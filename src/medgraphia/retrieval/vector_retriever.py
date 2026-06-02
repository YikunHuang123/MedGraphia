"""
Vector retriever.

Encodes the query with BGE-M3 and runs a hybrid (dense + sparse) search
against the Qdrant chunk collection.  Falls back to dense-only search if
the sparse vector is empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class VectorHit:
    """A single chunk result from the vector retriever."""
    chunk_id: str
    score: float
    text: str
    doc_id: str = ""
    source_id: str = ""
    source_title: str = ""
    source_version: str = ""
    section_path: str = ""
    language: str = ""
    page: int | None = None

    @classmethod
    def from_qdrant_payload(cls, chunk_id: str, score: float, payload: dict[str, Any]) -> "VectorHit":
        return cls(
            chunk_id=chunk_id,
            score=score,
            text=payload.get("text") or "",
            doc_id=payload.get("doc_id") or "",
            source_id=payload.get("source_id") or "",
            source_title=payload.get("source_title") or "",
            source_version=payload.get("source_version") or "",
            section_path=payload.get("section_path") or "",
            language=payload.get("language") or "",
            page=payload.get("page"),
        )


@dataclass
class VectorRetrievalResult:
    """Aggregated result from vector retrieval."""
    hits: list[VectorHit] = field(default_factory=list)
    query: str = ""

    @property
    def texts(self) -> list[str]:
        return [h.text for h in self.hits]

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]


# ---------------------------------------------------------------------------
# VectorRetriever
# ---------------------------------------------------------------------------

class VectorRetriever:
    """
    BGE-M3 query encoder + Qdrant hybrid search retriever.

    Usage::

        retriever = VectorRetriever.from_settings()
        result = await retriever.retrieve(
            "interaction between metformin and sitagliptin", limit=20
        )
        for hit in result.hits:
            print(hit.score, hit.text[:100])
    """

    def __init__(
        self,
        collection_name: str | None = None,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 1,
    ) -> None:
        self._collection_name = collection_name  # resolved from settings if None
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None  # lazy-loaded
        self._store: Any = None  # lazy-loaded QdrantStore

    @classmethod
    def from_settings(cls) -> "VectorRetriever":
        from medgraphia.config import get_settings
        cfg = get_settings()
        return cls(
            collection_name=cfg.qdrant_collection_chunks,
            batch_size=cfg.embedding_batch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        limit: int = 20,
        filter_payload: dict[str, Any] | None = None,
    ) -> VectorRetrievalResult:
        """
        Encode the query and run hybrid search.

        Args:
            query:          Raw query text (any language).
            limit:          Maximum number of results.
            filter_payload: Optional Qdrant payload filter (e.g. {"language": "en"}).

        Returns:
            VectorRetrievalResult sorted by score descending.
        """
        result = VectorRetrievalResult(query=query)

        try:
            dense, sparse = self._encode_query(query)
            store = self._get_store()
            collection = self._resolve_collection()

            raw = await store.hybrid_search(
                collection_name=collection,
                query_dense=dense,
                query_sparse=sparse,
                limit=limit,
                filter_payload=filter_payload,
            )

            for item in raw:
                hit = VectorHit.from_qdrant_payload(
                    chunk_id=item.get("chunk_id", ""),
                    score=float(item.get("score", 0.0)),
                    payload=item.get("payload", {}),
                )
                result.hits.append(hit)

            logger.info(
                "vector_retriever_done",
                query_len=len(query),
                hits=len(result.hits),
            )

        except Exception as exc:
            logger.warning("vector_retriever_failed", error=f"{type(exc).__name__}: {exc}")

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        """
        Encode a query text with BGE-M3.

        Returns (dense_vector, sparse_vector).
        Reuses _sparse_to_int_keys from embedder.py for consistent token IDs.
        """
        self._load_model()
        output = self._model.encode(
            [query],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense: list[float] = output["dense_vecs"][0].tolist()
        from medgraphia.ingestion.embedder import _sparse_to_int_keys
        sparse: dict[int, float] = _sparse_to_int_keys(output["lexical_weights"][0])
        return dense, sparse

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is not installed. Run: pip install FlagEmbedding"
            ) from exc
        logger.info("vector_retriever_loading_model", model=self._model_name)
        self._model = BGEM3FlagModel(self._model_name, use_fp16=True)
        logger.info("vector_retriever_model_loaded")

    def _get_store(self) -> Any:
        if self._store is None:
            from medgraphia.vector.qdrant_store import QdrantStore
            self._store = QdrantStore()
        return self._store

    def _resolve_collection(self) -> str:
        if self._collection_name:
            return self._collection_name
        from medgraphia.config import get_settings
        return get_settings().qdrant_collection_chunks
