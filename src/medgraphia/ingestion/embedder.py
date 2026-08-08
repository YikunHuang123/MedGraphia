"""
Embedding layer.

MedicalEmbedder  — BGE-M3 for text chunks; batch dense + sparse output.
EntityEmbedder   — SapBERT for entity nodes read from Neo4j.

Typical usage (chunk embedding)::

    embedder = MedicalEmbedder.from_settings()
    embedded_chunks = embedder.embed_chunks(chunks)  # sets .embedding + .sparse_embedding
    store = QdrantStore()
    await store.init_collection(cfg.qdrant_collection_chunks, vector_size=embedder.dense_dim)
    await store.upsert_chunks(cfg.qdrant_collection_chunks, embedded_chunks)

Typical usage (entity embedding)::

    entity_embedder = EntityEmbedder.from_settings()
    count = await entity_embedder.embed_and_store()
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from tqdm import tqdm

from medgraphia.domain import Chunk
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps — graceful degradation
# ---------------------------------------------------------------------------

try:
    from FlagEmbedding import BGEM3FlagModel

    _FLAG_AVAILABLE = True
except ImportError:
    _FLAG_AVAILABLE = False
    logger.warning(
        "flag_embedding_not_installed",
        msg="pip install FlagEmbedding for BGE-M3 chunk embedding",
    )

try:
    from sentence_transformers import SentenceTransformer as _ST  # type: ignore[import]

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence_transformers_not_installed",
        msg="pip install sentence-transformers for SapBERT entity embedding",
    )

_BGE_M3_DIM: int = 1024  # BGE-M3 dense output dimension
_SAPBERT_DIM: int = 768  # SapBERT dense output dimension


def _detect_device() -> str:
    """cuda > mps > cpu — explicit so we don't depend on a given library's own default."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# BGE-M3 chunk embedder
# ---------------------------------------------------------------------------


class MedicalEmbedder:
    """
    BGE-M3 embedder for text chunks.

    Per-chunk output:
      dense   — 1024-dim float32 list  → stored in Chunk.embedding
      sparse  — {token_id: weight} SPLADE-style dict → stored in Chunk.sparse_embedding
      colbert — (optional) token-level multi-vectors (not stored in domain model)

    The sparse representation uses abs(hash(token)) % 2**31 to map string tokens to
    integer IDs required by Qdrant SparseVector.  The same mapping is applied at query
    time through embed_texts(), so index↔query consistency is guaranteed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        use_colbert: bool = False,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._use_colbert = use_colbert
        self._device = device or _detect_device()
        self._model: Any = None  # lazy-loaded on first encode call

    @classmethod
    def from_settings(cls) -> MedicalEmbedder:
        """Build embedder from global Settings (uses embedding_batch_size)."""
        from medgraphia.config import get_settings

        cfg = get_settings()
        return cls(batch_size=cfg.embedding_batch_size)

    @property
    def dense_dim(self) -> int:
        """Output dimension of dense vectors — used when creating the Qdrant collection."""
        return _BGE_M3_DIM

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not _FLAG_AVAILABLE:
            raise RuntimeError("FlagEmbedding is not installed. Run: pip install FlagEmbedding")
        logger.info("embedder_loading", model=self._model_name)
        kwargs: dict[str, Any] = {"use_fp16": True}
        if self._device:
            kwargs["device"] = self._device
        self._model = BGEM3FlagModel(self._model_name, **kwargs)
        logger.info("embedder_loaded", model=self._model_name)

    async def embed_texts(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """
        Encode a list of texts in mini-batches.

        Returns:
            dense_vecs  — list of 1024-dim float lists (one per text)
            sparse_vecs — list of {token_id: weight} dicts (one per text)
        """
        self._load_model()

        dense_vecs: list[list[float]] = []
        sparse_vecs: list[dict[int, float]] = []

        with tqdm(total=len(texts), desc="Embedding chunks", unit="chk", leave=False) as pbar:
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                output = await asyncio.to_thread(
                    self._model.encode,
                    batch,
                    batch_size=len(batch),
                    max_length=512,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=self._use_colbert,
                )
                for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"]):
                    dense_vecs.append(dense.tolist())
                    sparse_vecs.append(_sparse_to_int_keys(sparse))
                pbar.update(len(batch))

        return dense_vecs, sparse_vecs

    async def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed all chunks in batch.
        Returns new Chunk objects with .embedding and .sparse_embedding set.
        Raises RuntimeError or Exception on failure.
        """
        if not chunks:
            return chunks

        # Do not catch exceptions here; let them bubble up to the pipeline
        # so the user knows WHY it failed (e.g., OOM or missing dependency).
        dense_vecs, sparse_vecs = await self.embed_texts([c.text for c in chunks])

        result = [
            chunk.model_copy(update={"embedding": d, "sparse_embedding": s})
            for chunk, d, s in zip(chunks, dense_vecs, sparse_vecs)
        ]
        logger.info("embed_chunks_done", count=len(result))
        return result


# ---------------------------------------------------------------------------
# SapBERT entity embedder
# ---------------------------------------------------------------------------


class EntityEmbedder:
    """
    SapBERT entity embedder.

    Reads all entity nodes (any EntityType) from Neo4j,
    encodes their canonical labels with SapBERT, and upserts dense vectors into the
    'medgraphia_entities' Qdrant collection for entity-level similarity search.

    Collection schema: single named vector 'dense' (768-dim cosine), no sparse vectors.
    Point IDs are deterministic UUIDs derived from the entity CUI so that repeated
    runs are idempotent (upsert semantics).
    """

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device or _detect_device()
        self._model: Any = None

    @classmethod
    def from_settings(cls) -> EntityEmbedder:
        """Build entity embedder from global Settings."""
        from medgraphia.config import get_settings

        cfg = get_settings()
        return cls(
            model_name=cfg.el_sapbert_model,
            batch_size=cfg.embedding_batch_size,
        )

    @property
    def dense_dim(self) -> int:
        return _SAPBERT_DIM

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not _ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            )
        logger.info("entity_embedder_loading", model=self._model_name, device=self._device)
        self._model = _ST(self._model_name, device=self._device)
        logger.info("entity_embedder_loaded")

    async def embed_and_store(
        self,
        collection_name: str | None = None,
    ) -> int:
        """
        Full pipeline: fetch entities from Neo4j → embed with SapBERT → write to Qdrant.

        Returns the number of entities successfully written.
        Returns 0 if Neo4j or Qdrant is unavailable.
        """
        from medgraphia.config import get_settings
        from medgraphia.graph.queries import get_all_entities
        from medgraphia.vector.qdrant_store import QdrantStore

        cfg = get_settings()
        collection = collection_name or cfg.qdrant_collection_entities

        entities = await get_all_entities()
        if not entities:
            logger.warning(
                "entity_embed_none", msg="No entity nodes found in Neo4j — nothing to embed"
            )
            return 0

        store = QdrantStore()
        try:
            await store.init_collection(
                collection_name=collection,
                vector_size=_SAPBERT_DIM,
                sparse=False,
            )
        except Exception as exc:
            logger.error("entity_embed_init_collection_failed", error=str(exc))
            return 0

        self._load_model()

        total_entities = len(entities)
        written_count = 0

        with tqdm(total=total_entities, desc="Embedding entities", unit="ent") as pbar:
            for start in range(0, total_entities, self._batch_size):
                batch = entities[start : start + self._batch_size]
                labels = [e["label"] for e in batch]

                embeddings = await asyncio.to_thread(
                    self._model.encode,
                    labels, 
                    normalize_embeddings=True, 
                    show_progress_bar=False
                )

                entity_points = [
                    {
                        "cui": e["cui"],
                        "label": e["label"],
                        "entity_type": e["entity_type"],
                        "lang_zh": e.get("lang_zh", ""),
                        "lang_de": e.get("lang_de", ""),
                        "embedding": emb.tolist(),
                    }
                    for e, emb in zip(batch, embeddings)
                ]

                written = await store.upsert_entities(collection, entity_points)
                written_count += written
                pbar.update(len(batch))

        logger.info("entity_embed_done", collection=collection, count=written_count)
        return written_count


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _sparse_to_int_keys(lexical_weights: dict) -> dict[int, float]:
    """
    Convert BGE-M3 lexical_weights to Qdrant-compatible {int token_id: weight}.

    The same function must be applied at query time so index↔query IDs match.
    """
    result: dict[int, float] = {}
    for token, weight in lexical_weights.items():
        w = float(weight)
        if w <= 0:
            continue
        token_id = int(token)
        if token_id not in result or result[token_id] < w:
            result[token_id] = w
    return result


def entity_to_qdrant_id(cui: str) -> str:
    """
    Deterministic UUID for a CUI — used as the Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, cui))
