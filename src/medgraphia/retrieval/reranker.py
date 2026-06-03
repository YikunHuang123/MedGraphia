"""
Cross-encoder reranker.

Re-ranks a FusionResult (top-N items) using bge-reranker-v2-m3

Model:  BAAI/bge-reranker-v2-m3
  - Supports 100+ languages (EN / ZH / DE all covered)
  - 568M parameters (manageable on CPU for small N)
  - FlagEmbedding integration: FlagReranker class
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medgraphia.logger import get_logger
from medgraphia.retrieval.fusion import FusedItem, FusionResult

logger = get_logger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


from medgraphia.domain.base import QueryType

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class RerankedResult:
    """
    Final result after cross-encoder reranking.

    Attributes:
        items         — Re-ranked FusedItem list (best first).
        query         — Original query text.
        reranked      — True if reranking actually ran; False if it was skipped.
        query_type    — The intent classified by the router (useful for generation).
    """
    items: list[FusedItem] = field(default_factory=list)
    query: str = ""
    reranked: bool = False
    query_type: QueryType = QueryType.PATIENT_FAQ
    linked_cuis: list[str] = field(default_factory=list)
    unlinked_mentions: list[str] = field(default_factory=list)

    def texts(self) -> list[str]:
        return [it.text for it in self.items]

    def as_context_lines(self) -> list[str]:
        return [it.as_context_str() for it in self.items]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    Multilingual cross-encoder reranker (bge-reranker-v2-m3).

    Usage::

        reranker = Reranker.from_settings()
        result = reranker.rerank(
            query="What is the interaction between metformin and alcohol?",
            fusion_result=fusion_result.top(20),
            top_k=5,
        )
        for item in result.items:
            print(item.text)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        use_fp16: bool = True,
    ) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model: Any = None  # lazy-loaded
        self._backend: str | None = None  # "flag" | "sentence_transformers"

    @classmethod
    def from_settings(cls) -> "Reranker":
        return cls()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        fusion_result: FusionResult,
        top_k: int = 5,
    ) -> RerankedResult:
        """
        Re-rank up to len(fusion_result.items) candidates and return top_k.

        Args:
            query:         Raw query text.
            fusion_result: FusionResult from RRFFusion (ideally .top(20)).
            top_k:         Number of items to return.

        Returns:
            RerankedResult with at most top_k items sorted by cross-encoder score.
        """
        items = fusion_result.items
        if not items:
            return RerankedResult(query=query)

        # Try to load the cross-encoder model
        try:
            self._load_model()
        except Exception as exc:
            logger.warning("reranker_load_failed", error=str(exc))
            return RerankedResult(
                items=items[:top_k],
                query=query,
                reranked=False,
            )

        # Build (query, text) pairs
        pairs = [(query, it.text) for it in items]

        try:
            scores = self._score_pairs(pairs)
        except Exception as exc:
            logger.warning("reranker_score_failed", error=str(exc))
            return RerankedResult(
                items=items[:top_k],
                query=query,
                reranked=False,
            )

        # Sort by cross-encoder score descending
        ranked = sorted(
            zip(scores, items),
            key=lambda x: -x[0],
        )[:top_k]

        reranked_items = []
        for score, item in ranked:
            item.metadata["reranker_score"] = float(score)
            reranked_items.append(item)

        logger.info(
            "reranker_done",
            input=len(items),
            output=len(reranked_items),
            top_score=f"{ranked[0][0]:.4f}" if ranked else "n/a",
        )
        return RerankedResult(
            items=reranked_items,
            query=query,
            reranked=True,
        )

    # ------------------------------------------------------------------
    # Internal: model loading and scoring
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Try FlagEmbedding first, fall back to sentence-transformers."""
        if self._model is not None:
            return

        # ── Determine best available device ──────────────────────────────────
        import torch
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        
        logger.info("reranker_device_selected", device=device)

        # Attempt 1: FlagEmbedding FlagReranker (preferred)
        try:
            from FlagEmbedding import FlagReranker  # type: ignore[import]
            logger.info("reranker_loading", model=self._model_name, backend="FlagEmbedding", device=device)
            # FlagReranker takes 'devices' as a string or list
            self._model = FlagReranker(
                self._model_name, 
                use_fp16=self._use_fp16, 
                devices=device
            )
            self._backend = "flag"
            logger.info("reranker_loaded", backend="FlagEmbedding")
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("reranker_flag_load_failed", error=str(exc))

        # Attempt 2: sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            logger.info("reranker_loading", model=self._model_name, backend="sentence-transformers", device=device)
            self._model = CrossEncoder(
                self._model_name,
                max_length=512,
                device=device,
            )
            self._backend = "sentence_transformers"
            logger.info("reranker_loaded", backend="sentence-transformers")
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("reranker_st_load_failed", error=str(exc))

        raise RuntimeError(
            "No reranker backend available. Install one of:\n"
            "  pip install FlagEmbedding\n"
            "  pip install sentence-transformers"
        )

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Compute relevance scores for (query, passage) pairs.

        FlagReranker.compute_score() and CrossEncoder.predict() have
        different signatures and return different types; this method normalises
        both to a list[float].
        """
        if self._backend == "flag":
            # FlagReranker.compute_score accepts list of [query, passage] pairs
            flag_pairs = [[q, p] for q, p in pairs]
            raw = self._model.compute_score(flag_pairs, normalize=True)
            # May return a numpy array or a list of floats
            if hasattr(raw, "tolist"):
                return raw.tolist()
            return [float(s) for s in raw]

        elif self._backend == "sentence_transformers":
            # CrossEncoder.predict returns numpy array of shape (N,)
            raw = self._model.predict(pairs, show_progress_bar=False)
            if hasattr(raw, "tolist"):
                return raw.tolist()
            return [float(s) for s in raw]

        return [0.0] * len(pairs)
