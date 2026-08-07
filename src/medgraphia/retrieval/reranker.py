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

from medgraphia.config import settings
from medgraphia.domain.base import QueryType
from medgraphia.generation.llm_router import ModelTier
from medgraphia.logger import get_logger
from medgraphia.retrieval.fusion import FusedItem, FusionResult

logger = get_logger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class RerankedResult:
    """
    Final result after cross-encoder reranking.

    Attributes:
        items            — Re-ranked FusedItem list (best first).
        query            — Original query text.
        reranked         — True if reranking actually ran; False if it was skipped.
        query_type       — The intent classified by the router (useful for generation).
        complexity_tier  — DSPy-assessed routing tier from the rewriter; None means
                           the router will fall back to the static QueryType→tier table.
    """

    items: list[FusedItem] = field(default_factory=list)
    query: str = ""
    reranked: bool = False
    query_type: QueryType = QueryType.PATIENT_FAQ
    linked_cuis: list[str] = field(default_factory=list)
    unlinked_mentions: list[str] = field(default_factory=list)
    entity_labels: dict[str, str] = field(default_factory=dict)  # cui -> label, linked entities only
    complexity_tier: ModelTier | None = None
    is_chitchat: bool = False  # no medical signal detected; caller should skip DSPy generation
    no_evidence: bool = False  # fallback content scored below the noise floor; caller should skip DSPy generation
    qa_memories: list[Any] = field(default_factory=list)  # this user's relevant past QA turns (QAMemory)

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
        threshold: float | None = None,
        fallback_top_n: int | None = None,
        noise_floor: float | None = None,
    ) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._threshold = threshold if threshold is not None else settings.reranker_threshold
        self._fallback_top_n = (
            fallback_top_n if fallback_top_n is not None else settings.reranker_fallback_top_n
        )
        self._noise_floor = noise_floor if noise_floor is not None else settings.reranker_noise_floor
        self._model: Any = None  # lazy-loaded
        self._backend: str | None = None  # "flag" | "sentence_transformers"

    @classmethod
    def from_settings(cls) -> Reranker:
        return cls(
            model_name=settings.reranker_model,
            threshold=settings.reranker_threshold,
            fallback_top_n=settings.reranker_fallback_top_n,
            noise_floor=settings.reranker_noise_floor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
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

        # Build texts list
        documents = [it.text for it in items]

        try:
            from medgraphia.config import get_settings
            import httpx
            cfg = get_settings()
            
            # 1. Base URL override and fallback
            api_url = cfg.reranker_api_url
            if not api_url:
                if cfg.reranker_provider == "siliconflow":
                    api_url = "https://api.siliconflow.com/v1/rerank"
                elif cfg.reranker_provider == "jina":
                    api_url = "https://api.jina.ai/v1/rerank"
                elif cfg.reranker_provider == "cohere":
                    api_url = "https://api.cohere.v1/rerank"
                elif cfg.reranker_provider in ["fireworks", "fireworks_ai"]:
                    api_url = "https://api.fireworks.ai/inference/v1/rerank"
                else:
                    raise ValueError(f"Unknown reranker_provider '{cfg.reranker_provider}'. Please set reranker_api_url explicitly.")

            # 2. API Key override and fallback
            api_key = cfg.reranker_api_key.get_secret_value()
            if not api_key:
                # Dynamically fetch the global key for this provider
                p_name = cfg.reranker_provider.lower()
                if p_name == "fireworks":
                    p_name = "fireworks_ai"
                # For fireworks_ai, the config attribute is fireworks_api_key
                if p_name == "fireworks_ai":
                    global_key_attr = "fireworks_api_key"
                else:
                    global_key_attr = f"{p_name}_api_key"
                secret_obj = getattr(cfg, global_key_attr, None)
                if secret_obj and hasattr(secret_obj, "get_secret_value"):
                    api_key = secret_obj.get_secret_value()
            
            if not api_key:
                raise ValueError(
                    f"No API key configured for reranker provider '{cfg.reranker_provider}'. "
                    "Set either reranker_api_key or the global provider key in config."
                )

            payload = {
                "model": self._model_name,
                "query": query,
                "return_documents": False,
                "top_n": len(documents)
            }
            if cfg.reranker_provider in ["siliconflow", "jina", "cohere", "fireworks", "fireworks_ai"]:
                payload["documents"] = documents
            else:
                payload["texts"] = documents

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
            
            # Reconstruct scores array aligned with original documents order
            # The API returns `results` which might be sorted. Each has an `index`.
            scores = [0.0] * len(documents)
            for res in data.get("results", []):
                idx = res.get("index")
                if idx is not None and 0 <= idx < len(scores):
                    scores[idx] = float(res.get("relevance_score", 0.0))
            
        except Exception as exc:
            logger.warning("reranker_score_failed", error=repr(exc))
            return RerankedResult(
                items=items[:top_k],
                query=query,
                reranked=False,
            )

        # Sort by cross-encoder score descending
        ranked = sorted(
            zip(scores, items),
            key=lambda x: -x[0],
        )

        # Apply threshold filtering
        filtered_ranked = [(score, item) for score, item in ranked if score >= self._threshold][
            :top_k
        ]

        # Threshold filtered out everything — fall back to the top-scoring items so the
        # generator never gets an empty context. Low reranker_score still lets the LLM
        # (or a downstream check) recognize weak evidence and hedge or decline.
        used_fallback = False
        if not filtered_ranked and ranked and self._fallback_top_n > 0:
            filtered_ranked = ranked[: self._fallback_top_n]
            used_fallback = True

        reranked_items = []
        for score, item in filtered_ranked:
            item.metadata["reranker_score"] = float(score)
            reranked_items.append(item)

        # Below the noise floor, fallback content is noise, not weak evidence — skip generation.
        no_evidence = used_fallback and bool(ranked) and ranked[0][0] < self._noise_floor

        logger.info(
            "reranker_done",
            input=len(items),
            output=len(reranked_items),
            top_score=f"{ranked[0][0]:.4f}" if ranked else "n/a",
            threshold=self._threshold,
            used_fallback=used_fallback,
            no_evidence=no_evidence,
        )
        return RerankedResult(
            items=reranked_items,
            query=query,
            reranked=True,
            no_evidence=no_evidence,
        )

    # ------------------------------------------------------------------
    # Internal: model loading and scoring
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        pass
