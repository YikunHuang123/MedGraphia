"""
Retrieval orchestration pipeline.

This module connects all retrieval components into a single, cohesive workflow:
  1. Routing (NER + Intent Classification)
  2. Concurrent multi-path retrieval (Graph, Vector, Community)
  3. Reciprocal Rank Fusion (RRF)
  4. Cross-encoder Reranking
"""

from __future__ import annotations

import asyncio
from typing import Any

from medgraphia.domain import Language
from medgraphia.domain.chat import Message
from medgraphia.logger import get_logger
from medgraphia.retrieval.community_retriever import CommunityRetriever
from medgraphia.retrieval.fusion import RRFFusion
from medgraphia.retrieval.graph_retriever import GraphRetriever
from medgraphia.retrieval.query_translator import QueryTranslator, TranslatedQuery
from medgraphia.retrieval.reranker import RerankedResult, Reranker
from medgraphia.retrieval.rewriter import QueryRewriter
from medgraphia.retrieval.router import QueryRouter, RetrievalPlan
from medgraphia.retrieval.vector_retriever import VectorRetriever

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    End-to-end orchestrator for the GraphRAG retrieval phase.

    Usage::

        pipeline = RetrievalPipeline.from_settings()
        result = await pipeline.execute("Is it safe to take aspirin with warfarin?")
        for item in result.items:
            print(f"[{item.rrf_score:.2f}] {item.text}")
    """

    def __init__(
        self,
        router: QueryRouter | None = None,
        rewriter: QueryRewriter | None = None,
        graph_retriever: GraphRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        community_retriever: CommunityRetriever | None = None,
        fusion: RRFFusion | None = None,
        reranker: Reranker | None = None,
        query_translator: QueryTranslator | None = None,
    ) -> None:
        self.router = router or QueryRouter.from_settings()
        self.rewriter = rewriter or QueryRewriter.from_settings()
        self.graph_retriever = graph_retriever or GraphRetriever.from_settings()
        self.vector_retriever = vector_retriever or VectorRetriever.from_settings()
        self.community_retriever = community_retriever or CommunityRetriever.from_settings()
        self.fusion = fusion or RRFFusion.from_settings()
        self.reranker = reranker or Reranker.from_settings()
        self.query_translator = query_translator or QueryTranslator.from_settings()

    @classmethod
    def from_settings(cls) -> RetrievalPipeline:
        """Instantiate the full pipeline using global configuration."""
        return cls()

    async def execute(
        self,
        query: str,
        history: list[Message] | None = None,
        language: Language | None = None,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> RerankedResult:
        """
        Execute the full GraphRAG retrieval pipeline.

        Args:
            query:    The user's raw question.
            history:  Optional conversation history for context-aware retrieval.
            language: Optional language override.
            user_id:  Optional user ID for personalized graph retrieval.
            top_k:    Number of final context passages to return.

        Returns:
            RerankedResult containing the optimal context items.
        """
        logger.info(
            "retrieval_pipeline_started",
            query_len=len(query),
            has_history=bool(history),
            user_id=user_id,
        )

        # ---------------------------------------------------------
        # Step 0: Contextual Query Rewriting + Complexity Assessment
        # Runs unconditionally — complexity scoring depends on the question
        # itself, not history.  Empty history is valid (first message).
        # ---------------------------------------------------------
        from medgraphia.generation.llm_router import ModelTier

        complexity_tier: ModelTier | None = None
        search_query, complexity_tier = await self.rewriter.rewrite(
            query=query, history=history or [], language=language or Language.EN
        )
        if history:
            logger.info("retrieval_using_rewritten_query", rewritten=search_query)
        logger.info("complexity_tier_assessed", tier=complexity_tier.value)

        # ---------------------------------------------------------
        # Step 0.5: Multilingual query expansion
        # ---------------------------------------------------------
        # Translate the search query into all other supported languages so that
        # the vector retriever can run per-language searches.  This removes the
        # lexical-bias disadvantage that hybrid (dense+sparse) search introduces
        # for cross-language queries (e.g. Chinese query vs. German corpus chunk).
        queries_by_language: dict[Language, str] | None = None
        _known = {Language.EN, Language.ZH, Language.DE}
        src_lang = language if language in _known else None
        # Expand whenever the source language is a known corpus language.
        # The lexical-bias problem affects ALL source languages — an English
        # query "kidney failure" misses German "Nierenversagen" for the same
        # sparse-token-mismatch reason as a Chinese query would.
        if src_lang is not None:
            try:
                from medgraphia.config import get_settings

                cfg = get_settings()
                if cfg.multilingual_retrieval_enabled:
                    translated: TranslatedQuery = await self.query_translator.translate(
                        query=search_query,
                        source_language=src_lang,
                    )
                    queries_by_language = translated.all_queries()
                    logger.info(
                        "multilingual_queries_ready",
                        languages=[lg.value for lg in queries_by_language],
                    )
            except Exception as exc:
                logger.warning("multilingual_expansion_skipped", error=str(exc))

        # ---------------------------------------------------------
        # Step 1: Route & Plan (using the rewritten query)
        # ---------------------------------------------------------
        plan: RetrievalPlan = await self.router.route_async(search_query, language=language)

        # ---------------------------------------------------------
        # Step 2: Concurrent Retrieval
        # ---------------------------------------------------------
        tasks: list[asyncio.Task[Any]] = []
        task_names: list[str] = []

        # Setup Graph task
        if plan.use_graph and plan.linked_cuis:
            tasks.append(
                asyncio.create_task(
                    self.graph_retriever.retrieve(
                        cuis=plan.linked_cuis,
                        hops=plan.graph_hops,
                        user_id=user_id,
                    )
                )
            )
            task_names.append("graph")
        else:
            task_names.append("skip_graph")

        # Setup Vector task
        if plan.use_vector:
            if queries_by_language is not None:
                from medgraphia.config import get_settings

                _cfg = get_settings()
                tasks.append(
                    asyncio.create_task(
                        self.vector_retriever.retrieve_multilingual(
                            queries_by_language=queries_by_language,
                            per_lang_quota=_cfg.multilingual_per_lang_quota,
                            total_limit=plan.vector_limit,
                        )
                    )
                )
            else:
                tasks.append(
                    asyncio.create_task(
                        self.vector_retriever.retrieve(
                            query=search_query,
                            limit=plan.vector_limit,
                        )
                    )
                )
            task_names.append("vector")
        else:
            task_names.append("skip_vector")

        # Setup Community task
        if plan.use_community:
            tasks.append(
                asyncio.create_task(
                    self.community_retriever.retrieve(
                        query=search_query,
                        limit=plan.community_limit,
                    )
                )
            )
            task_names.append("community")
        else:
            task_names.append("skip_community")

        # Execute all scheduled retrievers in parallel
        # We filter out "skip_*" markers when awaiting
        active_tasks = [t for t in tasks]
        results = await asyncio.gather(*active_tasks, return_exceptions=True)

        # Map results back to their respective sources safely
        graph_result = None
        vector_result = None
        community_result = None

        result_idx = 0
        for name in task_names:
            if name.startswith("skip_"):
                continue

            res = results[result_idx]
            result_idx += 1

            if isinstance(res, Exception):
                logger.error("retriever_task_failed", source=name, error=str(res))
                continue

            if name == "graph":
                graph_result = res
            elif name == "vector":
                vector_result = res
            elif name == "community":
                community_result = res

        # ---------------------------------------------------------
        # Step 3: Reciprocal Rank Fusion (RRF)
        # ---------------------------------------------------------
        fusion_result = self.fusion.fuse(
            query=search_query,
            graph_result=graph_result,
            vector_result=vector_result,
            community_result=community_result,
        )

        # ---------------------------------------------------------
        # Step 4: Cross-Encoder Reranking
        # ---------------------------------------------------------
        # Optimization: Only rerank the top-N candidates from the fusion result.
        # Reranking 60+ items is slow; top 20 is typically enough for high recall.
        rerank_candidates = fusion_result.top(20)

        final_result = self.reranker.rerank(
            query=search_query,
            fusion_result=rerank_candidates,
            top_k=top_k,
        )

        # Inject additional metadata into the result for downstream use
        final_result.query_type = plan.query_type
        final_result.linked_cuis = plan.linked_cuis
        final_result.unlinked_mentions = plan.query_entities.unlinked_mentions
        final_result.complexity_tier = complexity_tier

        logger.info(
            "retrieval_pipeline_completed",
            query_type=plan.query_type.value,
            final_items=len(final_result.items),
        )
        return final_result
