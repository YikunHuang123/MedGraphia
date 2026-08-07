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
from medgraphia.retrieval.memory_retriever import MemoryRetriever
from medgraphia.retrieval.query_translator import QueryTranslator, TranslatedQuery
from medgraphia.retrieval.reranker import RerankedResult, Reranker
from medgraphia.retrieval.rewriter import QueryRewriter
from medgraphia.retrieval.router import QueryRouter, RetrievalPlan
from medgraphia.retrieval.vector_retriever import VectorRetriever

logger = get_logger(__name__)


def _dedupe_overlapping_entities(entities: list[Any]) -> list[Any]:
    """Collapse overlapping-span entities into the highest-confidence one per cluster."""
    ranked = sorted(entities, key=lambda e: -e.confidence)
    kept: list[Any] = []
    for e in ranked:
        if e.start_char is None or e.end_char is None:
            kept.append(e)
            continue
        overlaps = any(
            k.start_char is not None and not (e.end_char <= k.start_char or e.start_char >= k.end_char)
            for k in kept
        )
        if not overlaps:
            kept.append(e)
    return kept


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
        memory_retriever: MemoryRetriever | None = None,
    ) -> None:
        self.router = router or QueryRouter.from_settings()
        self.rewriter = rewriter or QueryRewriter.from_settings()
        self.graph_retriever = graph_retriever or GraphRetriever.from_settings()
        self.vector_retriever = vector_retriever or VectorRetriever.from_settings()
        self.community_retriever = community_retriever or CommunityRetriever.from_settings()
        self.fusion = fusion or RRFFusion.from_settings()
        self.reranker = reranker or Reranker.from_settings()
        self.query_translator = query_translator or QueryTranslator.from_settings()
        self.memory_retriever = memory_retriever or MemoryRetriever.from_settings()

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
        top_k: int = 10,
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
        # Step 1: Route & Plan (using the rewritten query or its English translation)
        # We prefer English for NER/Routing because SapBERT has 99% accuracy on 
        # English entities but can hallucinate on Chinese phonetic transliterations (e.g. Ibuprofen -> Baclofen).
        # ---------------------------------------------------------
        routing_query = search_query
        routing_lang = language or Language.EN

        if queries_by_language and Language.EN in queries_by_language:
            routing_query = queries_by_language[Language.EN]
            routing_lang = Language.EN

        plan: RetrievalPlan = await self.router.route_async(routing_query, language=routing_lang)

        # No medical keyword matched and no entity was linked (e.g. a greeting) —
        # there is nothing to retrieve. Skip graph/vector/community/reranking
        # entirely so downstream generation doesn't hand chitchat to the
        # GEPA-tuned clinical generator, which expects grounded context.
        if plan.is_chitchat:
            logger.info("retrieval_pipeline_skipped_chitchat", query_type=plan.query_type.value)
            return RerankedResult(
                query=search_query,
                query_type=plan.query_type,
                complexity_tier=complexity_tier,
                is_chitchat=True,
            )

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

        # Setup per-user QA memory task — independent of the plan's use_graph/
        # use_vector flags since it reads a private subgraph, not the corpus
        if user_id and plan.linked_cuis:
            tasks.append(
                asyncio.create_task(
                    self.memory_retriever.retrieve(user_id=user_id, cuis=plan.linked_cuis)
                )
            )
            task_names.append("memory")
        else:
            task_names.append("skip_memory")

        # Execute all scheduled retrievers in parallel
        # We filter out "skip_*" markers when awaiting
        active_tasks = [t for t in tasks]
        results = await asyncio.gather(*active_tasks, return_exceptions=True)

        # Map results back to their respective sources safely
        graph_result = None
        vector_result = None
        community_result = None
        memory_result = None

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
            elif name == "memory":
                memory_result = res

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
        # Step 3.5: Semantic Deduplication
        # Filter out redundant chunks (near-identical text) to ensure
        # the reranker has a diverse candidate pool.
        # ---------------------------------------------------------
        import difflib
        
        unique_items = []
        seen_texts = []

        # Sort by RRF score to prioritize better-ranked versions
        fusion_items = sorted(fusion_result.items, key=lambda x: x.rrf_score, reverse=True)

        for item in fusion_items:
            # Simple deduplication: compare normalized first 300 chars
            text_prefix = item.text[:300].strip().lower()
            is_duplicate = False
            
            for seen in seen_texts:
                # Use SequenceMatcher for fuzzy comparison (90% similarity threshold)
                similarity = difflib.SequenceMatcher(None, text_prefix, seen).ratio()
                if similarity > 0.90:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_items.append(item)
                seen_texts.append(text_prefix)
            else:
                logger.debug("retrieval_deduplicated_item", source=item.source.value, item_id=item.item_id)

        # Update fusion_result with unique items
        fusion_result.items = unique_items

        # ---------------------------------------------------------
        # Step 4: Cross-Encoder Reranking
        # ---------------------------------------------------------
        # Optimization: Only rerank the top-N candidates from the fusion result.
        # Use top(25) to ensure the cross-encoder has enough candidates when top_k=10.
        rerank_candidates = fusion_result.top(25)

        final_result = await self.reranker.rerank(
            query=search_query,
            fusion_result=rerank_candidates,
            top_k=top_k,
        )

        # Step 4.5: single-entity live gap fill, only when local retrieval found
        # nothing (no_evidence) and the query names exactly one distinct topic.
        # Two+ distinct entities defer to the two-entity relationship-gap path in
        # generation/agentic_completion.py instead.
        from medgraphia.config import get_settings as _get_settings

        distinct_entities = _dedupe_overlapping_entities(plan.query_entities.entities)

        if (
            final_result.no_evidence
            and _get_settings().single_entity_gap_completion_enabled
            and len(distinct_entities) == 1
        ):
            entity_label = distinct_entities[0].label
            if entity_label:
                from medgraphia.retrieval.fusion import FusionResult, chunk_to_fused_item
                from medgraphia.retrieval.query_time_completion import complete_single_entity_gap

                cfg = _get_settings()
                _, new_chunks = await complete_single_entity_gap(
                    entity_label, pubmed_limit=cfg.gap_completion_pubmed_limit
                )
                if new_chunks:
                    retry_items = [chunk_to_fused_item(c) for c in new_chunks]
                    final_result = await self.reranker.rerank(
                        query=search_query,
                        fusion_result=FusionResult(items=retry_items, query=search_query),
                        top_k=top_k,
                    )
                    logger.info(
                        "single_entity_gap_completion_applied",
                        entity=entity_label,
                        new_chunks=len(new_chunks),
                        no_evidence_after_retry=final_result.no_evidence,
                    )

        # Inject additional metadata into the result for downstream use
        final_result.query_type = plan.query_type
        final_result.linked_cuis = plan.linked_cuis
        final_result.unlinked_mentions = plan.query_entities.unlinked_mentions
        final_result.entity_labels = {
            e.cui: e.label for e in plan.query_entities.entities if not e.cui.startswith("MENTION:")
        }
        final_result.complexity_tier = complexity_tier
        final_result.qa_memories = memory_result.memories if memory_result else []

        logger.info(
            "retrieval_pipeline_completed",
            query_type=plan.query_type.value,
            final_items=len(final_result.items),
            qa_memories=len(final_result.qa_memories),
        )
        return final_result
