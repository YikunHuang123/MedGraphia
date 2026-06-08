"""
LangGraph-based query router.

Classifies an incoming query into one of 5 QueryType categories and builds
a RetrievalPlan that downstream retrievers consume.

Routing strategy: rule-based keyword matching (fast, deterministic).
This can be swapped for an LLM classifier by replacing _classify_by_rules()
without touching the rest of the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict

from medgraphia.domain import Language, QueryType
from medgraphia.logger import get_logger
from medgraphia.retrieval.query_ner import QueryEntities, QueryNERLinker

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State and plan types
# ---------------------------------------------------------------------------


class RouterState(TypedDict, total=False):
    """LangGraph state dict flowing through the routing graph."""

    query: str
    language: Language
    query_entities: QueryEntities
    query_type: QueryType
    retrieval_plan: RetrievalPlan
    error: str


@dataclass
class RetrievalPlan:
    """
    Routing decision
    """

    query_type: QueryType
    query: str
    language: Language
    query_entities: QueryEntities

    # Which retrieval paths are active
    use_graph: bool = True
    use_vector: bool = True
    use_community: bool = False

    # Per-retriever parameters
    graph_hops: int = 2
    vector_limit: int = 20
    community_limit: int = 5

    @property
    def linked_cuis(self) -> list[str]:
        return self.query_entities.linked_cuis


# ---------------------------------------------------------------------------
# Rule-based keyword classifier
# ---------------------------------------------------------------------------

# Each entry: (QueryType, list-of-regex-patterns, minimum-matches-required)
# Patterns are matched case-insensitively against the query text.
_RULES: list[tuple[QueryType, list[str], int]] = [
    (
        QueryType.DRUG_INTERACTION,
        [
            r"\binteract(ion|s)?\b",
            r"\bcombine[sd]?\b",
            r"\bcontra.?indicat\w*\b",
            r"\bCYP\d+\w*\b",
            # "inhibits" (verb) indicates an inhibition action; "inhibitor" (noun)
            # is just a drug class name and must not trigger this rule alone
            r"\binhibits\b",
            r"\binduce[sd]?\b",
            r"\bpolypharmac\w*\b",
            r"\bco.?administr\w*\b",
            r"\bdrug.{0,10}drug\b",
            r"\b相互作用\b",  # ZH
            r"\bWechselwirkung\w*\b",  # DE
        ],
        1,
    ),
    (
        QueryType.CROSS_CORPUS,
        [
            r"\boverview\b",
            r"\bsummari(se|ze|s?ary)\b",
            r"\ball\b.{0,20}\b(drug|disease|patient|stud(y|ies))\b",
            r"\bglobal\b",
            r"\bcomprehensive\b",
            r"\bacross\b.{0,20}\b(studies|guidelines?|literature)\b",
            r"\bprevalence\b",
            r"\bepidemiolog\w*\b",
            r"\bcommon\b.{0,30}\b(comorbid|condition|disease)\w*\b",
            r"\b综述\b",  # ZH: review/survey
            r"\b概述\b",  # ZH: overview
            r"\bÜbersicht\b",  # DE: overview
        ],
        1,
    ),
    (
        QueryType.LITERATURE_MULTIHOP,
        [
            r"\b(clinical\s+)?trials?\b",
            r"\bstud(y|ies)\b",
            r"\bevidence\b",
            r"\bmeta.?analy\w*\b",
            r"\bsystematic\s+review\b",
            r"\baccording\s+to\b",
            r"\bpublished\b",
            r"\bresearch\b.{0,20}\b(show|suggest|find|report)\w*\b",
            r"\brandomized\b",
            r"\bcohort\b",
            r"\b文献\b",  # ZH: literature
            r"\bStudie\b",  # DE: study
        ],
        1,
    ),
    (
        QueryType.CLINICAL_DECISION,
        [
            r"\bdiagnos(is|e|ed|ing)\b",
            r"\btreat(ment|ing|ed)?\b",
            r"\bdosage?\b",
            r"\bdos(e|ing)\b",
            r"\bprescri(be|bing|bed|ption)\b",
            r"\bmanage(ment|ment\s+of)?\b",
            r"\btherapy\b",
            r"\bsymptom\b",
            r"\bprognosis\b",
            r"\bcomorbid\w*\b",
            r"\b诊断\b",  # ZH: diagnosis
            r"\b治疗\b",  # ZH: treatment
            r"\bBehandlung\b",  # DE: treatment
        ],
        1,
    ),
    # PATIENT_FAQ is checked last — any more-specific clinical type takes priority
    (
        QueryType.PATIENT_FAQ,
        [
            r"\bwhat\s+(is|are)\b",
            r"\bhow\s+(does|do|can|should)\b",
            r"\bcan\s+I\b",
            r"\bshould\s+I\b",
            r"\bis\s+it\s+(safe|ok|okay)\b",
            r"\bis\b.{0,30}\bsafe\b",
            r"\bexplain\b",
            r"\bsimple\b",
            r"\blay\b.{0,10}\bterm\b",
            r"\bpatient\b",
            r"\b什么是\b",  # ZH: what is
            r"\b如何\b",  # ZH: how to
            r"\bwas\s+ist\b",  # DE: what is
        ],
        1,
    ),
]


def _classify_by_rules(query: str, entities: QueryEntities) -> QueryType:
    """
    Return the most-likely QueryType for a query using keyword rules.

    Scoring:
      - Each matching pattern contributes 1 point to its category.
      - A category wins if it has >= its minimum required matches.
      - Tie-break: categories are checked in _RULES order; the first that
        meets its threshold wins (priority: DRUG_INTERACTION > CROSS_CORPUS >
        LITERATURE_MULTIHOP > PATIENT_FAQ > CLINICAL_DECISION).
      - If no category meets its threshold, fall back to CLINICAL_DECISION
        (most common query type in a medical system).
    """
    query_lower = query.lower()
    scores: dict[QueryType, int] = {}

    for qtype, patterns, _min_matches in _RULES:
        count = sum(1 for p in patterns if re.search(p, query_lower, re.IGNORECASE))
        scores[qtype] = count

    # Check rules in priority order
    for qtype, _patterns, min_matches in _RULES:
        if scores[qtype] >= min_matches:
            logger.debug(
                "router_classified",
                query_type=qtype.value,
                score=scores[qtype],
            )
            return qtype

    # Fallback: presence of linked entities → assume clinical decision
    return QueryType.CLINICAL_DECISION


def _plan_from_type(
    qtype: QueryType,
    query: str,
    language: Language,
    entities: QueryEntities,
) -> RetrievalPlan:
    """
    Map a QueryType to a RetrievalPlan with appropriate retriever settings.

    Routing logic:
      DRUG_INTERACTION     → graph (1 hop, focused on entity edges) + vector
      CLINICAL_DECISION    → graph (2 hops) + vector
      LITERATURE_MULTIHOP  → graph (2 hops) + vector (more results)
      CROSS_CORPUS         → vector + community (global summaries), skip graph
      PATIENT_FAQ          → vector only (no graph needed for lay answers)
    """
    if qtype == QueryType.DRUG_INTERACTION:
        plan = RetrievalPlan(
            query_type=qtype,
            query=query,
            language=language,
            query_entities=entities,
            use_graph=True,
            use_vector=True,
            use_community=False,
            graph_hops=1,
            vector_limit=15,
        )
    elif qtype == QueryType.CLINICAL_DECISION:
        plan = RetrievalPlan(
            query_type=qtype,
            query=query,
            language=language,
            query_entities=entities,
            use_graph=True,
            use_vector=True,
            use_community=False,
            graph_hops=2,
            vector_limit=20,
        )
    elif qtype == QueryType.LITERATURE_MULTIHOP:
        plan = RetrievalPlan(
            query_type=qtype,
            query=query,
            language=language,
            query_entities=entities,
            use_graph=True,
            use_vector=True,
            use_community=False,
            graph_hops=2,
            vector_limit=25,
        )
    elif qtype == QueryType.CROSS_CORPUS:
        plan = RetrievalPlan(
            query_type=qtype,
            query=query,
            language=language,
            query_entities=entities,
            use_graph=False,
            use_vector=True,
            use_community=True,
            graph_hops=0,
            vector_limit=15,
            community_limit=8,
        )
    else:  # PATIENT_FAQ
        plan = RetrievalPlan(
            query_type=qtype,
            query=query,
            language=language,
            query_entities=entities,
            use_graph=False,
            use_vector=True,
            use_community=False,
            graph_hops=0,
            vector_limit=10,
        )

    # --- CORE OVERRIDE: If medical entities are detected, ALWAYS use the graph ---
    if entities.has_linked_entities:
        plan.use_graph = True
        # Ensure at least 1-hop retrieval if it was set to 0
        if plan.graph_hops == 0:
            plan.graph_hops = 1

    return plan


# ---------------------------------------------------------------------------
# LangGraph routing graph
# ---------------------------------------------------------------------------


def _build_routing_graph(ner_linker: QueryNERLinker) -> Any:
    """
    Build and compile a LangGraph StateGraph for query routing.

    Node sequence: ner_node → classify_node → plan_node → END
    This allows future insertion of additional nodes (e.g., LLM fallback classification)
    between classify_node and plan_node without changing theinterface.
    """
    try:
        from langgraph.graph import END, StateGraph  # type: ignore[import]
    except ImportError:
        logger.warning("langgraph_not_installed", msg="pip install langgraph")
        return None

    def ner_node(state: RouterState) -> RouterState:
        query = state["query"]
        lang = state.get("language", Language.EN)
        entities = ner_linker.analyze(query, language=lang)
        return {**state, "query_entities": entities}

    def classify_node(state: RouterState) -> RouterState:
        entities: QueryEntities = state.get("query_entities") or QueryEntities(
            query=state["query"], language=state.get("language", Language.EN)
        )
        qtype = _classify_by_rules(state["query"], entities)
        return {**state, "query_type": qtype}

    def plan_node(state: RouterState) -> RouterState:
        qtype: QueryType = state.get("query_type", QueryType.CLINICAL_DECISION)
        entities: QueryEntities = state.get("query_entities") or QueryEntities(
            query=state["query"], language=state.get("language", Language.EN)
        )
        plan = _plan_from_type(
            qtype,
            state["query"],
            state.get("language", Language.EN),
            entities,
        )
        return {**state, "retrieval_plan": plan}

    graph = StateGraph(RouterState)
    graph.add_node("ner", ner_node)
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)

    graph.set_entry_point("ner")
    graph.add_edge("ner", "classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public router class
# ---------------------------------------------------------------------------


class QueryRouter:
    """
    Stateless query router.  Wraps the LangGraph routing graph and provides
    a simple synchronous interface for callers.

    Usage::

        router = QueryRouter.from_settings()
        plan = router.route("What is the interaction between metformin and alcohol?")
        # plan.query_type == QueryType.DRUG_INTERACTION
        # plan.use_graph == True
        # plan.linked_cuis == ["D001234", ...]
    """

    def __init__(self, ner_linker: QueryNERLinker | None = None) -> None:
        self._ner_linker = ner_linker or QueryNERLinker.from_settings()
        self._graph = _build_routing_graph(self._ner_linker)

    @classmethod
    def from_settings(cls) -> QueryRouter:
        return cls()

    def route(self, query: str, language: Language | None = None) -> RetrievalPlan:
        """
        Classify a query and return a RetrievalPlan.
        """
        lang = language or Language.EN

        if self._graph is not None:
            try:
                final_state: RouterState = self._graph.invoke({"query": query, "language": lang})
                plan = final_state.get("retrieval_plan")
                if plan is not None:
                    logger.info(
                        "router_routed",
                        query_type=plan.query_type.value,
                        use_graph=plan.use_graph,
                        use_vector=plan.use_vector,
                        use_community=plan.use_community,
                        linked_cuis=len(plan.linked_cuis),
                    )
                    return plan
            except Exception as exc:
                logger.warning("router_graph_failed", error=str(exc))

        # Fallback path (no LangGraph or graph invocation failed)
        entities = self._ner_linker.analyze(query, language=lang)
        qtype = _classify_by_rules(query, entities)
        return _plan_from_type(qtype, query, lang, entities)

    async def route_async(self, query: str, language: Language | None = None) -> RetrievalPlan:
        """
        Async version of route() with Redis-backed NER result caching.

        Cache hit  — skips BERT/SapBERT inference entirely; runs only the
                     fast rule-based classifier on the cached QueryEntities.
                     Typical latency: ~3 ms (Redis) vs 300–500 ms (full NER).

        Cache miss — runs the synchronous pipeline inside asyncio.to_thread()
                     so the event loop is never blocked, then persists the
                     NER result for subsequent identical queries.

        Redis unavailable — silently falls back to the synchronous route()
                            path with no change in behaviour.
        """
        import asyncio

        from medgraphia.cache.ner_cache import ner_cache_get, ner_cache_set

        lang = language or Language.EN

        cached = await ner_cache_get(query, lang)
        if cached is not None:
            qtype = _classify_by_rules(query, cached)
            plan = _plan_from_type(qtype, query, lang, cached)
            logger.info(
                "router_routed_cached",
                query_type=plan.query_type.value,
                linked_cuis=len(plan.linked_cuis),
            )
            return plan

        # Cache miss: run blocking NER pipeline off the event loop
        plan = await asyncio.to_thread(self.route, query, language)

        # Persist NER result so the next identical query skips inference
        await ner_cache_set(query, lang, plan.query_entities)

        return plan
