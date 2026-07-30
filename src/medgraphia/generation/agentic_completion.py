"""
LangGraph-based agentic gap completion.

Runs just before final answer generation. An LLM inspects the question and
the retrieved context and decides whether any entity pair is missing a
documented relationship; if so it calls the query-time completion tool
to fetch and merge evidence, then re-assesses. Capped so a single
request can't loop indefinitely or run away on cost.

The DSPy-compiled answer generator itself is untouched by this — this graph
only augments `context` before the existing generation call runs.
"""

from __future__ import annotations

from typing import Any, TypedDict

from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class CompletionState(TypedDict, total=False):
    """LangGraph state dict flowing through the gap-completion graph."""

    question: str
    context: str
    entity_labels: list[str]  # candidate entities mentioned in the query
    entity_cui_map: dict[str, str]  # label -> cui, only for already-linked entities
    lm: Any  # dspy LM instance to use (same one chosen for final generation)
    max_tool_calls: int
    tool_calls_made: int
    gap_evidence: list[str]
    needs_completion: bool
    entity_a: str
    entity_b: str


# ---------------------------------------------------------------------------
# DSPy decision signature
# ---------------------------------------------------------------------------


def _assess_signature():
    import dspy

    class AssessKnowledgeGap(dspy.Signature):
        """Decide whether the retrieved context is missing a documented relationship
        between two of the medical entities mentioned in the question, that would be
        needed to answer it. Only propose a pair if their relationship is actually
        relevant to the question and is not already covered by the context. If the
        context is sufficient, or no entity pair applies, set needs_completion to false."""

        question: str = dspy.InputField()
        context: str = dspy.InputField(desc="Currently retrieved context, numbered")
        candidate_entities: str = dspy.InputField(desc="Comma-separated entity labels from the query")
        needs_completion: bool = dspy.OutputField()
        entity_a: str = dspy.OutputField(desc="First entity name; empty if needs_completion is false")
        entity_b: str = dspy.OutputField(desc="Second entity name; empty if needs_completion is false")

    return AssessKnowledgeGap


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def _assess_gap_node(state: CompletionState) -> CompletionState:
    import asyncio

    import dspy

    predictor = dspy.Predict(_assess_signature())
    lm = state["lm"]
    entities = ", ".join(state.get("entity_labels", []))

    current_context = state["context"]
    if state.get("gap_evidence"):
        current_context += "\n\n[Previously fetched missing evidence]\n" + "\n".join(state["gap_evidence"])

    def _run():
        with dspy.context(lm=lm):
            return predictor(
                question=state["question"],
                context=current_context,
                candidate_entities=entities,
            )

    if len(state.get("entity_labels", [])) < 2:
        return {**state, "needs_completion": False}

    try:
        pred = await asyncio.to_thread(_run)
    except Exception as exc:
        logger.warning("assess_gap_failed", error=str(exc))
        return {**state, "needs_completion": False}

    needs = bool(pred.needs_completion) and bool(pred.entity_a) and bool(pred.entity_b)
    return {
        **state,
        "needs_completion": needs,
        "entity_a": pred.entity_a,
        "entity_b": pred.entity_b,
    }


async def _execute_tool_node(state: CompletionState) -> CompletionState:
    from medgraphia.config import get_settings

    entity_a, entity_b = state["entity_a"], state["entity_b"]
    tool_calls_made = state.get("tool_calls_made", 0) + 1

    # Idempotency guard: if both entities are already linked CUIs and a real
    # path already connects them, don't re-fetch — just record that.
    cui_map = state.get("entity_cui_map", {})
    cui_a, cui_b = cui_map.get(entity_a), cui_map.get(entity_b)
    if cui_a and cui_b:
        from medgraphia.retrieval.graph_retriever import GraphRetriever

        already_connected = await GraphRetriever.from_settings().check_path_exists(cui_a, cui_b)
        if already_connected:
            evidence = f"{entity_a} and {entity_b} are already connected in the knowledge graph."
            gap_evidence = state.get("gap_evidence", []) + [evidence]
            return {**state, "gap_evidence": gap_evidence, "tool_calls_made": tool_calls_made, "needs_completion": False}

    from medgraphia.retrieval.query_time_completion import complete_gap

    cfg = get_settings()
    evidence = await complete_gap(entity_a, entity_b, pubmed_limit=cfg.gap_completion_pubmed_limit)
    gap_evidence = state.get("gap_evidence", []) + [evidence]
    return {**state, "gap_evidence": gap_evidence, "tool_calls_made": tool_calls_made, "needs_completion": False}


def _route_after_assess(state: CompletionState) -> str:
    if state.get("needs_completion") and state.get("tool_calls_made", 0) < state.get("max_tool_calls", 2):
        return "execute_tool"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_completion_graph() -> Any:
    """
    Build and compile the gap-completion LangGraph.

    Node sequence: assess -> (execute_tool -> assess)* -> END
    Loops back to assess after each tool call so the model can request a
    second pair, bounded by max_tool_calls.
    """
    try:
        from langgraph.graph import END, StateGraph  # type: ignore[import]
    except ImportError:
        logger.warning("langgraph_not_installed", msg="pip install langgraph")
        return None

    graph = StateGraph(CompletionState)
    graph.add_node("assess", _assess_gap_node)
    graph.add_node("execute_tool", _execute_tool_node)

    graph.set_entry_point("assess")
    graph.add_conditional_edges("assess", _route_after_assess, {"execute_tool": "execute_tool", "end": END})
    graph.add_edge("execute_tool", "assess")

    return graph.compile()


_compiled_graph: Any = None


async def run_gap_completion(
    question: str,
    context: str,
    entity_labels: list[str],
    entity_cui_map: dict[str, str],
    lm: Any,
    max_tool_calls: int = 2,
) -> list[str]:
    """
    Entry point used by the generation pipeline. Returns a list of evidence
    strings (empty if nothing was missing or LangGraph isn't installed).
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_completion_graph()
    if _compiled_graph is None:
        return []

    initial_state: CompletionState = {
        "question": question,
        "context": context,
        "entity_labels": entity_labels,
        "entity_cui_map": entity_cui_map,
        "lm": lm,
        "max_tool_calls": max_tool_calls,
        "tool_calls_made": 0,
        "gap_evidence": [],
    }

    try:
        final_state = await _compiled_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.warning("gap_completion_graph_failed", error=str(exc))
        return []

    return final_state.get("gap_evidence", [])
