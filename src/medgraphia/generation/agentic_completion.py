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
    gap_chunks: list[Any]
    needs_completion: bool
    entity_a: str
    entity_b: str
    language: str
    message_to_user: str


# ---------------------------------------------------------------------------
# DSPy decision signature
# ---------------------------------------------------------------------------


def _assess_signature():
    import dspy

    class AssessKnowledgeGap(dspy.Signature):
        """Decide whether the retrieved context is missing a documented relationship
        between two of the medical entities mentioned in the question, or if vital information
        is missing about a single medical entity. Only propose a search if the information is actually
        relevant to the question and is not already covered by the context. If the
        context is sufficient, or no entity applies, set needs_completion to false."""

        question: str = dspy.InputField()
        context: str = dspy.InputField(desc="Currently retrieved context, numbered")
        candidate_entities: str = dspy.InputField(desc="Comma-separated entity labels from the query")
        language: str = dspy.InputField(desc="The target language (e.g. 'zh' for Chinese, 'en' for English) to use for message_to_user.")
        needs_completion: bool = dspy.OutputField()
        entity_a: str = dspy.OutputField(desc="First entity name; empty if needs_completion is false")
        entity_b: str = dspy.OutputField(desc="Second entity name; leave entirely empty if there is only one entity, or if needs_completion is false")
        message_to_user: str = dspy.OutputField(desc="A brief message in the requested language explaining why you are searching (e.g. 'I am searching PubMed for X...'); empty if needs_completion is false.")

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
                language=state.get("language", "en"),
            )

    try:
        pred = await asyncio.to_thread(_run)
        
        # Clean up string artifacts that open-source models sometimes emit
        ea = str(pred.entity_a).strip() if pred.entity_a else ""
        eb = str(pred.entity_b).strip() if pred.entity_b else ""
        if eb.lower() in ("none", "n/a", "null", "empty", "[]", "''", '""'):
            eb = ""
            
        needs = bool(pred.needs_completion) and bool(ea)
        
        logger.info("gap_assessment_result", needs=needs, raw_needs=pred.needs_completion, entity_a=ea, entity_b=eb, message=getattr(pred, "message_to_user", ""))
        
        return {
            **state,
            "needs_completion": needs,
            "entity_a": ea if needs else "",
            "entity_b": eb if needs else "",
            "message_to_user": getattr(pred, "message_to_user", "") if needs else "",
        }
    except Exception as exc:
        logger.warning("assess_gap_failed", error=str(exc))
        return {**state, "needs_completion": False}


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

    from medgraphia.retrieval.query_time_completion import complete_gap, complete_single_entity_gap

    cfg = get_settings()
    if not entity_b:
        evidence_str, new_chunks = await complete_single_entity_gap(entity_a, pubmed_limit=cfg.gap_completion_pubmed_limit)
    else:
        evidence_str, new_chunks = await complete_gap(entity_a, entity_b, pubmed_limit=cfg.gap_completion_pubmed_limit)
        
    gap_evidence = state.get("gap_evidence", []) + [evidence_str]
    gap_chunks = state.get("gap_chunks", []) + new_chunks
    return {**state, "gap_evidence": gap_evidence, "gap_chunks": gap_chunks, "tool_calls_made": tool_calls_made, "needs_completion": False}


def _route_after_assess(state: CompletionState) -> str:
    if state.get("needs_completion") and state.get("tool_calls_made", 0) < state.get("max_tool_calls", 2):
        return "execute_tool"
    return "end"


def _route_after_tool(state: CompletionState) -> str:
    # Once the cap is hit, another assess() call would be discarded by
    # _route_after_assess anyway — skip it instead of paying for an LLM
    # round-trip whose answer can never be acted on.
    if state.get("tool_calls_made", 0) < state.get("max_tool_calls", 2):
        return "assess"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_completion_graph() -> Any:
    """
    Build and compile the gap-completion LangGraph.

    Node sequence: assess -> (execute_tool -> assess)* -> END
    Loops back to assess after each tool call so the model can request a
    second pair, bounded by max_tool_calls. Skips the final assess once the
    cap is hit, since its answer would be discarded anyway.
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
    graph.add_conditional_edges("execute_tool", _route_after_tool, {"assess": "assess", "end": END})

    return graph.compile()


_compiled_graph: Any = None


async def run_gap_completion(
    question: str,
    context: str,
    entity_labels: list[str],
    entity_cui_map: dict[str, str],
    lm: Any,
    language_name: str = "en",
    max_tool_calls: int = 2,
) -> AsyncIterator[dict]:
    """
    Entry point used by the generation pipeline. Yields events and eventually a 'gap_result' dict
    containing evidence strings and new Chunk objects.
    """
    from typing import AsyncIterator

    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_completion_graph()
    if _compiled_graph is None:
        yield {"type": "gap_result", "evidence": [], "chunks": []}
        return

    initial_state: CompletionState = {
        "question": question,
        "context": context,
        "entity_labels": entity_labels,
        "entity_cui_map": entity_cui_map,
        "lm": lm,
        "language": language_name,
        "max_tool_calls": max_tool_calls,
        "tool_calls_made": 0,
        "gap_evidence": [],
        "gap_chunks": [],
    }

    final_state = dict(initial_state)
    try:
        async for event in _compiled_graph.astream(initial_state, stream_mode="updates"):
            for node_name, state_update in event.items():
                final_state.update(state_update)
                if node_name == "assess" and state_update.get("needs_completion"):
                    msg = state_update.get("message_to_user")
                    if msg:
                        yield {"type": "gap_message", "content": msg}
    except Exception as exc:
        logger.warning("gap_completion_graph_failed", error=str(exc))

    yield {
        "type": "gap_result",
        "evidence": final_state.get("gap_evidence", []),
        "chunks": final_state.get("gap_chunks", []),
    }
