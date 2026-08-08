"""
pyvis-powered knowledge-graph visualisation.

`render_subgraph()` converts the JSON payload returned by
`GET /graph/entity` into an interactive HTML widget embedded via
`streamlit.components.v1.html`.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components

# Distinct colours for the entity labels the backend whitelists (see EntityType).
_NODE_COLOURS = {
    "Disease": "#E63946",  # red
    "Drug": "#1D4ED8",  # blue
    "Symptom": "#F59E0B",  # amber
    "Gene": "#10B981",  # green
    "Procedure": "#8B5CF6",  # violet
    "Anatomy": "#0EA5E9",  # sky blue
    "Physiology": "#84CC16",  # lime
    "LivingBeing": "#EC4899",  # pink
    "Community": "#6B7280",  # neutral grey
}
_DEFAULT_NODE_COLOUR = "#374151"
_SEED_NODE_COLOUR = "#0B3D91"  # match the app primary


def _node_label(node: dict[str, Any], lang: str = "en") -> str:
    """Pick the most informative single-line label for a node."""
    if lang == "zh" and node.get("lang_zh"):
        return node["lang_zh"]
    if lang == "de" and node.get("lang_de"):
        return node["lang_de"]
    return node.get("label") or node.get("name") or node.get("cui") or "?"


def _node_tooltip(node: dict[str, Any], lang: str = "en") -> str:
    """Build an HTML tooltip shown on hover."""
    lines = [f"<b>{_node_label(node, lang)}</b>"]
    if cui := node.get("cui"):
        lines.append(f"CUI: {cui}")
    if etype := node.get("entity_type"):
        lines.append(f"Type: {etype}")
    if zh := node.get("lang_zh"):
        lines.append(f"中文: {zh}")
    if de := node.get("lang_de"):
        lines.append(f"DE: {de}")
    return "<br>".join(lines)


def render_subgraph(
    subgraph: dict[str, Any],
    seed_cui: str | None = None,
    height: int = 560,
    lang: str = "en",
) -> None:
    """Render a Neo4j subgraph payload as an interactive pyvis network.

    Falls back to a friendly message if `pyvis` is not installed or the
    payload is empty.
    """
    nodes = subgraph.get("nodes") or []
    edges = subgraph.get("edges") or []

    if not nodes:
        st.info("No nodes in the returned subgraph.")
        return

    try:
        from pyvis.network import Network
    except ImportError:
        # Graceful degradation — render as a plain table.
        st.warning("`pyvis` is not installed; showing tabular fallback.")
        _render_table_fallback(nodes, edges)
        return

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#1A202C",
        directed=True,
        notebook=False,
    )
    # Force-directed physics — readable for small subgraphs (<200 nodes)
    net.barnes_hut(
        gravity=-12000,
        central_gravity=0.25,
        spring_length=130,
        spring_strength=0.04,
        damping=0.09,
    )

    seen_nodes: set[str] = set()
    for n in nodes:
        node_id = str(n.get("cui") or n.get("id") or _node_label(n, lang))
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        etype = n.get("entity_type") or n.get("type") or ""
        colour = (
            _SEED_NODE_COLOUR
            if seed_cui and node_id == seed_cui
            else _NODE_COLOURS.get(etype, _DEFAULT_NODE_COLOUR)
        )
        size = 26 if (seed_cui and node_id == seed_cui) else 16
        net.add_node(
            node_id,
            label=_node_label(n, lang),
            title=_node_tooltip(n, lang),
            color=colour,
            size=size,
            borderWidth=2,
        )

    for e in edges:
        src = str(e.get("source") or e.get("from") or "")
        dst = str(e.get("target") or e.get("to") or "")
        if not src or not dst or src not in seen_nodes or dst not in seen_nodes:
            continue
        rel = e.get("type") or e.get("label") or "RELATED_TO"
        net.add_edge(
            src,
            dst,
            label=rel,
            title=rel,
            color="#94A3B8",
            arrows="to",
        )

    # Don't write the file to disk — embed inline so the page stays self-contained
    html = net.generate_html(notebook=False)
    components.html(html, height=height + 20, scrolling=False)


def _render_table_fallback(nodes: list[dict], edges: list[dict]) -> None:
    """Tabular display used when pyvis is unavailable."""
    import pandas as pd

    if nodes:
        st.markdown("**Nodes**")
        st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
    if edges:
        st.markdown("**Edges**")
        st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)
