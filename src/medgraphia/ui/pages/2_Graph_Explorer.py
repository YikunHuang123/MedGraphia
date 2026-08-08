"""
Graph Explorer — entity search + interactive subgraph rendering.

Talks to:
    GET /graph/entity/search
    GET /graph/entity
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from api_client import APIError, MedGraphiaClient  # noqa: E402
from components.graph_viz import render_subgraph  # noqa: E402
from components.sidebar import render_common_sidebar  # noqa: E402
from components.styles import banner, inject_theme  # noqa: E402

st.set_page_config(page_title="Graph Explorer — MedGraphia", layout="wide")
inject_theme()
with st.sidebar:
    render_common_sidebar()
banner(
    "Graph Explorer",
    "Search a disease, drug or gene, then walk its 1–3 hop neighbourhood.",
)


# ---------------------------------------------------------------------------
# Cached search (debounces auto-complete on rerun)
# ---------------------------------------------------------------------------


@st.cache_resource
def _client_cached(base_url: str, api_key: str, admin_key: str) -> MedGraphiaClient:
    return MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key)


def _client() -> MedGraphiaClient:
    return _client_cached(
        st.session_state.get("api_base_url", "http://localhost:8058"),
        st.session_state.get("api_key", ""),
        st.session_state.get("admin_key", ""),
    )


@st.cache_data(ttl=15, show_spinner=False)
def _search_cached(
    base_url: str,
    api_key: str,
    admin_key: str,
    q: str,
    limit: int,
    entity_type: str | None,
) -> list[dict]:
    with MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key) as c:
        return c.search_entities(q=q, limit=limit, entity_type=entity_type)


_ENTITY_TYPES = ["(any)", "Disease", "Drug", "Symptom", "Gene", "Procedure", "Anatomy", "Physiology", "LivingBeing"]
_LANG_CHOICES = {"en": "English", "zh": "中文", "de": "Deutsch"}


# ---------------------------------------------------------------------------
# Search bar
# ---------------------------------------------------------------------------

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
    with c1:
        query = st.text_input(
            "Entity name",
            placeholder="e.g. metformin · 糖尿病 · Bluthochdruck",
            help="Minimum two characters; results auto-complete.",
            key="explorer_q",
            label_visibility="collapsed",
        )
    with c2:
        etype = st.selectbox("Type", _ENTITY_TYPES, index=0, label_visibility="collapsed")
    with c3:
        hops = st.slider("Hops", 1, 2, value=1, help="Maximum 2 hops to prevent browser lag.")
    with c4:
        lang = st.selectbox(
            "Lang",
            options=list(_LANG_CHOICES.keys()),
            format_func=_LANG_CHOICES.get,
            index=0,
            label_visibility="collapsed",
        )


# ---------------------------------------------------------------------------
# Auto-complete results
# ---------------------------------------------------------------------------

if query and len(query.strip()) >= 2:
    if not (st.session_state.get("api_key") or st.session_state.get("admin_key")):
        st.info("Add a User or Admin API key in the sidebar to query the backend.")
    else:
        try:
            results = _search_cached(
                st.session_state["api_base_url"],
                st.session_state["api_key"],
                st.session_state["admin_key"],
                q=query.strip(),
                limit=20,
                entity_type=None if etype == "(any)" else etype,
            )
        except APIError as exc:
            st.error(f"Search failed: {exc.detail}")
            results = []

        if not results:
            st.warning("No matching entities.")
        else:
            st.markdown(
                '<div class="mg-section-title">Matches</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(min(len(results), 4) or 1)
            for i, item in enumerate(results[:12]):
                col = cols[i % len(cols)]
                if lang == "zh" and item.get("lang_zh"):
                    label = item["lang_zh"]
                elif lang == "de" and item.get("lang_de"):
                    label = item["lang_de"]
                else:
                    label = item.get("label") or "?"
                cui = item.get("cui") or ""
                et = item.get("entity_type") or ""
                with col:
                    if st.button(
                        f"{label}  ·  {et}\nCUI: {cui}",
                        key=f"pick_{cui}_{i}",
                        use_container_width=True,
                    ):
                        st.session_state["explorer_picked"] = label
else:
    st.session_state.pop("explorer_picked", None)

target_name = st.session_state.get("explorer_picked")


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------

if target_name:
    st.markdown(
        f'<div class="mg-section-title">Subgraph · "{target_name}" ({hops}-hop)</div>',
        unsafe_allow_html=True,
    )
    try:
        payload = _client().get_entity_subgraph(name=target_name, hops=hops, lang=lang)
    except APIError as exc:
        st.error(f"Subgraph fetch failed: {exc.detail}")
        st.stop()

    st.session_state["last_subgraph"] = payload
    entity = payload.get("entity", {}) or {}
    subgraph = payload.get("subgraph", {}) or {}

    info = st.columns(4)
    info[0].metric("CUI", entity.get("cui", "—"))
    info[1].metric("Type", entity.get("entity_type", "—"))
    info[2].metric("Nodes", len(subgraph.get("nodes", []) or []))
    info[3].metric("Edges", len(subgraph.get("edges", []) or []))

    tab_viz, tab_raw = st.tabs(["Interactive graph", "Raw JSON"])
    with tab_viz:
        render_subgraph(subgraph, seed_cui=entity.get("cui"), lang=lang)
    with tab_raw:
        st.json(payload, expanded=False)
else:
    st.info("Type at least two characters above, then click a match to render its neighbourhood.")
