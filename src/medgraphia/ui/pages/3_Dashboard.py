"""
Dashboard — backend health + knowledge-graph statistics.

Talks to:
    GET /health/live
    GET /health/ready
    GET /graph/stats
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from api_client import APIError, MedGraphiaClient  # noqa: E402
from components.styles import (  # noqa: E402
    banner,
    connection_pill,
    inject_theme,
    render_brand,
    status_badge,
)


st.set_page_config(page_title="Dashboard — MedGraphia", layout="wide")
inject_theme()
with st.sidebar:
    render_brand()
banner("Dashboard", "Backend health snapshot and live graph statistics.")


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
def _ready_cached(base_url: str, api_key: str, admin_key: str) -> dict:
    with MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key) as c:
        return c.health_ready()


@st.cache_data(ttl=30, show_spinner=False)
def _stats_cached(base_url: str, api_key: str, admin_key: str) -> dict:
    with MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key) as c:
        return c.graph_stats()


# ---------------------------------------------------------------------------
# Health row
# ---------------------------------------------------------------------------

st.markdown('<div class="mg-section-title">Service health</div>',
            unsafe_allow_html=True)

h1, h2, h3 = st.columns(3)
client = _client()

with h1:
    try:
        live = client.health_live()
        ok = live.get("status") == "ok"
    except Exception:
        ok = False
    st.markdown("**API**", unsafe_allow_html=True)
    st.markdown(connection_pill(ok), unsafe_allow_html=True)

with h2:
    try:
        ready = _ready_cached(
            st.session_state["api_base_url"],
            st.session_state["api_key"],
            st.session_state["admin_key"],
        )
    except Exception:
        ready = {}
    neo_ok = ready.get("neo4j") == "ok"
    vec_ok = ready.get("qdrant") == "ok"
    st.markdown("**Backing stores**", unsafe_allow_html=True)
    st.markdown(
        status_badge("Neo4j",  "ok" if neo_ok else "err")
        + status_badge("Qdrant", "ok" if vec_ok else "err"),
        unsafe_allow_html=True,
    )

with h3:
    st.markdown("**Endpoint**", unsafe_allow_html=True)
    st.markdown(
        f"<code>{st.session_state['api_base_url']}</code>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

st.markdown('<div class="mg-section-title">Knowledge-graph statistics</div>',
            unsafe_allow_html=True)

try:
    stats = _stats_cached(
        st.session_state["api_base_url"],
        st.session_state["api_key"],
        st.session_state["admin_key"],
    )
except APIError as exc:
    st.error(f"Could not load graph statistics: {exc.detail}")
    st.stop()

s1, s2, s3, s4 = st.columns(4)
s1.metric("Total nodes",   f"{stats.get('nodes', 0):,}")
s2.metric("Relationships", f"{stats.get('relations', 0):,}")
s3.metric("Documents",     f"{stats.get('documents', 0):,}")
s4.metric("Chunks",        f"{stats.get('chunks', 0):,}")


# Per-label breakdown — render as a Plotly bar chart when available
breakdown = {
    k.replace("count_", "").title(): v
    for k, v in stats.items()
    if k.startswith("count_") and isinstance(v, (int, float))
}

if breakdown:
    try:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(
            sorted(breakdown.items(), key=lambda x: x[1], reverse=True),
            columns=["label", "count"],
        )
        fig = px.bar(
            df, x="count", y="label", orientation="h",
            color="count", color_continuous_scale=["#1E5BBF", "#0FB3A1"],
            text="count",
            title="Node count by label",
        )
        fig.update_layout(
            showlegend=False, height=380, coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif"),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import pandas as pd
        st.dataframe(
            pd.DataFrame(breakdown.items(), columns=["label", "count"]),
            hide_index=True, use_container_width=True,
        )
else:
    st.caption(
        "Per-label counts are not exposed by `/graph/stats`. "
        "Showing aggregate metrics only."
    )


with st.expander("Raw `/graph/stats` payload"):
    st.json(stats)

if st.button("Refresh", type="primary"):
    _ready_cached.clear()
    _stats_cached.clear()
    st.rerun()
