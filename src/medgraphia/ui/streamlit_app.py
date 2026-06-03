"""
MedGraphia — Streamlit frontend entrypoint.

Run locally:
    streamlit run src/medgraphia/ui/streamlit_app.py

In Docker (lite stack):
    docker compose -f docker-compose.lite.yml up
    Browse to http://localhost:8501
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# Allow `streamlit run src/medgraphia/ui/streamlit_app.py` to resolve the
# sibling `api_client` / `components` packages without an editable install.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from api_client import APIError, MedGraphiaClient  # noqa: E402
from components import chat_history  # noqa: E402
from components.styles import (  # noqa: E402
    banner,
    connection_pill,
    inject_theme,
    render_brand,
    status_badge,
)


# ---------------------------------------------------------------------------
# Page config — favicon uses the brand SVG via data URI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedGraphia — Medical GraphRAG",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "api_base_url": os.getenv("API_BASE_URL", "http://localhost:8058"),
        "api_key":      os.getenv("MEDGRAPHIA_API_KEY", ""),
        "admin_key":    os.getenv("MEDGRAPHIA_ADMIN_KEY", ""),
        "conversations": {},        # client-side multi-conversation store
        "active_conv_id": None,
        "last_subgraph": None,
        "_health_ts": 0.0,
        "_health_cached": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()

from components.sidebar import render_common_sidebar  # noqa: E402

# ---------------------------------------------------------------------------
# Cached resources / data
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client(base_url: str, api_key: str, admin_key: str) -> MedGraphiaClient:
    """Persistent HTTP client (cached by credentials triplet)."""
    return MedGraphiaClient(
        base_url=base_url, api_key=api_key, admin_key=admin_key,
    )


def current_client() -> MedGraphiaClient:
    return get_client(
        st.session_state["api_base_url"],
        st.session_state["api_key"],
        st.session_state["admin_key"],
    )


@st.cache_data(ttl=60, show_spinner=False)
def _graph_stats_cached(
    base_url: str, api_key: str, admin_key: str
) -> dict[str, int]:
    with MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key) as client:
        return client.graph_stats()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    render_common_sidebar()


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

banner(
    "MedGraphia",
    "Multilingual medical knowledge-graph & GraphRAG console — clinical "
    "Q&A, drug-interaction lookups and graph exploration in one place.",
)


# ── Quick stats row ────────────────────────────────────────────────────────
stats: dict[str, int] = {}
try:
    stats = _graph_stats_cached(
        st.session_state["api_base_url"],
        st.session_state["api_key"],
        st.session_state["admin_key"],
    )
except Exception as exc:
    st.warning(
        f"Could not load graph statistics — verify your API key / connection. "
        f"Details: {exc}"
    )

m1, m2, m3 = st.columns(3)
m1.metric("Graph nodes",   f"{stats.get('nodes', 0):,}")
m2.metric("Relationships", f"{stats.get('relations', 0):,}")
m3.metric(
    "Conversations",
    f"{len(st.session_state.get('conversations', {})):,}",
    help="Local browser session only.",
)


# ── Capability tiles ───────────────────────────────────────────────────────
st.markdown(
    '<div class="mg-section-title">Workspaces</div>',
    unsafe_allow_html=True,
)

_TILES = [
    ("Chat",
     "Stream traceable answers in EN / ZH / DE with inline citations and "
     "conversation history.",
     "Chat"),
    ("Graph Explorer",
     "Search a disease, drug or gene then walk its 1–3 hop neighbourhood "
     "with an interactive graph.",
     "Graph_Explorer"),
    ("Dashboard",
     "Backend liveness, vector-store readiness, and a per-label breakdown of "
     "the knowledge graph.",
     "Dashboard"),
    ("Admin",
     "Kick off ingestion pipelines, monitor progress, and manage user / "
     "admin API keys.",
     "Admin"),
]

cols = st.columns(4)
for col, (title, desc, _page) in zip(cols, _TILES):
    with col:
        st.markdown(
            f"""
            <div class="mg-tile">
              <span class="mg-tile-icon"></span>
              <div class="mg-tile-title">{title}</div>
              <div class="mg-tile-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Sync history from backend ──────────────────────────────────────────────
if (st.session_state.get("api_key") or st.session_state.get("admin_key")):
    chat_history.sync_from_backend(current_client())

# ── Recent conversations ───────────────────────────────────────────────────
convs = chat_history.list_conversations()
if convs:
    st.markdown(
        '<div class="mg-section-title">Recent conversations</div>',
        unsafe_allow_html=True,
    )
    for c in convs[:5]:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["updated_at"]))
        n_msgs = len(c["messages"])
        cc1, cc2, cc3 = st.columns([6, 1, 1])
        with cc1:
            st.markdown(
                f"**{c['title']}**  "
                f"<span style='color:#5A6478;font-size:0.78rem'> · "
                f"{ts} · {n_msgs} message(s) · lang `{c['language']}`</span>",
                unsafe_allow_html=True,
            )
        with cc2:
            if st.button("Open", key=f"open_{c['id']}", use_container_width=True):
                chat_history.set_active(c["id"])
                st.switch_page("pages/1_Chat.py")
        with cc3:
            if st.button("Delete", key=f"del_home_{c['id']}", use_container_width=True):
                chat_history.delete_conversation(c["id"])
                st.rerun()
else:
    st.info(
        "No conversations yet. Open **Chat** in the sidebar to start your "
        "first one."
    )


# ── Disclaimer ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="mg-disclaimer">'
    "MedGraphia is a decision-support tool, not a substitute for professional "
    "medical advice. Always verify generated answers against the cited "
    "sources before acting on them."
    "</div>",
    unsafe_allow_html=True,
)
