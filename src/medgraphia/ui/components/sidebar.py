"""
Shared sidebar components for MedGraphia.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st
from components.styles import (
    connection_pill,
    render_brand,
    status_badge,
)
from api_client import MedGraphiaClient


@st.cache_data(ttl=30, show_spinner=False)
def _check_health_cached(
    base_url: str, api_key: str, admin_key: str
) -> dict[str, Any]:
    with MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key) as client:
        return client.health_ready()


def render_common_sidebar() -> None:
    """Render the standard top-level sidebar sections."""
    # Ensure health state exists in session_state to avoid blocking first run
    if "_health_data" not in st.session_state:
        st.session_state["_health_data"] = {}
    if "_last_health_check" not in st.session_state:
        st.session_state["_last_health_check"] = 0

    # 1. Asynchronous-like Health Check Trigger
    # We only check if enough time has passed, and we do it WITHOUT a spinner
    now = time.time()
    if (now - st.session_state["_last_health_check"]) > 30:
        try:
            # We use a shorter timeout for the sidebar check to avoid hanging
            with MedGraphiaClient(
                base_url=st.session_state["api_base_url"],
                api_key=st.session_state["api_key"],
                admin_key=st.session_state["admin_key"]
            ) as client:
                # Use a very short timeout for this background-ish check
                client.client.timeout = httpx.Timeout(2.0)
                st.session_state["_health_data"] = client.health_ready()
                st.session_state["_last_health_check"] = now
        except Exception:
            # Silently fail, let the UI show based on last known or empty data
            st.session_state["_health_data"] = {"overall": "error"}
            st.session_state["_last_health_check"] = now

    ready = st.session_state["_health_data"]
    neo_ok = ready.get("neo4j") == "ok"
    vec_ok = ready.get("qdrant") == "ok"
    overall = ready.get("overall", "")
    is_warming = "warming_up" in overall
    online = neo_ok and vec_ok

    # 2. Brand with Overlay Status
    p_label = "WARM" if is_warming else ("ON" if online else "OFF")
    pill_html = connection_pill(online or is_warming, label_ok=p_label, label_off="OFF")
    render_brand(status_html=pill_html)

    # 3. Compact Database Badges & Warming Alert
    if online or is_warming:
        st.markdown(
            f'<div style="margin: -0.8rem 1.4rem 0.5rem; display: flex; gap: 4px;">'
            f'{status_badge("N4J", "ok" if neo_ok else "err")} '
            f'{status_badge("VEC", "ok" if vec_ok else "err")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        
        if is_warming:
            st.warning("⚙️ **System Warming Up**")
            st.caption("Eagerly loading medical AI models (GLiNER, BGE-M3, SapBERT). Chat will be available in ~30s.")
    elif not ready:
         st.caption("⏳ Checking connectivity...")

    # 4. Navigation (Custom Module Selector)
    st.markdown('<div class="mg-section">Workspaces</div>', unsafe_allow_html=True)
    st.page_link("streamlit_app.py", label="Home", icon=":material/home:")
    st.page_link("pages/1_Chat.py", label="Clinical Chat", icon=":material/chat:")
    st.page_link("pages/2_Graph_Explorer.py", label="Graph Explorer", icon=":material/account_tree:")
    st.page_link("pages/4_Admin.py", label="Admin Console", icon=":material/admin_panel_settings:")

    # 4. Compact Database Badges (Optional, only if online)
    if online:
        neo_ok = ready.get("neo4j") == "ok"
        vec_ok = ready.get("qdrant") == "ok"
        st.markdown(
            f'<div style="margin: -0.8rem 1.4rem 0.5rem; display: flex; gap: 4px;">'
            f'{status_badge("N4J", "ok" if neo_ok else "err")} '
            f'{status_badge("VEC", "ok" if vec_ok else "err")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 4. Account
    st.markdown('<div class="mg-section">Account</div>', unsafe_allow_html=True)
    with st.container(border=False):
        st.session_state["api_key"] = st.text_input(
            "User API Key",
            value=st.session_state["api_key"],
            type="password",
            placeholder="Paste User API Key...",
            help="Required for clinical Q&A and graph search.",
            key="sidebar_user_key" # unique key for this widget instance
        )
        # Sync back to main state if changed (Streamlit widget state can be tricky)
        st.session_state["api_key"] = st.session_state["sidebar_user_key"]
        
        st.session_state["admin_key"] = st.text_input(
            "Admin API Key",
            value=st.session_state["admin_key"],
            type="password",
            placeholder="Paste Admin Key...",
            help="Optional. Required for pipeline management and stats.",
            key="sidebar_admin_key"
        )
        st.session_state["admin_key"] = st.session_state["sidebar_admin_key"]

    if st.session_state["admin_key"]:
        st.caption("🔒 **Admin Mode** enabled")
    elif st.session_state["api_key"]:
        st.caption("✅ **Standard Access** enabled")
    else:
        st.caption("⚠️ Please provide an API key to start.")
