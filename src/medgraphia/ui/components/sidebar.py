"""
Shared sidebar components for MedGraphia.
"""

from __future__ import annotations

import html
import os
import time
from typing import Any

import httpx
import streamlit as st
from api_client import MedGraphiaClient
from components.guest_id import get_guest_id
from components.styles import (
    connection_pill,
    render_brand,
)
from streamlit_cookies_controller import CookieController

_API_KEY_COOKIE = "mg_api_key"
_ADMIN_KEY_COOKIE = "mg_admin_key"
_KEY_COOKIE_MAX_AGE_SECONDS = 365 * 86400


@st.cache_data(ttl=30, show_spinner=False)
def _check_health_cached(base_url: str, api_key: str, admin_key: str) -> dict[str, Any]:
    with MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key) as client:
        return client.health_ready()


def render_common_sidebar(include_settings: bool = True) -> None:
    """Render the standard top-level sidebar sections."""
    # Keys typed into the sidebar are only saved to st.session_state, which
    # Streamlit wipes on a full page refresh — persist them in a first-party
    # cookie (same approach as guest_id.py) so a refresh doesn't log you out.
    _key_cookies = CookieController(key="mg_cookie_controller_keys")
    saved_api_key = _key_cookies.get(_API_KEY_COOKIE) or ""
    saved_admin_key = _key_cookies.get(_ADMIN_KEY_COOKIE) or ""

    # Ensure core state exists (required for all pages)
    defaults = {
        "api_base_url": os.getenv("API_BASE_URL", "http://localhost:8058"),
        "conversations": {},
        "active_conv_id": None,
        "last_subgraph": None,
        "_health_ts": 0.0,
        "_health_cached": None,
        "_health_data": {},
        "_last_health_check": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # api_key/admin_key are handled separately from the loop above: the cookie
    # component's value resolves asynchronously, so on a session's first-ever
    # script run saved_api_key/saved_admin_key may still read back "" even
    # though the cookie holds a real value — a Streamlit rerun fires once the
    # component actually resolves. Re-checking on every rerun (instead of only
    # when the key is absent from session_state) means that follow-up rerun
    # still picks up the real value instead of locking in an empty default.
    if not st.session_state.get("api_key"):
        st.session_state["api_key"] = os.getenv("MEDGRAPHIA_API_KEY", "") or saved_api_key
    if not st.session_state.get("admin_key"):
        st.session_state["admin_key"] = os.getenv("MEDGRAPHIA_ADMIN_KEY", "") or saved_admin_key

    # 1. Asynchronous-like Health Check Trigger
    now = time.time()
    if (now - st.session_state["_last_health_check"]) > 30:
        try:
            with MedGraphiaClient(
                base_url=st.session_state["api_base_url"],
                api_key=st.session_state["api_key"],
                admin_key=st.session_state["admin_key"],
            ) as client:
                client.client.timeout = httpx.Timeout(2.0)
                st.session_state["_health_data"] = client.health_ready()
                st.session_state["_last_health_check"] = now
        except Exception:
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

    # 3. Warming Alert
    if is_warming:
        st.warning("⚙️ **System Warming Up**")
        st.caption(
            "Eagerly loading medical AI models (GLiNER, BGE-M3, SapBERT). Chat will be available in ~30s."
        )
    elif not ready:
        st.caption("⏳ Checking connectivity...")

    # 4. Navigation
    st.page_link("streamlit_app.py", label="Home", icon=":material/home:")
    st.page_link("pages/1_Chat.py", label="Clinical Chat", icon=":material/chat:")
    st.page_link(
        "pages/2_Graph_Explorer.py", label="Graph Explorer", icon=":material/account_tree:"
    )
    st.page_link("pages/4_Admin.py", label="Admin Console", icon=":material/admin_panel_settings:")

    if include_settings:
        render_api_settings()


def _validate_keys():
    from api_client import MedGraphiaClient
    base_url = st.session_state.get("api_base_url", "http://localhost:8058")

    # Validate User Key
    if st.session_state.get("api_key") and not st.session_state.get("_api_key_validated"):
        try:
            with MedGraphiaClient(base_url=base_url, api_key=st.session_state["api_key"]) as client:
                client.graph_stats()
            st.session_state["_api_key_validated"] = True
            st.session_state.pop("api_error", None)
        except Exception as e:
            st.session_state["api_error"] = str(e)
            st.session_state.pop("_api_key_validated", None)

    # Validate Admin Key
    if st.session_state.get("admin_key") and not st.session_state.get("_admin_key_validated"):
        try:
            with MedGraphiaClient(base_url=base_url, admin_key=st.session_state["admin_key"]) as client:
                client.admin_graph_stats()
            st.session_state["_admin_key_validated"] = True
            st.session_state.pop("admin_error", None)
        except Exception as e:
            st.session_state["admin_error"] = str(e)
            st.session_state.pop("_admin_key_validated", None)


def get_auth_state() -> tuple[bool, bool, bool, bool]:
    """Returns (has_api_error, has_admin_error, api_key_valid, admin_key_valid)"""
    _validate_keys()
    has_api_error = bool(st.session_state.get("api_error"))
    has_admin_error = bool(st.session_state.get("admin_error"))
    api_key_valid = bool(st.session_state.get("api_key")) and not has_api_error
    admin_key_valid = bool(st.session_state.get("admin_key")) and not has_admin_error
    return has_api_error, has_admin_error, api_key_valid, admin_key_valid


def has_any_valid_key() -> bool:
    """Returns True if the user has at least one valid key (User or Admin)."""
    _, _, api_valid, admin_valid = get_auth_state()
    return api_valid or admin_valid

@st.cache_resource
def _client_cached(base_url: str, api_key: str, admin_key: str, client_id: str) -> MedGraphiaClient:
    return MedGraphiaClient(base_url=base_url, api_key=api_key, admin_key=admin_key, client_id=client_id)


def get_current_client() -> MedGraphiaClient:
    """
    Returns a configured MedGraphiaClient using ONLY valid keys.
    If a key is marked as error in the UI, it is stripped out so the
    backend client does not attempt to use it (and thus fail 403).

    Cached by credentials (+ guest ID) so repeated calls across reruns reuse
    the same httpx connection pool instead of leaking a new one each time.
    """
    _, _, api_valid, admin_valid = get_auth_state()

    base_url = st.session_state.get("api_base_url", "http://localhost:8058")
    api_key = st.session_state.get("api_key", "") if api_valid else ""
    admin_key = st.session_state.get("admin_key", "") if admin_valid else ""

    return _client_cached(base_url, api_key, admin_key, get_guest_id())

def render_api_settings() -> None:
    """Render the API keys expander at the bottom of the sidebar."""
    st.markdown("<br>", unsafe_allow_html=True)

    def _sync_api_key():
        st.session_state["api_key"] = st.session_state["sidebar_user_key"]
        st.session_state["_api_key_dirty"] = True
        st.session_state.pop("_history_synced", None)
        st.session_state.pop("api_error", None)
        st.session_state.pop("_api_key_validated", None)
        st.session_state["conversations"] = {}
        st.session_state["active_conv_id"] = None

    def _sync_admin_key():
        st.session_state["admin_key"] = st.session_state["sidebar_admin_key"]
        st.session_state["_admin_key_dirty"] = True
        st.session_state.pop("_history_synced", None)
        st.session_state.pop("_admin_key_validated", None)
        st.session_state.pop("admin_error", None)
        st.session_state["conversations"] = {}
        st.session_state["active_conv_id"] = None

    with st.expander("🔑 API Settings", expanded=False):
        st.text_input(
            "User API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="Paste User API Key...",
            help="Required for clinical Q&A and graph search.",
            key="sidebar_user_key",
            on_change=_sync_api_key,
        )

        st.text_input(
            "Admin API Key",
            value=st.session_state.get("admin_key", ""),
            type="password",
            placeholder="Paste Admin Key...",
            help="Optional. Required for pipeline management and stats.",
            key="sidebar_admin_key",
            on_change=_sync_admin_key,
        )

    # Persist key changes to the cookie here in the main script body — custom
    # components (CookieController) can't be invoked from inside the on_change
    # callbacks above, since callbacks run outside the normal render context.
    # Gated on the "_dirty" flags (set only by an actual user edit above) rather
    # than diffing against session_state directly: on a session's first-ever
    # script run, the cookie *read* in render_common_sidebar() may not have
    # resolved yet (component round-trip is async), so session_state["admin_key"]
    # can still be "" even though the cookie holds a real saved value — writing
    # unconditionally in that window would clobber the saved key with "".
    api_key_dirty = st.session_state.pop("_api_key_dirty", False)
    admin_key_dirty = st.session_state.pop("_admin_key_dirty", False)
    if api_key_dirty or admin_key_dirty:
        _key_cookies = CookieController(key="mg_cookie_controller_keys_settings")
        if api_key_dirty:
            _key_cookies.set(
                _API_KEY_COOKIE, st.session_state.get("api_key", ""), max_age=_KEY_COOKIE_MAX_AGE_SECONDS
            )
        if admin_key_dirty:
            _key_cookies.set(
                _ADMIN_KEY_COOKIE, st.session_state.get("admin_key", ""), max_age=_KEY_COOKIE_MAX_AGE_SECONDS
            )

    has_api_error, has_admin_error, api_key_valid, admin_key_valid = get_auth_state()

    # Display User Key Status (Hide error if Admin Key is valid)
    if has_api_error and not admin_key_valid:
        st.markdown(
            f"<div style='font-size:0.8rem; color:#ef4444; padding:6px 10px; border-radius:6px; background:rgba(239,68,68,0.1); margin-bottom:4px; line-height:1.4;'>❌ <b>User Key Error:</b> {html.escape(st.session_state['api_error'])}</div>",
            unsafe_allow_html=True,
        )
    elif api_key_valid:
        st.caption("✅ **Standard Access** enabled")

    # Display Admin Key Status (Hide error if User Key is valid)
    if has_admin_error and not api_key_valid:
        st.markdown(
            f"<div style='font-size:0.8rem; color:#ef4444; padding:6px 10px; border-radius:6px; background:rgba(239,68,68,0.1); margin-bottom:4px; line-height:1.4;'>❌ <b>Admin Key Error:</b> {html.escape(st.session_state['admin_error'])}</div>",
            unsafe_allow_html=True,
        )
    elif admin_key_valid:
        st.caption("🔒 **Admin Access** enabled")

    if not st.session_state.get("api_key") and not st.session_state.get("admin_key"):
        st.caption("⚠️ Please provide an API key to start.")
