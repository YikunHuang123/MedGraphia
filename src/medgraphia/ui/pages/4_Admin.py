"""
Admin — pipeline orchestration + API-key lifecycle management.

All endpoints require an admin-role X-API-Key (set in the sidebar).

Talks to:
    POST   /admin/pipeline/trigger
    GET    /admin/pipeline/status
    POST   /admin/keys
    GET    /admin/keys
    DELETE /admin/keys/{prefix}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from api_client import APIError, MedGraphiaClient  # noqa: E402
from components.styles import (  # noqa: E402
    banner,
    inject_theme,
    render_brand,
    status_badge,
)


st.set_page_config(page_title="Admin — MedGraphia", layout="wide")
inject_theme()
with st.sidebar:
    render_brand()
banner("Admin", "Ingestion pipelines and API-key lifecycle.")


@st.cache_resource
def _client_cached(base_url: str, admin_key: str) -> MedGraphiaClient:
    return MedGraphiaClient(base_url=base_url, admin_key=admin_key)


def _client() -> MedGraphiaClient:
    return _client_cached(
        st.session_state.get("api_base_url", "http://localhost:8058"),
        st.session_state.get("admin_key", ""),
    )


# Gate page until an admin key is present.
if not st.session_state.get("admin_key"):
    st.warning(
        "Set an **Admin API key** in the sidebar to access this page. "
        "All endpoints below require admin role."
    )
    st.stop()


tab_pipeline, tab_keys, tab_models = st.tabs(
    ["Pipeline", "API keys", "Model configuration"]
)


# ===========================================================================
# Pipeline tab
# ===========================================================================

with tab_pipeline:
    st.markdown(
        '<div class="mg-section-title">Trigger an ingestion run</div>',
        unsafe_allow_html=True,
    )

    with st.form("trigger_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            domain = st.text_input(
                "Vertical domain",
                value="t2dm",
                help="Used as the partition key, e.g. `t2dm`, `cardio`.",
            )
            pubmed_query = st.text_input(
                "Custom PubMed query (optional)",
                value="",
                help="Leave blank to use the default query for this domain.",
            )
        with c2:
            pubmed_limit = st.number_input(
                "PubMed article limit",
                min_value=1, max_value=5000, value=200, step=50,
            )
            include_drugbank = st.checkbox("Include DrugBank", value=False)
            include_ema_smpc = st.checkbox("Include EMA SmPC", value=False)

        submitted = st.form_submit_button("Start pipeline", type="primary")

    if submitted:
        try:
            resp = _client().trigger_pipeline(
                domain=domain.strip() or "t2dm",
                pubmed_query=pubmed_query.strip() or None,
                pubmed_limit=int(pubmed_limit),
                include_drugbank=include_drugbank,
                include_ema_smpc=include_ema_smpc,
            )
            st.success(resp.get("message", "Pipeline accepted."))
        except APIError as exc:
            if exc.status_code == 409:
                st.warning("A run for this domain is already in progress.")
            else:
                st.error(f"Trigger failed: {exc.detail}")

    st.markdown(
        '<div class="mg-section-title">Run status</div>',
        unsafe_allow_html=True,
    )

    poll_domain = st.text_input(
        "Domain to poll", value=domain if submitted else "t2dm",
        key="status_domain",
    )
    auto_refresh = st.toggle("Auto-refresh every 5 s", value=False)

    try:
        status = _client().pipeline_status(domain=poll_domain.strip() or "t2dm")
    except APIError as exc:
        st.error(f"Status fetch failed: {exc.detail}")
        status = {}

    stage = status.get("stage", "idle")
    stage_kind = {
        "completed": "ok", "running": "info", "starting": "info",
        "failed": "err", "idle": "warn",
    }.get(stage, "info")

    live_dot = ""
    if stage in ("running", "starting"):
        live_dot = (
            ' <span style="color:#0FB3A1;font-weight:700;font-size:0.8rem">'
            '● live</span>'
        )

    st.markdown(
        f"**Current stage:** {status_badge(stage, stage_kind)}{live_dot}",
        unsafe_allow_html=True,
    )

    progress = float(status.get("progress") or 0.0)
    st.progress(min(max(progress, 0.0), 1.0),
                text=f"{progress * 100:.1f}% complete")

    mc = st.columns(4)
    mc[0].metric("Nodes created",    f"{status.get('nodes_created', 0):,}")
    mc[1].metric("Edges created",    f"{status.get('edges_created', 0):,}")
    mc[2].metric("Docs processed",   f"{status.get('documents_processed', 0):,}")
    mc[3].metric("Docs total",       f"{status.get('documents_total', 0):,}")

    if err := status.get("error"):
        st.error(f"Last error: {err}")

    with st.expander("Raw status payload"):
        st.json(status)

    if auto_refresh and stage in {"running", "starting"}:
        time.sleep(5)
        st.rerun()


# ===========================================================================
# API keys tab
# ===========================================================================

with tab_keys:
    st.markdown(
        '<div class="mg-section-title">Issue a new API key</div>',
        unsafe_allow_html=True,
    )

    cc1, cc2, _ = st.columns([1, 1, 2])
    role = cc1.selectbox("Role", ["user", "admin"], index=0)
    if cc2.button("Generate", type="primary", use_container_width=True):
        try:
            new = _client().create_api_key(role=role)
            st.success("Key created — copy it now, it cannot be retrieved again.")
            st.code(new.get("api_key", ""), language=None)
            st.caption(
                f"Role: `{new.get('role')}`  ·  Prefix: `{new.get('prefix')}`"
            )
        except APIError as exc:
            st.error(f"Could not create key: {exc.detail}")

    st.markdown(
        '<div class="mg-section-title">Active keys</div>',
        unsafe_allow_html=True,
    )

    try:
        keys = _client().list_api_keys()
    except APIError as exc:
        st.error(f"Could not list keys: {exc.detail}")
        keys = []

    if not keys:
        st.info("No active API keys.")
    else:
        for k in keys:
            prefix_full = k.get("prefix", "")
            prefix_clean = prefix_full.rstrip("…")
            r1, r2, r3 = st.columns([2, 1, 1])
            r1.markdown(f"**`{prefix_full}`**")
            r2.markdown(
                status_badge(k.get("role", "user"), "info"),
                unsafe_allow_html=True,
            )
            if r3.button("Revoke", key=f"revoke_{prefix_clean}",
                         use_container_width=True):
                try:
                    res = _client().revoke_api_key(prefix_clean[:8])
                    st.success(f"Revoked {res.get('count', 0)} key(s).")
                    st.rerun()
                except APIError as exc:
                    st.error(f"Revoke failed: {exc.detail}")


# ===========================================================================
# Model configuration tab
# ===========================================================================

with tab_models:
    st.markdown(
        '<div class="mg-section-title">Backend model routing</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These values are read from the API process environment "
        "(`config.py`). Adjust `.env` and restart the API to change them."
    )

    try:
        _client().admin_graph_stats()
        st.markdown(status_badge("backend reachable", "ok"),
                    unsafe_allow_html=True)
    except APIError as exc:
        st.markdown(status_badge("backend unreachable", "err"),
                    unsafe_allow_html=True)
        st.caption(exc.detail[:120])

    st.markdown(
        """
        | Tier | Used for | Default model |
        |---|---|---|
        | SMALL  | Patient FAQ, lightweight chat       | `qwen2.5:3b` (ollama) |
        | MEDIUM | Drug-interaction lookups            | `qwen2.5:3b` (ollama) |
        | LARGE  | Clinical reasoning, multi-hop QA    | `qwen2.5:3b` (ollama) |

        Override per tier with `LLM_SMALL_PROVIDER`, `LLM_SMALL_MODEL`,
        `LLM_MEDIUM_*`, `LLM_LARGE_*`. Supported providers: `openai`,
        `anthropic`, `deepseek`, `gemini`, `ollama`.
        """
    )
