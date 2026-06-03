"""
Chat — multi-conversation GraphRAG Q&A with citation modals + history.

Talks to:
    POST /chat/stream  (SSE)
    POST /chat         (sync, fallback)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator

import streamlit as st

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from api_client import APIError, MedGraphiaClient  # noqa: E402
from components import chat_history  # noqa: E402
from components.citations import (  # noqa: E402
    render_citation_cards,
    render_message_with_citations,
)
from components.styles import (  # noqa: E402
    banner,
    inject_theme,
    render_brand,
    status_badge,
)


st.set_page_config(page_title="Chat — MedGraphia", layout="wide")
inject_theme()


# ---------------------------------------------------------------------------
# Cached client (mirrors streamlit_app.py)
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


# ---------------------------------------------------------------------------
# Sidebar — brand, navigation, conversation list
# ---------------------------------------------------------------------------

def render_chat_sidebar() -> None:
    with st.sidebar:
        render_brand()

        st.markdown('<div class="mg-section">Conversations</div>',
                    unsafe_allow_html=True)

        if st.button("+ New conversation", use_container_width=True, type="primary"):
            chat_history.new_conversation(language="unknown")
            st.rerun()

        convs = chat_history.list_conversations()
        if not convs:
            st.caption("No conversations yet.")
        else:
            active_id = st.session_state.get("active_conv_id")
            for c in convs:
                is_active = (c["id"] == active_id)
                is_editing = (st.session_state.get("_editing_conv") == c["id"])

                if is_editing:
                    ec, bc = st.columns([5, 1])
                    with ec:
                        new_title = st.text_input(
                            "Rename",
                            value=c["title"],
                            key=f"rename_val_{c['id']}",
                            label_visibility="collapsed",
                        )
                    with bc:
                        if st.button("OK", key=f"save_{c['id']}",
                                     use_container_width=True):
                            chat_history.rename_conversation(c["id"], new_title)
                            st.session_state["_editing_conv"] = None
                            st.rerun()
                else:
                    sc, rc, dc = st.columns([6, 1, 1])
                    label = c["title"]
                    short = label if len(label) <= 22 else label[:21] + "…"
                    with sc:
                        if st.button(
                            ("→ " + short) if is_active else short,
                            key=f"pick_{c['id']}",
                            use_container_width=True,
                            disabled=is_active,
                        ):
                            chat_history.set_active(c["id"])
                            st.rerun()
                    with rc:
                        if st.button("Rn", key=f"ren_{c['id']}",
                                     help="Rename"):
                            st.session_state["_editing_conv"] = c["id"]
                            st.rerun()
                    with dc:
                        if st.button("×", key=f"del_{c['id']}",
                                     help="Delete"):
                            chat_history.delete_conversation(c["id"])
                            st.rerun()


render_chat_sidebar()
banner("Chat", "Ask a clinical question — answers cite their source chunks.")


# ---------------------------------------------------------------------------
# Top control bar — active conv summary
# ---------------------------------------------------------------------------

active = chat_history.ensure_active(
    language=st.session_state.get("chat_language", "unknown")
)

st.markdown(
    f"**{active['title']}**  "
    f"<span style='color:#5A6478;font-size:0.82rem'>· {len(active['messages'])} "
    f"message(s) · backend session "
    f"<code>{active.get('backend_session_id') or 'pending'}</code></span>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Replay history
# ---------------------------------------------------------------------------

for i, msg in enumerate(active["messages"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_with_citations(
                msg.get("content", ""),
                msg.get("citations", []),
                message_key=f"hist-{active['id']}-{i}",
            )
            render_citation_cards(msg.get("citations", []))
            disc = msg.get("disclaimer", "")
            if disc:
                st.markdown(
                    f'<div class="mg-disclaimer">{disc}</div>',
                    unsafe_allow_html=True,
                )
            mdl = msg.get("model_used", "")
            if mdl:
                st.caption(f"Model: `{mdl}`")
        else:
            st.markdown(msg.get("content", ""))


# ---------------------------------------------------------------------------
# SSE token iterator (consumed by st.write_stream)
# ---------------------------------------------------------------------------

def _stream_tokens(events: Iterator[dict], meta_sink: dict) -> Iterator[str]:
    for ev in events:
        kind = ev.get("type")
        if kind == "chunk":
            yield ev.get("content", "")
        elif kind == "error":
            yield f"\n\n_[error] {ev.get('detail', 'stream interrupted')}_"
            return
        elif kind == "citations":
            meta_sink["citations"] = ev.get("citations", [])
        elif kind == "done":
            meta_sink["done"] = ev
            return


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask a medical question…")
if prompt:
    if not (st.session_state.get("api_key") or st.session_state.get("admin_key")):
        st.error("Please paste a User or Admin API key in the sidebar first.")
        st.stop()

    chat_history.append_message(
        active["id"],
        {"role": "user", "content": prompt, "ts": time.time()},
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    client = _client()
    with st.chat_message("assistant"):
        try:
            meta: dict = {}
            events = client.chat_stream(
                message=prompt,
                session_id=active.get("backend_session_id"),
                language="unknown",  # Force auto-detection on backend
            )
            full_text = st.write_stream(_stream_tokens(events, meta))
            done = meta.get("done", {})
            citations = meta.get("citations", [])
            disclaimer = done.get("disclaimer", "")
            model_used = done.get("model_used", "")
            if sid := done.get("session_id"):
                chat_history.attach_backend_session(active["id"], sid)

            # Citation cards + disclaimer + model badge
            render_citation_cards(citations)
            if disclaimer:
                st.markdown(
                    f'<div class="mg-disclaimer">{disclaimer}</div>',
                    unsafe_allow_html=True,
                )
            if model_used:
                st.caption(f"Model: `{model_used}`")

            # Persist to client-side history
            chat_history.append_message(
                active["id"],
                {
                    "role": "assistant",
                    "content": full_text or "",
                    "citations": citations,
                    "disclaimer": disclaimer,
                    "model_used": model_used,
                    "ts": time.time(),
                },
            )

        except APIError as exc:
            st.error(f"Chat failed: {exc.detail}")

    # Re-render so the inline citation modal anchors register properly.
    st.rerun()
