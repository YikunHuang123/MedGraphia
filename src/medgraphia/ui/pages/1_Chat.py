"""
Chat — multi-conversation GraphRAG Q&A with citation modals + history.

Talks to:
    POST /chat/stream  (SSE)
    POST /chat         (sync, fallback)
"""

from __future__ import annotations

import html as html_lib
import sys
import time
from collections.abc import Iterator
from pathlib import Path

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
)

st.set_page_config(page_title="Chat — MedGraphia", layout="wide")
inject_theme()


from components.sidebar import render_api_settings, render_common_sidebar, has_any_valid_key, get_current_client  # noqa: E402

# ── Sync history from backend on first load / API key input ────────────────
if st.session_state.get("api_key") or st.session_state.get("admin_key"):
    chat_history.sync_from_backend(get_current_client())

def render_chat_sidebar() -> None:
    with st.sidebar:
        # 1. Standard Sidebar Sections (No Settings Yet)
        render_common_sidebar(include_settings=False)

        # 2. Conversations Section
        has_valid_key = has_any_valid_key()
        
        if st.button(
            "➕ New Chat",
            use_container_width=True,
            type="primary",
            disabled=not has_valid_key,
            help="Please configure your API key below to start chatting." if not has_valid_key else None
        ):
            chat_history.new_conversation(language="unknown")
            st.rerun()

        convs = chat_history.list_conversations()
        if not convs:
            st.caption("No conversations yet.")
        else:
            active_id = st.session_state.get("active_conv_id")

            # Pagination Logic
            page_size = 8
            total_convs = len(convs)
            total_pages = (total_convs + page_size - 1) // page_size
            curr_page = st.session_state.setdefault("conv_page", 1)

            start_idx = (curr_page - 1) * page_size
            end_idx = start_idx + page_size
            page_convs = convs[start_idx:end_idx]

            for c in page_convs:
                is_active = c["id"] == active_id
                is_editing = st.session_state.get("_editing_conv") == c["id"]

                if is_editing:
                    ec, bc = st.columns([4, 1])
                    with ec:
                        new_title = st.text_input(
                            "Rename",
                            value=c["title"],
                            key=f"rename_val_{c['id']}",
                            label_visibility="collapsed",
                        )
                    with bc:
                        if st.button("OK", key=f"save_{c['id']}", use_container_width=True):
                            chat_history.rename_conversation(c["id"], new_title)
                            st.session_state["_editing_conv"] = None
                            st.rerun()
                else:
                    sc, ac = st.columns([6, 1], vertical_alignment="center")
                    label = c["title"]
                    short = label if len(label) <= 18 else label[:17] + "…"
                    with sc:
                        if st.button(
                            short,
                            key=f"pick_{c['id']}",
                            use_container_width=True,
                            disabled=is_active,
                            type="secondary" if is_active else "tertiary"
                        ):
                            chat_history.set_active(c["id"])
                            st.rerun()
                    with ac:
                        with st.popover("⋮", use_container_width=True):
                            if st.button("✏️ Rename", key=f"ren_{c['id']}", use_container_width=True, type="tertiary"):
                                st.session_state["_editing_conv"] = c["id"]
                                st.rerun()
                            if st.button("🗑️ Delete", key=f"del_{c['id']}", use_container_width=True, type="tertiary"):
                                chat_history.delete_conversation(c["id"])
                                st.rerun()

            # Pagination Controls
            if total_pages > 1:
                st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
                pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
                with pcol1:
                    if st.button(
                        "◀", key="prev_c_page", disabled=curr_page <= 1, use_container_width=True
                    ):
                        st.session_state.conv_page -= 1
                        st.rerun()
                with pcol2:
                    st.markdown(
                        f"<div style='text-align:center; font-size:0.75rem; color:#64748B; line-height:30px;'>"
                        f"{curr_page} / {total_pages}</div>",
                        unsafe_allow_html=True,
                    )
                with pcol3:
                    if st.button(
                        "▶",
                        key="next_c_page",
                        disabled=curr_page >= total_pages,
                        use_container_width=True,
                    ):
                        st.session_state.conv_page += 1
                        st.rerun()

        # 3. Settings at the very bottom
        render_api_settings()



render_chat_sidebar()
banner("Clinical Chat", "Ask a clinical question — answers cite their source chunks.")

# ---------------------------------------------------------------------------
# Top control bar — active conv summary
# ---------------------------------------------------------------------------

active = chat_history.get_active()
if not active:
    has_key = bool(st.session_state.get("api_key") or st.session_state.get("admin_key"))
    has_valid_key = has_any_valid_key()

    if not has_key:
        st.warning("⚠️ Please provide your API key in the sidebar below to start chatting.")
    elif not has_valid_key:
        st.warning("⚠️ Please provide a valid API key in the sidebar to start chatting.")
    else:
        st.info("👈 Please select a conversation from the sidebar or click **+ New Chat** to start.")
    st.stop()

# Lazy-load content if this was synced from backend summary
if active.get("is_lazy"):
    with st.spinner("Loading conversation..."):
        chat_history.load_full_session(get_current_client(), active["id"])

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
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
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


def _stream_tokens(events: Iterator[dict], meta_sink: dict, status_placeholder=None) -> Iterator[str]:
    first_chunk = True
    thoughts_html = ""
    current_progress = "Processing..."

    for ev in events:
        kind = ev.get("type")
        if kind == "progress":
            if status_placeholder and first_chunk:
                # ev.content ultimately comes from LLM output — escape before interpolating into HTML.
                current_progress = html_lib.escape(ev.get("content", "").rstrip("."))
                if thoughts_html:
                    html = f'''<div class="mg-status-block">
<div class="mg-progress" style="margin-bottom:8px;"><span class="mg-progress-icon">⏳</span><span class="mg-progress-text">{current_progress}</span></div>
<div style="border-left: 2px solid #e2e8f0; padding-left: 12px; margin-left: 6px; display: flex; flex-direction: column; gap: 6px;">
{thoughts_html}
</div>
</div>'''
                else:
                    html = f'<div class="mg-progress"><span class="mg-progress-icon">⏳</span><span class="mg-progress-text">{current_progress}</span></div>'
                status_placeholder.markdown(html, unsafe_allow_html=True)
        elif kind == "thought":
            if status_placeholder and first_chunk:
                # ev.content is the LLM's own message_to_user — escape before interpolating into HTML.
                content = html_lib.escape(ev.get("content", "").strip())
                if content:
                    thoughts_html += f'<div style="font-size: 0.85rem; color: #71717a; line-height: 1.4;">&gt; {content}</div>\n'
                    html = f'''<div class="mg-status-block">
<div class="mg-progress" style="margin-bottom:8px;"><span class="mg-progress-icon">⏳</span><span class="mg-progress-text">{current_progress}</span></div>
<div style="border-left: 2px solid #e2e8f0; padding-left: 12px; margin-left: 6px; display: flex; flex-direction: column; gap: 6px;">
{thoughts_html}
</div>
</div>'''
                    status_placeholder.markdown(html, unsafe_allow_html=True)
        elif kind == "chunk":
            if first_chunk and status_placeholder:
                status_placeholder.empty()
                first_chunk = False
            yield ev.get("content", "")
        elif kind == "error":
            yield f"\n\n_[error] {ev.get('detail', 'stream interrupted')}_"
            return
        elif kind == "moderated":
            meta_sink["moderated"] = True
            yield f"\n\n⚠️ **[MODERATED]** {ev.get('detail', 'Content redacted for safety')}"
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
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    client = get_current_client()
    with st.chat_message("assistant", avatar="🤖"):
        status_placeholder = st.empty()
        try:
            meta: dict = {}
            events = client.chat_stream(
                message=prompt,
                session_id=active.get("backend_session_id"),
                language="unknown",  # Force auto-detection on backend
            )
            full_text = st.write_stream(_stream_tokens(events, meta, status_placeholder=status_placeholder))
            status_placeholder.empty()

            # Check for moderation
            if meta.get("moderated"):
                full_text = (
                    "⚠️ [Blocked by Safety Guardrails] The generated response was found to "
                    "violate safety policies. Please rephrase your question."
                )
                citations = []
                disclaimer = ""
                model_used = "guardrail"
            else:
                done = meta.get("done", {})
                citations = meta.get("citations", [])
                disclaimer = done.get("disclaimer", "")
                model_used = done.get("model_used", "")
                if sid := done.get("session_id"):
                    chat_history.attach_backend_session(active["id"], sid)

            # Citation cards + disclaimer + model badge
            if not meta.get("moderated"):
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
