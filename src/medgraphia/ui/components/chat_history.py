"""
Client-side conversation history for the Chat page.

The backend issues one `session_id` per conversation, but the chat history
itself is not exposed by an API endpoint yet, so we track multiple
conversations in `st.session_state["conversations"]`. Each entry holds:

    {
        "id":         "uuid issued by backend on first reply",
        "title":      "auto-derived from first user message",
        "language":   "en|zh|de",
        "messages":   [{role, content, citations, model_used, ts}, …],
        "created_at": epoch seconds,
        "updated_at": epoch seconds,
    }

This is intentionally lightweight — survives page navigation within the
same browser tab; lost on full reload (acceptable for a Phase 9 MVP).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import streamlit as st


_STATE_KEY = "conversations"
_ACTIVE_KEY = "active_conv_id"


def _store() -> dict[str, dict[str, Any]]:
    """Lazily initialise the conversations dict."""
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = {}
    return st.session_state[_STATE_KEY]


def list_conversations() -> list[dict[str, Any]]:
    """Return conversations sorted by most recently updated first."""
    convs = list(_store().values())
    convs.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
    return convs


def get_active() -> dict[str, Any] | None:
    """Return the currently-selected conversation, if any."""
    return _store().get(st.session_state.get(_ACTIVE_KEY))


def set_active(conv_id: str | None) -> None:
    st.session_state[_ACTIVE_KEY] = conv_id


def new_conversation(language: str = "en") -> dict[str, Any]:
    """Create a fresh empty conversation and mark it active."""
    cid = f"local-{uuid.uuid4().hex[:10]}"
    now = time.time()
    conv = {
        "id": cid,
        "backend_session_id": None,   # filled in after first reply
        "title": "New conversation",
        "language": language,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    _store()[cid] = conv
    set_active(cid)
    return conv


def delete_conversation(conv_id: str) -> None:
    store = _store()
    if conv_id in store:
        del store[conv_id]
    if st.session_state.get(_ACTIVE_KEY) == conv_id:
        set_active(None)


def rename_conversation(conv_id: str, title: str) -> None:
    conv = _store().get(conv_id)
    if conv:
        conv["title"] = title.strip() or conv["title"]
        conv["updated_at"] = time.time()


def append_message(conv_id: str, message: dict[str, Any]) -> None:
    """Append a message and touch updated_at."""
    conv = _store().get(conv_id)
    if not conv:
        return
    message.setdefault("ts", time.time())
    conv["messages"].append(message)
    conv["updated_at"] = message["ts"]
    # Auto-derive title from the first user message
    if conv["title"] == "New conversation" and message.get("role") == "user":
        content = (message.get("content") or "").strip().replace("\n", " ")
        conv["title"] = content[:48] + ("…" if len(content) > 48 else "")


def attach_backend_session(conv_id: str, backend_session_id: str) -> None:
    """Bind the backend-issued session_id once the first reply comes back."""
    conv = _store().get(conv_id)
    if conv and not conv.get("backend_session_id"):
        conv["backend_session_id"] = backend_session_id


def ensure_active(language: str = "en") -> dict[str, Any]:
    """Return the active conversation, creating one if none exists."""
    active = get_active()
    if active is None:
        active = new_conversation(language=language)
    return active
