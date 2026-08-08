"""
Client-side conversation history for the Chat page.
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
        "backend_session_id": None,  # filled in after first reply
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


def sync_from_backend(client: Any) -> None:
    """
    Fetch the list of sessions from the backend and populate st.session_state.
    Only does this once per browser session unless forced.
    """
    if st.session_state.get("_history_synced"):
        return

    try:
        backend_sessions = client.list_sessions()
        store = _store()
        for s in backend_sessions:
            sid = s["session_id"]
            # Avoid overwriting if already in local store
            if sid not in store:
                # Derive a title if first_message exists
                title = s.get("first_message") or "Untitled Conversation"
                if len(title) > 40:
                    title = title[:37] + "..."

                # Convert ISO string to epoch if needed
                updated_at = s.get("updated_at", time.time())
                if isinstance(updated_at, str):
                    try:
                        from datetime import datetime

                        updated_at = datetime.fromisoformat(updated_at).timestamp()
                    except:
                        updated_at = time.time()

                store[sid] = {
                    "id": sid,
                    "backend_session_id": sid,
                    "title": title,
                    "language": s.get("language", "en"),
                    "messages": [],  # Messages will be lazy-loaded when active
                    "created_at": updated_at,
                    "updated_at": updated_at,
                    "is_lazy": True,  # Flag indicating messages need fetching
                }
        st.session_state["_history_synced"] = True
        st.session_state.pop("api_error", None)
    except Exception as e:
        st.session_state["api_error"] = str(e)


def load_full_session(client: Any, conv_id: str) -> None:
    """Lazy-load full message history for a specific session."""
    conv = _store().get(conv_id)
    if not conv or not conv.get("is_lazy"):
        return

    try:
        full_data = client.get_session(conv["backend_session_id"])
        # Map backend message model to UI message dict
        messages = []
        for m in full_data.get("messages", []):
            messages.append(
                {
                    "role": m["role"],
                    "content": m["content"],
                    "citations": m.get("citations", []),
                    "model_used": m.get("model_used", ""),
                    "ts": m.get("created_at") or time.time(),
                }
            )
        conv["messages"] = messages
        conv["is_lazy"] = False
    except Exception as e:
        st.error(f"Failed to load session details: {e}")


def attach_backend_session(conv_id: str, backend_session_id: str) -> None:
    """Bind the backend-issued session_id once the first reply comes back."""
    conv = _store().get(conv_id)
    if conv and not conv.get("backend_session_id"):
        conv["backend_session_id"] = backend_session_id


def ensure_active(language: str = "en") -> dict[str, Any]:
    """Return the active conversation, creating one if none exists."""
    active = get_active()
    if active is None:
        store = _store()
        if store:
            # Pick the most recently updated conversation
            latest = max(store.values(), key=lambda c: c.get("updated_at", 0))
            set_active(latest["id"])
            active = latest
        else:
            active = new_conversation(language=language)
    return active
