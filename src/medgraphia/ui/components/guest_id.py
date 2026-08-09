"""
Persistent per-browser guest ID (first-party cookie) for anonymous session isolation.
"""

from __future__ import annotations

import time
import uuid

import streamlit as st
from streamlit_cookies_controller import CookieController

_COOKIE_NAME = "mg_guest_id"
_COOKIE_MAX_AGE_SECONDS = 365 * 86400
_MAX_RESOLVE_RETRIES = 10
_RETRY_DELAY_SECONDS = 0.3


def get_guest_id() -> str:
    """
    Return a stable per-browser guest ID, persisted in a first-party cookie.

    Mints a UUID on first visit and stores it in a long-lived cookie (same
    approach as GA/Mixpanel anonymous IDs) so refreshes and new tabs in the
    same browser reuse it; a different browser/device always gets a new one.

    CookieController.get() may return None — or even raise (its internal
    cookie dict is None until the component's browser round-trip resolves;
    streamlit_cookies_controller doesn't guard against reading it before
    then) — on a fresh Streamlit session's first script run, because that
    round-trip hasn't completed yet.

    Handing back a placeholder id in that window (the original approach) is
    unsafe: callers like sync_from_backend() use this id to fetch/scope chat
    history immediately, so a page can render "no history" against the wrong
    identity and never self-correct until some unrelated interaction happens
    to trigger another rerun. Instead, force an explicit st.rerun() (bounded
    by a retry counter) so the page never renders against an unresolved
    identity in the first place — the user briefly sees a rerun instead of a
    flash of "no history."
    """
    if st.session_state.get("_guest_id"):
        return st.session_state["_guest_id"]

    controller = CookieController(key="mg_cookie_controller")
    try:
        guest_id = controller.get(_COOKIE_NAME)
    except Exception:
        guest_id = None

    if guest_id:
        st.session_state["_guest_id"] = guest_id
        return guest_id

    retries = st.session_state.get("_guest_id_retries", 0)
    if retries < _MAX_RESOLVE_RETRIES:
        st.session_state["_guest_id_retries"] = retries + 1
        # A bare rerun can outrun the browser's iframe/postMessage round-trip
        # under real network conditions (production behind a tunnel is slower
        # than a local Docker test) — give it real wall-clock time to land.
        time.sleep(_RETRY_DELAY_SECONDS)
        st.rerun()  # halts this run; the next run gets another chance to resolve

    # Retries exhausted — the cookie is genuinely absent (true first-ever
    # visit, or the component never resolved). Mint and persist a new one.
    guest_id = str(uuid.uuid4())
    try:
        controller.set(_COOKIE_NAME, guest_id, max_age=_COOKIE_MAX_AGE_SECONDS)
    except Exception:
        pass  # will retry persisting on a later rerun
    st.session_state["_guest_id"] = guest_id
    return guest_id
