"""
Persistent per-browser guest ID (first-party cookie) for anonymous session isolation.
"""

from __future__ import annotations

import uuid

import streamlit as st
from streamlit_cookies_controller import CookieController

_COOKIE_NAME = "mg_guest_id"
_COOKIE_MAX_AGE_SECONDS = 365 * 86400


def get_guest_id() -> str:
    """
    Return a stable per-browser guest ID, persisted in a first-party cookie.

    Mints a UUID on first visit and stores it in a long-lived cookie (same
    approach as GA/Mixpanel anonymous IDs) so refreshes and new tabs in the
    same browser reuse it; a different browser/device always gets a new one.
    """
    if "_guest_id" in st.session_state:
        return st.session_state["_guest_id"]

    controller = CookieController(key="mg_cookie_controller")
    guest_id = controller.get(_COOKIE_NAME)

    if not guest_id:
        guest_id = str(uuid.uuid4())
        controller.set(_COOKIE_NAME, guest_id, max_age=_COOKIE_MAX_AGE_SECONDS)

    st.session_state["_guest_id"] = guest_id
    return guest_id
