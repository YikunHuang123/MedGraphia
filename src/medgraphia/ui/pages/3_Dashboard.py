"""
Dashboard — Placeholder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_UI_ROOT = Path(__file__).resolve().parents[1]
if str(_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UI_ROOT))

from components.sidebar import render_common_sidebar  # noqa: E402
from components.styles import banner, inject_theme  # noqa: E402

st.set_page_config(page_title="Dashboard — MedGraphia", layout="wide")
inject_theme()

with st.sidebar:
    render_common_sidebar()

banner("Health Dashboard", "This section is currently under review.")

st.info(
    "The Health Dashboard has been decommissioned as requested. Its features may be integrated into other sections in the future."
)
