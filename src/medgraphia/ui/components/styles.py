"""
MedGraphia global theme — clinical, modern, with calm gradient accents.

Exports:
    inject_theme()       — call once per page after `st.set_page_config`
    render_brand()       — sidebar logo + wordmark + status pill
    banner()             — gradient page header
    status_badge()       — coloured pill HTML
    LOGO_SVG_INLINE      — raw SVG used as the favicon
"""
from __future__ import annotations

import streamlit as st

# Palette — deep clinical navy with teal accent
PRIMARY = "#0B3D91"
PRIMARY_LIGHT = "#1E5BBF"
ACCENT = "#0FB3A1"      # healing teal
SURFACE = "#F7F9FC"
CARD = "#FFFFFF"
BORDER = "#E2E8F0"
TEXT = "#1A202C"
MUTED = "#5A6478"
DANGER = "#C53030"
SUCCESS = "#2F855A"
WARN = "#B7791F"


# --------------------------------------------------------------------------
# Inline SVG Logo — a medical cross interwoven with knowledge-graph nodes.
# Used both as the favicon and in the sidebar brand block.
# --------------------------------------------------------------------------
LOGO_SVG_INLINE = """
<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="mg-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%"   stop-color="#0B3D91"/>
      <stop offset="100%" stop-color="#0FB3A1"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="14" fill="url(#mg-grad)"/>
  <!-- Medical cross -->
  <rect x="28" y="14" width="8"  height="36" rx="2" fill="#FFFFFF" opacity="0.95"/>
  <rect x="14" y="28" width="36" height="8"  rx="2" fill="#FFFFFF" opacity="0.95"/>
  <!-- Graph nodes overlay -->
  <circle cx="14" cy="14" r="3.4" fill="#FFFFFF"/>
  <circle cx="50" cy="14" r="3.4" fill="#FFFFFF"/>
  <circle cx="14" cy="50" r="3.4" fill="#FFFFFF"/>
  <circle cx="50" cy="50" r="3.4" fill="#FFFFFF"/>
  <line x1="14" y1="14" x2="32" y2="32" stroke="#FFFFFF" stroke-width="1.4" opacity="0.6"/>
  <line x1="50" y1="14" x2="32" y2="32" stroke="#FFFFFF" stroke-width="1.4" opacity="0.6"/>
  <line x1="14" y1="50" x2="32" y2="32" stroke="#FFFFFF" stroke-width="1.4" opacity="0.6"/>
  <line x1="50" y1="50" x2="32" y2="32" stroke="#FFFFFF" stroke-width="1.4" opacity="0.6"/>
</svg>
"""


_CSS = f"""
<style>
:root {{
    --mg-primary: {PRIMARY};
    --mg-primary-light: {PRIMARY_LIGHT};
    --mg-accent: {ACCENT};
    --mg-surface: {SURFACE};
    --mg-card: {CARD};
    --mg-border: {BORDER};
    --mg-text: {TEXT};
    --mg-muted: {MUTED};
}}

/* ── Page chrome ───────────────────────────────────────────────────────── */
#MainMenu, footer, .stDeployButton {{ visibility: hidden; }}
header {{ background: transparent !important; }}
.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}}

/* ── Banner ────────────────────────────────────────────────────────────── */
.mg-banner {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 60%, {ACCENT} 130%);
    color: white;
    padding: 1.3rem 1.6rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 28px rgba(11, 61, 145, 0.18);
    position: relative;
    overflow: hidden;
}}
.mg-banner::after {{
    content: ""; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px; border-radius: 50%;
    background: rgba(255,255,255,0.08);
}}
.mg-banner h1 {{
    color: white !important; font-size: 1.6rem; margin: 0;
    font-weight: 700; letter-spacing: 0.2px;
}}
.mg-banner p {{
    color: rgba(255,255,255,0.88); margin: 0.3rem 0 0;
    font-size: 0.92rem; max-width: 720px;
}}

/* ── Brand block in the sidebar ───────────────────────────────────────── */
.mg-brand {{
    display: flex; align-items: center; gap: 12px;
    padding: 1.2rem 1.5rem;
    border-bottom: 1px solid {BORDER};
    background: {SURFACE};
}}
[data-testid="stSidebar"] > div:first-child {{
    display: flex !important;
    flex-direction: column !important;
}}
[data-testid="stSidebarNav"] {{
    order: 2 !important;
}}
[data-testid="stSidebarContent"] {{
    order: 1 !important;
}}
.mg-brand-logo {{
    width: 44px; height: 44px; border-radius: 12px; flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(11, 61, 145, 0.28);
    overflow: hidden;
}}
.mg-brand-logo svg {{ width: 100%; height: 100%; display: block; }}
.mg-wordmark {{ line-height: 1.15; }}
.mg-name {{
    font-size: 1.25rem; font-weight: 800;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.mg-tag {{
    font-size: 0.62rem; letter-spacing: 1.6px; text-transform: uppercase;
    color: {MUTED};
}}

/* ── Connection pill ──────────────────────────────────────────────────── */
.mg-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
}}
.mg-pill-dot {{ width: 7px; height: 7px; border-radius: 50%; }}
.mg-online  {{ background: #E6F6EC; color: {SUCCESS};
               border: 1px solid rgba(47,133,90,.25); }}
.mg-online  .mg-pill-dot {{ background: {SUCCESS}; box-shadow: 0 0 6px {SUCCESS}; }}
.mg-offline {{ background: #FBE7E7; color: {DANGER};
               border: 1px solid rgba(197,48,48,.25); }}
.mg-offline .mg-pill-dot {{ background: {DANGER}; }}

/* ── Sidebar section heading ──────────────────────────────────────────── */
.mg-section {{
    font-size: 0.64rem; font-weight: 800; letter-spacing: 1.8px;
    text-transform: uppercase; color: {MUTED};
    margin: 1.2rem 0 0.5rem;
}}

/* ── Generic badge ─────────────────────────────────────────────────────── */
.mg-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.74rem; font-weight: 700; margin-right: 4px;
}}
.mg-badge-ok    {{ background: #E6F6EC; color: {SUCCESS}; }}
.mg-badge-warn  {{ background: #FEF6E4; color: {WARN}; }}
.mg-badge-err   {{ background: #FBE7E7; color: {DANGER}; }}
.mg-badge-info  {{ background: #E5EEFB; color: {PRIMARY}; }}

/* ── Cards & section titles ───────────────────────────────────────────── */
.mg-card {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    transition: border-color .18s, box-shadow .18s;
}}
.mg-card:hover {{
    border-color: {PRIMARY_LIGHT};
    box-shadow: 0 6px 18px rgba(11, 61, 145, 0.08);
}}
.mg-section-title {{
    font-size: 1.05rem; font-weight: 700; color: {PRIMARY};
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.3rem; border-bottom: 2px solid {BORDER};
}}

/* ── Feature tile (Home) ──────────────────────────────────────────────── */
.mg-tile {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.1rem 1.2rem; height: 100%;
    transition: transform .15s, box-shadow .18s, border-color .18s;
}}
.mg-tile:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(11, 61, 145, 0.1);
    border-color: {ACCENT};
}}
.mg-tile-icon {{
    font-size: 0; line-height: 0;
    width: 38px; height: 38px; border-radius: 9px;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    display: inline-block; margin-bottom: 0.6rem;
    box-shadow: 0 4px 10px rgba(11,61,145,0.2);
}}
.mg-tile-title {{ font-size: 1rem; font-weight: 700; color: {TEXT}; margin: 0; }}
.mg-tile-desc  {{ font-size: 0.84rem; color: {MUTED}; margin: 0.3rem 0 0; }}

/* ── Citation card (expander body) ────────────────────────────────────── */
.mg-cite {{
    display: flex; gap: 10px; align-items: flex-start;
    padding: 0.65rem 1rem; margin: 0.35rem 0; border-radius: 8px;
    background: rgba(15, 179, 161, 0.06);
    border: 1px solid rgba(15, 179, 161, 0.18);
    border-left: 3px solid {ACCENT};
}}
.mg-cite-num {{
    font-weight: 800; font-size: 0.78rem; color: {ACCENT};
    background: rgba(15, 179, 161, 0.14); border-radius: 4px;
    padding: 1px 7px; flex-shrink: 0; margin-top: 1px;
}}
.mg-cite-title {{ font-size: 0.82rem; font-weight: 700; color: {PRIMARY}; margin-bottom: 2px; }}
.mg-cite-meta  {{ font-size: 0.70rem; color: {MUTED}; margin-bottom: 4px; }}
.mg-cite-snippet {{ font-size: 0.78rem; color: {TEXT}; line-height: 1.55; }}

/* ── Inline citation ref (clickable [N]) ──────────────────────────────── */
.mg-cref {{
    display: inline-block; text-decoration: none;
    color: {ACCENT}; font-size: 0.72em; font-weight: 800;
    vertical-align: super; cursor: pointer;
    border: 1px solid rgba(15, 179, 161, .45);
    border-radius: 4px; padding: 0 4px; margin: 0 2px;
    line-height: 1.2; transition: background .15s;
}}
.mg-cref:hover {{ background: rgba(15, 179, 161, 0.18); color: {PRIMARY}; }}

/* ── Citation modal (CSS :target trick — no JS) ───────────────────────── */
.mg-covl {{
    display: none; position: fixed; inset: 0;
    background: rgba(11, 22, 50, 0.55);
    z-index: 99999; justify-content: center; align-items: center;
}}
.mg-covl:target {{ display: flex; }}
.mg-covl-bg {{ position: absolute; inset: 0; }}
.mg-cbox {{
    position: relative; z-index: 1;
    background: {CARD}; border-radius: 14px;
    padding: 1.3rem 1.5rem; max-width: 640px; width: 90%;
    border: 1px solid {BORDER};
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.28);
}}
.mg-cbox-hdr {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.8rem;
}}
.mg-cbox-num {{ color: {ACCENT}; font-weight: 800; font-size: 0.95rem; }}
.mg-cbox-x {{
    color: {MUTED}; text-decoration: none; font-size: 1rem;
    padding: 3px 9px; border-radius: 5px;
}}
.mg-cbox-x:hover {{ color: {TEXT}; background: {SURFACE}; }}
.mg-cbox-title {{ color: {PRIMARY}; font-weight: 800; font-size: 0.95rem; margin-bottom: 0.2rem; }}
.mg-cbox-src   {{ color: {MUTED}; font-size: 0.72rem; margin-bottom: 0.9rem; }}
.mg-cbox-body  {{
    color: {TEXT}; font-size: 0.85rem; line-height: 1.65;
    background: {SURFACE}; padding: 0.85rem 1rem; border-radius: 8px;
    border-left: 3px solid {ACCENT};
    white-space: pre-wrap; word-break: break-word;
    max-height: 480px; overflow-y: auto;
}}

/* ── Sidebar tweaks ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {SURFACE}; border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stButton button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}}
[data-testid="stSidebar"] .stButton > button:disabled {{
    background-color: rgba(15, 179, 161, 0.14) !important;
    color: {PRIMARY} !important;
    border: 1px solid rgba(15, 179, 161, 0.35) !important;
    opacity: 1 !important; cursor: default !important;
}}

/* ── Chat message hover + feedback fade ───────────────────────────────── */
[data-testid="stChatMessage"] {{
    border-radius: 10px !important; transition: background .18s;
}}
[data-testid="stChatMessage"]:hover {{
    background: rgba(11, 61, 145, 0.035) !important;
}}

/* ── Progress bar gradient ────────────────────────────────────────────── */
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT}) !important;
}}

/* ── Tab highlight ────────────────────────────────────────────────────── */
.stTabs [aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* ── Disclaimer ───────────────────────────────────────────────────────── */
.mg-disclaimer {{
    background: #FFF8E7; border-left: 4px solid {WARN};
    padding: 0.65rem 0.95rem; font-size: 0.82rem;
    color: #6B5B23; border-radius: 6px; margin-top: 0.6rem;
}}

/* ── Entity result chips (Graph Explorer) ─────────────────────────────── */
.mg-entity-chip {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 0.55rem 0.85rem; text-align: left;
    font-size: 0.85rem;
}}
</style>
"""


def inject_theme() -> None:
    """Inject the global CSS (idempotent — safe to call once per page)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_brand() -> None:
    """Render the sidebar brand block: logo + wordmark."""
    st.markdown(
        f"""
        <div class="mg-brand">
          <div class="mg-brand-logo">{LOGO_SVG_INLINE}</div>
          <div class="mg-wordmark">
            <div class="mg-name">MedGraphia</div>
            <div class="mg-tag">Medical GraphRAG</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def banner(title: str, subtitle: str | None = None) -> None:
    """Gradient page header used on every page."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""<div class="mg-banner"><h1>{title}</h1>{sub}</div>""",
        unsafe_allow_html=True,
    )


def status_badge(label: str, kind: str = "info") -> str:
    """Return HTML for a status pill (kind: ok | warn | err | info)."""
    return f'<span class="mg-badge mg-badge-{kind}">{label}</span>'


def connection_pill(online: bool, label_ok: str = "Connected",
                    label_off: str = "Offline") -> str:
    """Return HTML for an animated connection pill (sidebar)."""
    cls = "mg-online" if online else "mg-offline"
    txt = label_ok if online else label_off
    return (
        f'<span class="mg-pill {cls}">'
        f'<span class="mg-pill-dot"></span>{txt}</span>'
    )
