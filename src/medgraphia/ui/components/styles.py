"""
MedGraphia global theme — clinical, modern, with calm gradient accents.

Exports:
    inject_theme()       — call once per page after `st.set_page_config`
    render_brand()       — sidebar logo + wordmark + status pill
    banner()             — gradient page header
    status_badge()       — coloured pill HTML
    connection_pill()    — animated connection status
    LOGO_SVG_INLINE      — raw SVG used as the favicon
"""

from __future__ import annotations

import streamlit as st

# Palette — deep clinical navy with vibrant accents
PRIMARY = "#0B3D91"
PRIMARY_LIGHT = "#1E5BBF"
ACCENT = "#0FB3A1"  # healing teal
ACCENT_VIBRANT = "#10B981"  # emerald
SURFACE = "#F8FAFC"
CARD = "#FFFFFF"
BORDER = "#E2E8F0"
TEXT_DARK = "#0F172A"
TEXT_MAIN = "#1E293B"
TEXT_MUTED = "#64748B"
DANGER = "#EF4444"
SUCCESS = "#22C55E"
WARN = "#F59E0B"


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
/* ── Global ── */
:root {{
    --mg-primary: {PRIMARY};
    --mg-primary-light: {PRIMARY_LIGHT};
    --mg-accent: {ACCENT};
    --mg-surface: {SURFACE};
    --mg-card: {CARD};
    --mg-border: {BORDER};
    --mg-text-dark: {TEXT_DARK};
    --mg-text-main: {TEXT_MAIN};
    --mg-text-muted: {TEXT_MUTED};
}}

#MainMenu, footer, .stDeployButton {{ visibility: hidden; }}
header {{ background: transparent !important; }}
.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}}

/* ── Chat Content Layout ── */
[data-testid="stChatMessage"],
[data-testid="stChatInput"] {{
    max-width: 850px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}}

/* ── Sidebar Reordering & Nav Hide ── */
[data-testid="stSidebarHeader"] {{
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
}}
[data-testid="stSidebarUserContent"] {{
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    display: flex !important;
    flex-direction: column !important;
}}
/* Hide the default auto-generated navigation */
[data-testid="stSidebarNav"] {{
    display: none !important;
}}
[data-testid="stSidebarContent"] {{
    order: 1 !important;
}}

/* ── Custom Page Links ── */
[data-testid="stPageLink"] a {{
    padding: 0.4rem 1rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: {TEXT_MAIN} !important;
    transition: all 0.2s ease !important;
    margin: 0 !important;
    border: 1px solid transparent !important;
}}
[data-testid="stPageLink"] a:hover {{
    background-color: rgba(11, 61, 145, 0.05) !important;
    border-color: rgba(11, 61, 145, 0.1) !important;
    color: {PRIMARY} !important;
}}
/* Active link styling — Streamlit sets a specific background */
[data-testid="stPageLink"] a[aria-current="page"] {{
    background-color: rgba(15, 179, 161, 0.1) !important;
    color: {ACCENT} !important;
    border-color: rgba(15, 179, 161, 0.2) !important;
}}


/* ── History Items Pill Styling ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {{
    gap: 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) {{
    border-radius: 8px !important;
    align-items: center !important;
    transition: all 0.2s;
    margin-bottom: 2px;
    margin-top: -0.6rem !important; /* Pull items closer */
}}
/* Inactive hover */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]):not(:has(button:disabled)):hover {{
    background-color: rgba(11, 61, 145, 0.05) !important;
}}
/* Active bg */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]):has(button:disabled) {{
    background-color: rgba(15, 179, 161, 0.15) !important;
}}
/* Active text bold */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]):has(button:disabled) button p {{
    font-weight: 700 !important;
    color: {PRIMARY} !important;
}}
/* Strip button borders */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.2rem 0.4rem !important;
    min-height: 2rem !important;
}}
/* Hide chevron in popover button completely */
[data-testid="stSidebar"] div[data-testid="stPopover"] svg {{
    display: none !important;
}}
/* Shrink popover body and buttons */
[data-testid="stPopoverBody"] {{
    padding: 0.3rem !important;
    min-width: 110px !important;
}}
[data-testid="stPopoverBody"] button {{
    padding: 0.1rem 0.5rem !important;
    min-height: 1.8rem !important;
    margin: 0 !important;
    border: none !important;
}}

/* ── Banner ── */
.mg-banner {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 60%, {ACCENT} 130%);
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(11, 61, 145, 0.1);
    position: relative;
    overflow: hidden;
}}
.mg-banner::after {{
    content: ""; position: absolute; top: -20px; right: -20px;
    width: 80px; height: 80px; border-radius: 50%;
    background: rgba(255,255,255,0.06);
}}
.mg-banner h1 {{
    color: white !important; font-size: 1.25rem; margin: 0;
    font-weight: 700; letter-spacing: -0.01em;
}}
.mg-banner p {{
    color: rgba(255,255,255,0.85); margin: 0.2rem 0 0;
    font-size: 0.8rem; max-width: 800px;
}}

/* ── Brand ── */
.mg-brand {{
    display: flex; align-items: center; gap: 12px;
    padding: 0.5rem 1.4rem 1rem;
    border-bottom: 1px solid {BORDER};
    margin-top: -2.5rem;
    margin-bottom: 0.2rem;
    position: relative;
}}
.mg-brand-logo {{
    width: 42px; height: 42px; border-radius: 11px; flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(11, 61, 145, 0.2);
    overflow: hidden;
}}
.mg-brand-logo svg {{ width: 100%; height: 100%; display: block; }}
.mg-wordmark {{ line-height: 1.2; }}
.mg-name {{
    font-size: 1.2rem; font-weight: 800;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.mg-tag {{
    font-size: 0.6rem; letter-spacing: 1.8px; text-transform: uppercase;
    color: {TEXT_MUTED};
}}

/* ── Connection Pill ── */
.mg-pill {{
    position: absolute;
    top: 10px; right: 14px;
    display: flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 10px;
    font-size: 0.58rem; font-weight: 800;
    text-transform: uppercase;
    z-index: 10;
}}
.mg-pill-dot {{ width: 6px; height: 6px; border-radius: 50%; }}
.mg-online  {{ background: rgba(34,197,94,0.1); color: {SUCCESS};
               border: 1px solid rgba(34,197,94,0.2); }}
.mg-online  .mg-pill-dot {{ background: {SUCCESS}; box-shadow: 0 0 8px {SUCCESS}; }}
.mg-offline {{ background: rgba(239,68,68,0.1); color: {DANGER};
               border: 1px solid rgba(239,68,68,0.2); }}
.mg-offline .mg-pill-dot {{ background: {DANGER}; }}

/* ── Sidebar Sections ── */
.mg-section {{
    font-size: 0.65rem; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: {TEXT_MUTED};
    margin: 0.2rem 1.4rem 0.6rem;
}}

/* ── Custom Cards ── */
.mg-card {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    transition: all .2s ease;
}}
.mg-card:hover {{
    border-color: {PRIMARY_LIGHT};
    box-shadow: 0 8px 20px rgba(11, 61, 145, 0.06);
}}
.mg-section-title {{
    font-size: 1.1rem; font-weight: 700; color: {PRIMARY};
    margin: 1.8rem 0 0.8rem;
    padding-bottom: 0.4rem; border-bottom: 2px solid {BORDER};
}}

/* ── Feature Tiles ── */
.mg-tile {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 1.3rem; height: 100%;
    transition: all .2s ease;
    display: flex; flex-direction: column;
}}
.mg-tile:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(11, 61, 145, 0.08);
    border-color: {ACCENT};
}}
.mg-tile-icon {{
    width: 42px; height: 42px; border-radius: 10px;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 12px rgba(11,61,145,0.15);
}}
.mg-tile-title {{ font-size: 1.05rem; font-weight: 700; color: {TEXT_DARK}; margin: 0; }}
.mg-tile-desc  {{ font-size: 0.85rem; color: {TEXT_MUTED}; margin: 0.4rem 0 0; line-height: 1.5; }}

/* ── Progress Indicator ── */
.mg-progress {{
    display: flex; align-items: center; gap: 10px;
    color: {ACCENT}; font-weight: 600; font-size: 0.95rem;
    padding: 0.5rem 0;
}}
.mg-progress-icon {{
    display: inline-block;
    animation: flipHourglass 1.5s infinite ease-in-out;
}}
@keyframes flipHourglass {{
    0% {{ transform: rotate(0deg); }}
    50% {{ transform: rotate(180deg); }}
    100% {{ transform: rotate(180deg); }}
}}
.mg-progress-text::after {{
    content: '';
    animation: loadingDots 1.5s infinite steps(4, end);
}}
@keyframes loadingDots {{
    0% {{ content: ''; }}
    25% {{ content: '.'; }}
    50% {{ content: '..'; }}
    75% {{ content: '...'; }}
    100% {{ content: ''; }}
}}

/* ── Citations ── */
.mg-cite {{
    display: flex; gap: 12px; align-items: flex-start;
    padding: 0.8rem 1.1rem; margin: 0.4rem 0; border-radius: 10px;
    background: rgba(15, 179, 161, 0.04);
    border: 1px solid rgba(15, 179, 161, 0.12);
    border-left: 4px solid {ACCENT};
}}
.mg-cite-num {{
    font-weight: 800; font-size: 0.75rem; color: #fff;
    background: {ACCENT}; border-radius: 5px;
    padding: 2px 8px; flex-shrink: 0; margin-top: 1px;
}}
.mg-cite-title {{ font-size: 0.85rem; font-weight: 700; color: {PRIMARY}; margin-bottom: 3px; overflow-wrap: break-word; word-break: break-word; }}
.mg-cite-meta  {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-bottom: 6px; }}
.mg-cite-snippet {{ font-size: 0.82rem; color: {TEXT_MAIN}; line-height: 1.6; }}

/* ── Inline citation ref ── */
.mg-cref {{
    display: inline-block; text-decoration: none;
    color: {ACCENT}; font-size: 0.75em; font-weight: 800;
    vertical-align: super; cursor: pointer;
    border: 1px solid rgba(15, 179, 161, .3);
    border-radius: 4px; padding: 0 5px; margin: 0 2px;
    transition: all .15s;
}}
.mg-cref:hover {{ background: {ACCENT}; color: #fff; }}

/* ── Modals ── */
.mg-covl {{
    display: none; position: fixed; inset: 0;
    background: rgba(15, 23, 42, 0.6);
    z-index: 99999; justify-content: center; align-items: center;
    backdrop-filter: blur(4px);
}}
.mg-covl:target {{ display: flex; }}
.mg-covl-bg {{ position: absolute; inset: 0; }}
.mg-cbox {{
    position: relative; z-index: 1;
    background: {CARD}; border-radius: 18px;
    padding: 1.5rem 1.8rem; max-width: 680px; width: 92%;
    border: 1px solid {BORDER};
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.25);
}}
.mg-cbox-hdr {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1rem;
}}
.mg-cbox-num {{ color: {ACCENT}; font-weight: 800; font-size: 1rem; }}
.mg-cbox-x {{
    color: {TEXT_MUTED}; text-decoration: none; font-size: 1.2rem;
    padding: 4px 10px; border-radius: 8px; transition: background .15s;
}}
.mg-cbox-x:hover {{ color: {TEXT_DARK}; background: {SURFACE}; }}
.mg-cbox-title {{ color: {PRIMARY}; font-weight: 800; font-size: 1.05rem; margin-bottom: 0.3rem; }}
.mg-cbox-src   {{ color: {TEXT_MUTED}; font-size: 0.75rem; margin-bottom: 1rem; }}
.mg-cbox-body  {{
    color: {TEXT_MAIN}; font-size: 0.9rem; line-height: 1.7;
    background: {SURFACE}; padding: 1.1rem; border-radius: 10px;
    border-left: 4px solid {ACCENT};
    white-space: pre-wrap; word-break: break-word;
    max-height: 500px; overflow-y: auto;
}}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {{
    background: {SURFACE}; border-right: 1px solid {BORDER};
}}
/* Fix for centering icons (Emoji) in small buttons in sidebar */
[data-testid="stSidebar"] .stButton button {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    height: 30px !important;
    min-width: 30px !important;
    width: 100% !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all .18s !important;
}}
[data-testid="stSidebar"] .stButton button p {{
    margin: 0 !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
[data-testid="stSidebar"] .stButton > button:disabled {{
    background-color: rgba(11, 61, 145, 0.08) !important;
    color: {PRIMARY} !important;
    border: 1px solid rgba(11, 61, 145, 0.2) !important;
    opacity: 1 !important; cursor: default !important;
}}

/* ── Input Styling ── */
.stTextInput input, .stSelectbox div[data-baseweb="select"] {{
    border-radius: 8px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Tabs & Progress ── */
.stTabs [aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT}) !important;
}}

/* ── Disclaimer ── */
.mg-disclaimer {{
    font-size: 0.8rem; color: {TEXT_MUTED};
    margin-top: 0.1rem; margin-bottom: 0.2rem;
    text-align: left;
}}

/* ── Entity result chips (Graph Explorer) ── */
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


def render_brand(status_html: str = "") -> None:
    """Render the sidebar brand block: logo + wordmark + optional status."""
    st.markdown(
        f"""
        <div class="mg-brand">
          <div class="mg-brand-logo">{LOGO_SVG_INLINE}</div>
          <div class="mg-wordmark">
            <div class="mg-name">MedGraphia</div>
            <div class="mg-tag">Medical GraphRAG</div>
          </div>
          {status_html}
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


def connection_pill(online: bool, label_ok: str = "Connected", label_off: str = "Offline") -> str:
    """Return HTML for an animated connection pill (sidebar)."""
    cls = "mg-online" if online else "mg-offline"
    txt = label_ok if online else label_off
    return f'<span class="mg-pill {cls}"><span class="mg-pill-dot"></span>{txt}</span>'
