"""
Citation rendering helpers.

`render_message_with_citations()` converts `[N]` tokens inside the answer
text into clickable links that open a CSS `:target` modal — no JavaScript
required. The modal HTML for every citation is inlined alongside the
answer; clicking `[N]` jumps to its modal anchor, clicking the backdrop
or the close button jumps back to a sentinel anchor and the modal closes.

The pattern is borrowed from the Neat-RAG reference UI and adapted to
the MedGraphia design palette (see styles.py).
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

import streamlit as st

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _safe(value: Any) -> str:
    """HTML-escape any value, coalescing None to empty string."""
    return _html.escape(str(value or ""))


def render_message_with_citations(
    content: str,
    citations: list[dict[str, Any]],
    message_key: str,
) -> None:
    """Render `content` with `[N]` replaced by clickable refs + modal popups.

    Falls back to plain markdown when there are no citations.
    """
    if not citations:
        st.markdown(content)
        return

    cite_map = {
        int(c.get("number", c.get("citation_number", 0))): c
        for c in citations
        if c.get("number") or c.get("citation_number")
    }
    if not cite_map:
        # Citations exist but have no usable numbers — render them by order.
        cite_map = {i + 1: c for i, c in enumerate(citations)}

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        if n not in cite_map:
            return m.group(0)
        return f'<a href="#mg-cm-{message_key}-{n}" class="mg-cref">[{n}]</a>'

    text_html = _CITATION_RE.sub(_replace, content)
    close_id = f"mg-cc-{message_key}"
    close_anchor = f'<span id="{close_id}"></span>'

    modals: list[str] = []
    for n in sorted(cite_map):
        c = cite_map[n]
        title = _safe(c.get("source_title") or c.get("chunk_id") or f"Source {n}")
        source = _safe(
            f"version: {c.get('source_version', 'n/a')}   ·   section: {c.get('section_path', '')}"
        )
        snippet = _safe(c.get("content_snippet") or c.get("text") or "")
        modals.append(
            f'<div id="mg-cm-{message_key}-{n}" class="mg-covl">'
            f'<a href="#{close_id}" class="mg-covl-bg"></a>'
            f'<div class="mg-cbox">'
            f'<div class="mg-cbox-hdr">'
            f'<span class="mg-cbox-num">Citation [{n}]</span>'
            f'<a href="#{close_id}" class="mg-cbox-x">×</a>'
            f"</div>"
            f'<div class="mg-cbox-title">{title}</div>'
            f'<div class="mg-cbox-src">{source}</div>'
            f'<div class="mg-cbox-body">{snippet}</div>'
            f"</div></div>"
        )

    # Wrap the answer text in a paragraph so markdown line-breaks survive.
    body = text_html.replace("\n", "<br>")
    st.markdown(close_anchor + body + "".join(modals), unsafe_allow_html=True)


def render_citation_cards(citations: list[dict[str, Any]]) -> None:
    """Render a flat, expandable list of citation cards beneath an answer."""
    if not citations:
        return
    with st.expander(f"Sources  ·  {len(citations)} citation(s)", expanded=False):
        for i, c in enumerate(citations, start=1):
            n = c.get("number") or c.get("citation_number") or i
            title = _safe(c.get("source_title") or c.get("chunk_id") or f"Source {n}")
            version = _safe(c.get("source_version") or "n/a")
            section = _safe(c.get("section_path") or "")
            snippet = _safe(c.get("content_snippet") or c.get("text") or "")
            meta = f"version: {version}" + (f"   ·   section: {section}" if section else "")
            st.markdown(
                f"""
                <div class="mg-cite">
                  <span class="mg-cite-num">[{n}]</span>
                  <div style="flex:1; min-width:0;">
                    <div class="mg-cite-title">{title}</div>
                    <div class="mg-cite-meta">{meta}</div>
                    <div class="mg-cite-snippet">{snippet}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
