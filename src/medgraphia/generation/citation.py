"""
Citation injection for MedGraphia.

Workflow:
1.  The retrieval pipeline returns a list of FusedItems (ranked context passages).
2.  build_numbered_context() formats them as "[1] text…\n\n[2] text…" for the LLM.
3.  The LLM generates an answer that contains [N] inline references.
4.  inject_citations() parses those [N] markers, maps each integer back to the
    corresponding FusedItem (index N-1), and builds a Citation object from the
    item's metadata.

Citation fields populated from FusedItem.metadata:
  chunk_id        ← metadata["chunk_id"]       (Vector / Graph path)
  source_title    ← metadata["source_title"]   or community_id / item_id as fallback
  source_version  ← metadata["source_version"]
  section_path    ← metadata["section_path"]
  content_snippet ← first 200 chars of item.text  (for UI hover cards)
  citation_number ← the [N] integer from the answer

Unresolvable citation numbers (e.g. the LLM hallucinated [99] but only 5 passages
were given) are collected in CitationResult.unresolved for observability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medgraphia.domain.chat import Citation
from medgraphia.logger import get_logger
from medgraphia.retrieval.fusion import FusedItem

logger = get_logger(__name__)

# Matches "[1]", "[12]", but NOT "[abc]" or "[]"
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Maximum content_snippet length stored in the Citation object (for UI previews)
_SNIPPET_MAX = 500


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class CitationResult:
    """
    A single LLM response enriched with resolved Citation objects.

    Attributes:
        answer_text  — Original LLM response (citation markers preserved).
        citations    — Resolved Citation objects, ordered by first appearance in text.
        unresolved   — [N] numbers that could not be mapped to a context item.
    """

    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    unresolved: list[int] = field(default_factory=list)

    @property
    def is_fully_resolved(self) -> bool:
        return len(self.unresolved) == 0

    def as_dict(self) -> dict:
        return {
            "answer_text": self.answer_text,
            "citations": [c.model_dump() for c in self.citations],
            "unresolved": self.unresolved,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inject_citations(
    answer_text: str,
    context_items: list[FusedItem],
    explicit_citations: list[int] | None = None,
) -> CitationResult:
    """
    Resolve [N] markers and explicit citation numbers to Citation objects.

    Args:
        answer_text:        LLM-generated response text containing [N] references.
        context_items:      Ordered list of FusedItems presented to the LLM.
        explicit_citations: List of integer citations extracted from the JSON response
                            (e.g., from MedicalAnswer.citations). Provides a robust
                            fallback if regex fails due to LLM formatting quirks.

    Returns:
        CitationResult with fully populated Citation objects and any unresolved indices.
    """
    if not answer_text and not explicit_citations:
        return CitationResult(answer_text=answer_text)

    # ── Step 1: collect unique citation numbers ──
    seen_nums: list[int] = []
    seen_set: set[int] = set()

    # First priority: explicit structured citations from the LLM
    if explicit_citations:
        for n in explicit_citations:
            if n not in seen_set:
                seen_nums.append(n)
                seen_set.add(n)

    # Second priority: regex extraction from the text (in case the array was incomplete)
    for m in _CITATION_RE.finditer(answer_text):
        n = int(m.group(1))
        if n not in seen_set:
            seen_nums.append(n)
            seen_set.add(n)

    if not seen_nums:
        logger.debug("citation_inject_no_markers", text_len=len(answer_text))
        return CitationResult(answer_text=answer_text)

    # ── Step 2: map each number to a FusedItem ────────────────────────────
    citations: list[Citation] = []
    unresolved: list[int] = []

    for n in seen_nums:
        idx = n - 1  # 1-based → 0-based
        if idx < 0 or idx >= len(context_items):
            unresolved.append(n)
            logger.warning(
                "citation_unresolved",
                number=n,
                context_size=len(context_items),
            )
            continue

        item = context_items[idx]
        cit = _item_to_citation(n, item)
        citations.append(cit)
        logger.debug(
            "citation_resolved",
            number=n,
            source=cit.source_title,
            chunk_id=cit.chunk_id,
            section=cit.section_path,
        )

    return CitationResult(
        answer_text=answer_text,
        citations=citations,
        unresolved=unresolved,
    )


def build_numbered_context(
    items: list[FusedItem],
    max_chars_per_item: int = 3000,
) -> str:
    """
    Format *items* into the numbered context string that gets injected into prompts.

    Output format::

        [1] <text of item 0, truncated to max_chars_per_item>

        [2] <text of item 1>

        …

    The 1-based numbering matches the [N] citation convention used in prompts.py.

    Args:
        items:              Ordered FusedItems from the retrieval pipeline.
        max_chars_per_item: Hard character limit per item (default 3000).

    Returns:
        Multi-line string suitable for the ``context`` input field of any predictor.
    """
    if not items:
        return ""
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        text = item.text[:max_chars_per_item].strip()
        # Append source attribution hint after the text
        source_hint = _source_hint(item)
        line = f"[{i}] {text}"
        if source_hint:
            line += f"  ({source_hint})"
        lines.append(line)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item_to_citation(number: int, item: FusedItem) -> Citation:
    """Build a Citation object from a citation number + FusedItem."""
    meta = item.metadata

    chunk_id = _str_or_empty(meta.get("chunk_id"))
    source_title = _str_or_empty(meta.get("source_title"))
    section_path = _str_or_empty(meta.get("section_path"))
    community_id = _str_or_empty(meta.get("community_id"))

    # Logic: Prioritize Title + Section for context clarity
    if source_title and section_path:
        raw_title = f"{source_title} › {section_path}"
    elif source_title:
        raw_title = source_title
    elif community_id:
        raw_title = f"Medical Community {community_id}"
    else:
        # Avoid showing raw UUIDs to the user
        raw_title = "Unstructured Medical Context"

    # Clean up repetitive FDA/EMA boilerplate from titles
    clean_title = _clean_title(raw_title)

    # Add source type prefix for better intuition (e.g. "[Vector] Metformin...")
    source_type_label = f"[{item.source.value.capitalize()}]"
    final_title = f"{source_type_label} {clean_title}"

    source_version = _str_or_empty(meta.get("source_version"))
    content_snippet = item.text[:_SNIPPET_MAX].strip()

    return Citation(
        citation_number=number,
        source_title=final_title,
        source_version=source_version,
        section_path=section_path,
        content_snippet=content_snippet,
        chunk_id=chunk_id,
    )


def _clean_title(title: str) -> str:
    """Remove boilerplate 'highlights' text and truncate overly long titles."""
    if not title:
        return "Unknown Source"

    # Remove common FDA boilerplate strings
    noise = [
        "These highlights do not include all the information needed to use",
        "safely and effectively. See full prescribing information for",
        "HIGHLIGHTS OF PRESCRIBING INFORMATION",
    ]
    for n in noise:
        title = title.replace(n, "")

    # Remove double spaces, trailing dots or dashes
    title = " ".join(title.split())
    title = title.strip(". -")

    # If it's still empty, provide a fallback
    if not title:
        return "Medical Reference"

    # Hard truncation for UI sanity
    if len(title) > 100:
        title = title[:97] + "..."
    return title


def _source_hint(item: FusedItem) -> str:
    """Short attribution label appended to each context line."""
    meta = item.metadata
    title = _str_or_empty(meta.get("source_title"))
    section = _str_or_empty(meta.get("section_path"))
    if title and section:
        return f"{title} › {section}"
    if title:
        return title
    return item.source.value


def _str_or_empty(val: object) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return s if s != "None" else ""
