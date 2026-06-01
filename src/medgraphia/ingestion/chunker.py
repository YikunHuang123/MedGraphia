"""
Section-aware medical document chunker.

Strategy:
  - Documents with .sections: chunk per section, split large sections by paragraph
  - Documents without sections: split abstract + full_text by paragraph → sentence
  - Every Chunk carries section_path + doc provenance for full audit traceability

Token estimation is intentionally lightweight (no ML tokenizer dependency)
EN/DE: word_count × 1.3 (BPE inflation approximation)
ZH:    cjk_char_count × 0.75  (CJK char ≈ 1.3 BPE tokens on BGE-M3)
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from medgraphia.domain import Chunk, Language, ParsedSection, RawDocument
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# _DEFAULT_MAX_TOKENS: int = 512
_DEFAULT_MAX_TOKENS: int = 300  # GLiNER model can only handle 384 tokens max
_DEFAULT_OVERLAP: int = 50


class MedicalChunker:
    """
    Splits a RawDocument into Chunk objects preserving section_path provenance.

    Usage::

        chunker = MedicalChunker(max_tokens=512, overlap_tokens=50)
        chunks  = chunker.chunk(doc)
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        overlap_tokens: int = _DEFAULT_OVERLAP,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        """Return all Chunks for a RawDocument."""
        chunks: list[Chunk] = []

        if doc.sections:
            for section in doc.sections:
                chunks.extend(self._chunk_section(section, doc))
        else:
            # Fall back to abstract and/or full_text
            text_parts: list[tuple[str, str]] = []
            if doc.abstract.strip():
                text_parts.append(("Abstract", doc.abstract))
            if doc.full_text.strip() and doc.full_text.strip() != doc.abstract.strip():
                text_parts.append(("Body", doc.full_text))
            for section_name, text in text_parts:
                chunks.extend(
                    self._chunk_plain_text(text, doc, section_path=section_name)
                )

        logger.info(
            "chunking_complete",
            doc_id=doc.doc_id,
            sections=len(doc.sections),
            chunks=len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Section-based chunking
    # ------------------------------------------------------------------

    def _chunk_section(self, section: ParsedSection, doc: RawDocument) -> list[Chunk]:
        if not section.content.strip():
            return []

        tok = self._estimate_tokens(section.content, doc.language)
        if tok <= self.max_tokens:
            return [
                Chunk(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    language=doc.language,
                    section_path=section.section_path,
                    text=section.content.strip(),
                    token_count=tok,
                    page=section.page_start,
                )
            ]

        return list(
            self._split_text_into_chunks(
                text=section.content,
                doc=doc,
                section_path=section.section_path,
                base_page=section.page_start,
            )
        )

    # ------------------------------------------------------------------
    # Plain-text chunking (no sections available)
    # ------------------------------------------------------------------

    def _chunk_plain_text(
        self,
        text: str,
        doc: RawDocument,
        section_path: str = "",
    ) -> list[Chunk]:
        if not text.strip():
            return []
        return list(
            self._split_text_into_chunks(
                text=text,
                doc=doc,
                section_path=section_path,
            )
        )

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split_text_into_chunks(
        self,
        text: str,
        doc: RawDocument,
        section_path: str,
        base_page: int | None = None,
    ) -> Iterator[Chunk]:
        """
        Greedily pack paragraphs into chunks up to max_tokens.
        Carries a small overlap window into the next chunk for context continuity.
        Paragraphs that individually exceed max_tokens are split by sentence.
        """
        paragraphs = _split_paragraphs(text)
        buffer: list[str] = []
        buffer_tokens: int = 0

        def _flush() -> Chunk:
            chunk_text = "\n\n".join(buffer).strip()
            return Chunk(
                doc_id=doc.doc_id,
                source=doc.source,
                language=doc.language,
                section_path=section_path,
                text=chunk_text,
                token_count=self._estimate_tokens(chunk_text, doc.language),
                page=base_page,
            )

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._estimate_tokens(para, doc.language)

            # Paragraph is already larger than one chunk — split by sentence
            if para_tokens > self.max_tokens:
                if buffer:
                    yield _flush()
                    buffer = _overlap_tail(buffer, self.overlap_tokens, doc.language, self._estimate_tokens)
                    buffer_tokens = sum(self._estimate_tokens(p, doc.language) for p in buffer)
                yield from self._split_by_sentence(para, doc, section_path, base_page)
                continue

            # Adding this paragraph would overflow — flush first
            if buffer and buffer_tokens + para_tokens > self.max_tokens:
                yield _flush()
                buffer = _overlap_tail(buffer, self.overlap_tokens, doc.language, self._estimate_tokens)
                buffer_tokens = sum(self._estimate_tokens(p, doc.language) for p in buffer)

            buffer.append(para)
            buffer_tokens += para_tokens

        if buffer:
            yield _flush()

    def _split_by_sentence(
        self,
        paragraph: str,
        doc: RawDocument,
        section_path: str,
        base_page: int | None,
    ) -> Iterator[Chunk]:
        """Split a single oversized paragraph at sentence boundaries."""
        sentences = _split_sentences(paragraph, doc.language)
        buffer: list[str] = []
        buffer_tokens: int = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = self._estimate_tokens(sent, doc.language)

            if buffer and buffer_tokens + sent_tokens > self.max_tokens:
                chunk_text = " ".join(buffer).strip()
                yield Chunk(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    language=doc.language,
                    section_path=section_path,
                    text=chunk_text,
                    token_count=self._estimate_tokens(chunk_text, doc.language),
                    page=base_page,
                )
                buffer = _overlap_tail(buffer, self.overlap_tokens, doc.language, self._estimate_tokens)
                buffer_tokens = sum(self._estimate_tokens(s, doc.language) for s in buffer)

            buffer.append(sent)
            buffer_tokens += sent_tokens

        if buffer:
            chunk_text = " ".join(buffer).strip()
            yield Chunk(
                doc_id=doc.doc_id,
                source=doc.source,
                language=doc.language,
                section_path=section_path,
                text=chunk_text,
                token_count=self._estimate_tokens(chunk_text, doc.language),
                page=base_page,
            )

    # ------------------------------------------------------------------
    # Token estimation (no tokenizer dependency)
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str, language: Language = Language.EN) -> int:
        if not text:
            return 0
        if language == Language.ZH:
            cjk = sum(1 for c in text if "一" <= c <= "鿿")
            ascii_words = len(re.findall(r"[A-Za-z0-9]+", text))
            return max(1, int(cjk * 0.75 + ascii_words * 1.3))
        # EN / DE / UNKNOWN — word-count × BPE inflation factor
        return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Text splitting helpers (module-level, importable for tests)
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split on one or more blank lines."""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _split_sentences(text: str, language: Language = Language.EN) -> list[str]:
    """
    Regex-based sentence splitter.  Accurate enough for the NER pipeline without
    requiring spaCy / NLTK (avoids adding heavy startup deps to the build path).

    ZH: split on Chinese terminal punctuation 。？！
    EN/DE: split on .  !  ?  followed by whitespace + capital / digit / CJK
    """
    if language == Language.ZH:
        parts = re.split(r"(?<=[。？！…])\s*", text)
    else:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÜÖÄ一-鿿\d])", text)
    return [p.strip() for p in parts if p.strip()]


def _overlap_tail(
    items: list[str],
    overlap_tokens: int,
    language: Language,
    token_fn,
) -> list[str]:
    """
    Return the trailing items of `items` whose cumulative token count fits
    within `overlap_tokens`.  These are carried forward as context for the
    next chunk.
    """
    result: list[str] = []
    total = 0
    for item in reversed(items):
        tok = token_fn(item, language)
        if total + tok > overlap_tokens:
            break
        result.insert(0, item)
        total += tok
    return result
