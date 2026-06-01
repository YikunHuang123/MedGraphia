"""
Docling-based PDF parser for English and German medical documents.
Docling preserves:
  - Document section hierarchy → section_path metadata
  - Tables as structured JSON
  - Figure captions
  - Formula layout (LaTeX)

Install: pip install docling
Docling documentation: https://ds4sd.github.io/docling/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medgraphia.domain import Language, ParsedSection, RawDocument, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)


class DoclingParser:
    """
    Wraps the Docling PDF-to-document converter.
    Produces a RawDocument with a populated .sections list.
    """

    def __init__(self) -> None:
        # Lazy import — Docling is a heavy dependency (~500 MB models)
        self._converter: Any = None

    def _get_converter(self) -> Any:
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
            except ImportError:
                raise ImportError(
                    "Docling not installed.  Run: pip install docling"
                )
        return self._converter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: str,
        source_meta: SourceMeta,
        language: Language = Language.EN,
    ) -> RawDocument:
        """
        Parse a PDF at file_path using Docling.
        Returns a RawDocument with high-quality Markdown content.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info("docling_parsing", file=str(path))
        converter = self._get_converter()
        result = converter.convert(str(path))

        # Modern Docling: export to Markdown (includes tables, structure)
        full_text = result.document.export_to_markdown()
        
        # Split markdown into structured sections
        sections = _parse_markdown_sections(full_text)

        logger.info(
            "docling_parsed",
            file=str(path),
            chars=len(full_text),
            sections=len(sections),
        )
        return RawDocument(
            source=source_meta,
            language=language,
            title=source_meta.source_title,
            full_text=full_text,
            sections=sections,
            file_path=str(path),
            format="pdf",
        )

    def parse_to_json(self, file_path: str) -> dict:
        """Raw export: returns the Docling DoclingDocument as a JSON-serialisable dict."""
        converter = self._get_converter()
        result = converter.convert(file_path)
        return result.document.export_to_dict()


# ---------------------------------------------------------------------------
# Section extraction from Docling JSON / Markdown
# ---------------------------------------------------------------------------

def _parse_markdown_sections(md_text: str) -> list[ParsedSection]:
    """
    Parse a Markdown string into ParsedSection objects based on headings.
    Works for Docling and MinerU outputs.
    """
    sections: list[ParsedSection] = []
    current_path: list[str] = []
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    ParsedSection(
                        section_path=" > ".join(current_path) if current_path else "Preamble",
                        title=current_path[-1] if current_path else "",
                        content=content,
                    )
                )

    for line in md_text.splitlines():
        # Look for ATX headers like # Title, ## Subtitle
        if line.startswith("#"):
            flush()
            current_lines = []
            # Calculate heading level (number of #)
            level = 0
            for char in line:
                if char == "#":
                    level += 1
                else:
                    break
            
            heading = line.lstrip("#").strip()
            # Update current path based on level
            current_path = current_path[: level - 1] + [heading]
        else:
            current_lines.append(line)

    flush()
    return sections


def _extract_sections(doc_json: dict) -> list[ParsedSection]:
    """
    Walk the Docling document JSON and build ParsedSection objects.
    Docling represents structure as a list of 'body' elements with 'prov' (provenance)
    and 'label' (e.g. 'section_header', 'text', 'table').
    """
    sections: list[ParsedSection] = []
    current_path: list[str] = []
    current_content: list[str] = []
    current_tables: list[dict] = []
    current_page_start: int | None = None

    for item in doc_json.get("body", []):
        label: str = item.get("label", "")
        text: str = item.get("text", "")
        prov = item.get("prov", [{}])[0] if item.get("prov") else {}
        page = prov.get("page_no")

        if label == "section_header":
            # Flush previous section
            if current_content or current_tables:
                sections.append(
                    ParsedSection(
                        section_path=" > ".join(current_path) if current_path else "Preamble",
                        title=current_path[-1] if current_path else "",
                        content="\n".join(current_content),
                        page_start=current_page_start,
                        tables=current_tables,
                    )
                )
            # Determine heading level from Docling level field
            level = item.get("level", 1)
            current_path = current_path[: level - 1] + [text.strip()]
            current_content = []
            current_tables = []
            current_page_start = page

        elif label == "text":
            current_content.append(text)
            if current_page_start is None:
                current_page_start = page

        elif label == "table":
            table_data = item.get("data", {})
            current_tables.append(table_data)

    # Flush last section
    if current_content or current_tables:
        sections.append(
            ParsedSection(
                section_path=" > ".join(current_path) if current_path else "Body",
                title=current_path[-1] if current_path else "",
                content="\n".join(current_content),
                page_start=current_page_start,
                tables=current_tables,
            )
        )

    return sections


def _extract_full_text(doc_json: dict) -> str:
    """Concatenate all text items as a simple fallback."""
    parts = []
    for item in doc_json.get("body", []):
        if item.get("text"):
            parts.append(item["text"].strip())
    return "\n\n".join(parts)
