"""
MinerU-based PDF parser for Chinese academic documents.
MinerU is optimised for double-column Chinese layouts, formulas, and academic PDFs.
It produces a structured JSON (similar to Docling) with section hierarchy.
"""

from __future__ import annotations

from pathlib import Path

from medgraphia.domain import Language, ParsedSection, RawDocument, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)


class MinerUParser:
    """
    Wraps the MinerU PDF parser for Chinese medical PDFs.
    Falls back to OCRParser if MinerU is not installed.
    """

    def __init__(self) -> None:
        self._available: bool | None = None

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                import magic_pdf  # noqa: F401

                self._available = True
            except ImportError:
                logger.warning(
                    "mineru_not_installed",
                    hint="pip install magic-pdf[full]",
                )
                self._available = False
        return self._available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: str,
        source_meta: SourceMeta,
        language: Language = Language.ZH,
    ) -> RawDocument:
        """
        Parse a Chinese PDF with MinerU.
        Falls back to OCRParser (PaddleOCR) if MinerU is unavailable.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if not self._check_available():
            logger.info("mineru_fallback_ocr", file=str(path))
            from medgraphia.ingestion.parsers.ocr_parser import OCRParser

            return OCRParser().parse(file_path, source_meta, language)

        logger.info("mineru_parsing", file=str(path))
        sections, full_text = self._run_mineru(path)

        logger.info("mineru_parsed", file=str(path), sections=len(sections), chars=len(full_text))
        return RawDocument(
            source=source_meta,
            language=language,
            title=source_meta.source_title,
            full_text=full_text,
            sections=sections,
            file_path=str(path),
            format="pdf",
        )

    # ------------------------------------------------------------------
    # MinerU internals
    # ------------------------------------------------------------------

    def _run_mineru(self, path: Path) -> tuple[list[ParsedSection], str]:
        """Run MinerU conversion and extract sections from the output JSON."""
        from magic_pdf.config.make_content_config import DropMode
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

        # MinerU writes output to a temp directory; we read it back
        output_dir = path.parent / (path.stem + "_mineru_out")
        output_dir.mkdir(exist_ok=True)

        pdf_bytes = path.read_bytes()
        ds = PymuDocDataset(pdf_bytes)

        # Auto-detect pipeline (OCR vs text) based on page content
        model_json = doc_analyze(ds, ocr=False)  # set ocr=True for scanned docs

        writer = FileBasedDataWriter(str(output_dir))
        pipe = ds.apply(
            doc_analyze,
            model_json=model_json,
            drop_mode=DropMode.NONE,
        )
        pipe.dump_markdown(writer, md_writer=writer, img_dir=str(output_dir / "images"))

        # Read the generated markdown back
        md_files = list(output_dir.glob("*.md"))
        if not md_files:
            logger.warning("mineru_no_md_output", file=str(path))
            return [], ""

        md_text = md_files[0].read_text(encoding="utf-8")
        sections = _parse_markdown_sections(md_text)
        full_text = md_text
        return sections, full_text


# ---------------------------------------------------------------------------
# Markdown → sections (works for both MinerU and generic MD)
# ---------------------------------------------------------------------------


def _parse_markdown_sections(md_text: str) -> list[ParsedSection]:
    """
    Parse a Markdown string into ParsedSection objects.
    MinerU outputs ATX headings (# / ## / ###) as section headers.
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
        if line.startswith("#"):
            flush()
            current_lines = []
            level = len(line) - len(line.lstrip("#"))
            heading = line.lstrip("#").strip()
            current_path = current_path[: level - 1] + [heading]
        else:
            current_lines.append(line)

    flush()
    return sections
