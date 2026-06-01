"""
OCR fallback parser using Tesseract 5 + (optional) PaddleOCR for scanned PDFs.
Used when Docling or MinerU fail or when the PDF is image-only (no text layer).

Tesseract is auto-selected based on detected language.
PaddleOCR is used for Chinese documents (better ZH accuracy than Tesseract).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from medgraphia.domain import Language, ParsedSection, RawDocument, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Tesseract language codes
_TESS_LANG_MAP = {
    Language.EN: "eng",
    Language.DE: "deu",
    Language.ZH: "chi_sim",
}


class OCRParser:
    """
    Fallback OCR parser for scanned PDFs and images.
    Tries Tesseract first; falls back to PaddleOCR for Chinese if available.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: str,
        source_meta: SourceMeta,
        language: Language = Language.EN,
        dpi: int = 300,
    ) -> RawDocument:
        """Convert a scanned PDF to a RawDocument using OCR."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("ocr_parsing", file=str(path), language=language.value)

        if language == Language.ZH and _paddle_available():
            sections = self._paddle_parse(path, dpi)
        else:
            sections = self._tesseract_parse(path, language, dpi)

        full_text = "\n\n".join(s.content for s in sections if s.content)
        logger.info("ocr_parsed", file=str(path), chars=len(full_text))

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
    # Tesseract backend
    # ------------------------------------------------------------------

    def _tesseract_parse(
        self,
        path: Path,
        language: Language,
        dpi: int,
    ) -> list[ParsedSection]:
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image
        except ImportError as e:
            raise ImportError(f"Install pytesseract, pdf2image, Pillow: {e}")

        lang_code = _TESS_LANG_MAP.get(language, "eng")
        images = convert_from_path(str(path), dpi=dpi)
        sections: list[ParsedSection] = []

        for page_num, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img, lang=lang_code).strip()
            if text:
                sections.append(
                    ParsedSection(
                        section_path=f"Page {page_num}",
                        title=f"Page {page_num}",
                        content=text,
                        page_start=page_num,
                        page_end=page_num,
                    )
                )

        logger.debug("tesseract_done", pages=len(images), sections=len(sections))
        return sections

    # ------------------------------------------------------------------
    # PaddleOCR backend (Chinese)
    # ------------------------------------------------------------------

    def _paddle_parse(self, path: Path, dpi: int) -> list[ParsedSection]:
        try:
            from paddleocr import PaddleOCR
            from pdf2image import convert_from_path
        except ImportError as e:
            logger.warning("paddle_import_failed", error=str(e))
            return self._tesseract_parse(path, Language.ZH, dpi)

        ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        images = convert_from_path(str(path), dpi=dpi)
        sections: list[ParsedSection] = []

        for page_num, img in enumerate(images, start=1):
            import numpy as np
            img_array = np.array(img)
            result = ocr_engine.ocr(img_array, cls=True)
            if not result or not result[0]:
                continue
            lines = [line[1][0] for line in result[0] if line and line[1]]
            text = "\n".join(lines).strip()
            if text:
                sections.append(
                    ParsedSection(
                        section_path=f"Page {page_num}",
                        title=f"Page {page_num}",
                        content=text,
                        page_start=page_num,
                        page_end=page_num,
                    )
                )

        logger.debug("paddleocr_done", pages=len(images), sections=len(sections))
        return sections


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _paddle_available() -> bool:
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False
