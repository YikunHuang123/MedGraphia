"""
DrugBank connector.
DrugBank provides an official XML bulk export for academic / commercial license holders.
Download from: https://go.drugbank.com/releases/latest (requires free registration)

This module parses the downloaded XML file — it does NOT attempt to scrape or
access DrugBank without a valid license.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from medgraphia.domain import Language, ParsedSection, RawDocument, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_NS = {"db": "http://www.drugbank.ca"}


class DrugBankConnector:
    """
    Parses the official DrugBank full-database XML export.
    Requires the file to be downloaded and placed at xml_path.
    """

    def __init__(self, xml_path: str) -> None:
        self._xml_path = Path(xml_path)
        if not self._xml_path.exists():
            raise FileNotFoundError(
                f"DrugBank XML not found: {xml_path}. "
                "Download it from https://go.drugbank.com/releases/latest"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_drugs(self, limit: int | None = None) -> Iterator[RawDocument]:
        """
        Yield RawDocument objects for each drug entry in the XML.
        Uses iterparse for memory-efficient streaming over the ~1 GB file.
        """
        count = 0
        for event, elem in ET.iterparse(str(self._xml_path), events=("end",)):
            if elem.tag != f"{{{_NS['db']}}}drug":
                continue
            try:
                doc = _parse_drug_element(elem)
                yield doc
                count += 1
            except Exception as exc:
                logger.warning("drugbank_parse_error", error=str(exc))
            finally:
                elem.clear()  # release memory

            if limit is not None and count >= limit:
                break

        logger.info("drugbank_stream_complete", count=count)

    def fetch_all(self, limit: int | None = None) -> list[RawDocument]:
        return list(self.stream_drugs(limit=limit))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _text(elem: ET.Element, path: str) -> str:
    found = elem.find(path, _NS)
    return (found.text or "").strip() if found is not None else ""


def _parse_drug_element(drug: ET.Element) -> RawDocument:
    drugbank_id = _text(drug, "db:drugbank-id[@primary='true']") or _text(drug, "db:drugbank-id")
    name = _text(drug, "db:name")
    
    # Extract structural fields
    fields = [
        ("Description", "db:description"),
        ("Indication", "db:indication"),
        ("Mechanism of Action", "db:mechanism-of-action"),
        ("Pharmacodynamics", "db:pharmacodynamics"),
        ("Toxicity", "db:toxicity"),
        ("Metabolism", "db:metabolism"),
        ("Absorption", "db:absorption"),
    ]
    
    parsed_sections: list[ParsedSection] = []
    for label, path in fields:
        content = _text(drug, path)
        if content:
            parsed_sections.append(
                ParsedSection(
                    section_path=label,
                    title=label,
                    content=content
                )
            )

    full_text = "\n\n".join(f"{s.title}: {s.content}" for s in parsed_sections)

    source = SourceMeta(
        source_id=f"drugbank:{drugbank_id}",
        source_title=f"DrugBank — {name}",
        source_url=f"https://www.drugbank.ca/drugs/{drugbank_id}",
        language=Language.EN,
    )
    return RawDocument(
        source=source,
        language=Language.EN,
        title=name,
        full_text=full_text,
        sections=parsed_sections,
        format="xml",
    )
