"""
FDA DailyMed connector.
Uses the official DailyMed REST API (no scraping):
  https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm

Endpoints used:
  GET /spls.json?drug_name=<name>     — search by drug name → list of SPL set IDs
  GET /spls/<setid>.xml               — download the full SPL XML for a set ID
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from medgraphia.domain import Language, RawDocument, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_API_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


class FDADailyMedConnector:
    """
    Downloads FDA drug label XML from DailyMed.
    All data is publicly available under a U.S. government open data license.
    """

    def __init__(self, output_dir: str = "data/raw/fda_dailymed") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "FDADailyMedConnector":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"User-Agent": "MedGraphia/0.1 (academic research)"},
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_drug(self, drug_name: str, limit: int = 10) -> list[dict]:
        """Search DailyMed for drug labels matching drug_name.  Returns metadata list."""
        params = {"drug_name": drug_name, "pagesize": limit}
        assert self._session is not None
        async with self._session.get(f"{_API_BASE}/spls.json", params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return data.get("data", [])

    async def fetch_label(self, set_id: str, drug_name: str) -> RawDocument | None:
        """Download and parse the SPL XML for a given set_id."""
        assert self._session is not None
        try:
            xml_content = await self._download_xml(set_id)
            doc = _parse_spl_xml(xml_content, set_id, drug_name)
            # Persist raw XML for provenance
            out_path = self._output_dir / f"{set_id}.xml"
            out_path.write_bytes(xml_content)
            doc.file_path = str(out_path)
            logger.info("fda_label_fetched", set_id=set_id, drug=drug_name)
            return doc
        except Exception as exc:
            logger.warning("fda_fetch_failed", set_id=set_id, error=str(exc))
            return None

    async def fetch_by_drug_name(
        self,
        drug_name: str,
        limit: int = 5,
    ) -> list[RawDocument]:
        """Convenience: search + fetch in one call."""
        results = await self.search_drug(drug_name, limit=limit)
        docs: list[RawDocument] = []
        for item in results:
            set_id = item.get("setid", "")
            if not set_id:
                continue
            doc = await self.fetch_label(set_id, drug_name)
            if doc:
                docs.append(doc)
            await asyncio.sleep(0.3)  # polite rate limit
        return docs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _download_xml(self, set_id: str) -> bytes:
        assert self._session is not None
        url = f"{_API_BASE}/spls/{set_id}.xml"
        async with self._session.get(url) as resp:
            resp.raise_for_status()
            return await resp.read()


# ---------------------------------------------------------------------------
# SPL XML parsing
# ---------------------------------------------------------------------------

_SPL_NS = {
    "hl7": "urn:hl7-org:v3",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def _parse_spl_xml(xml_bytes: bytes, set_id: str, drug_name: str) -> RawDocument:
    """Extract title and main sections from an FDA SPL XML document."""
    root = ET.fromstring(xml_bytes)
    ns = "urn:hl7-org:v3"

    title_el = root.find(f".//{{{ns}}}title")
    title = (title_el.text or drug_name) if title_el is not None else drug_name

    sections: list[str] = []
    for section in root.findall(f".//{{{ns}}}section"):
        text_el = section.find(f".//{{{ns}}}text")
        if text_el is not None:
            raw_text = ET.tostring(text_el, encoding="unicode", method="text")
            sections.append(raw_text.strip())

    full_text = "\n\n".join(sections)

    source = SourceMeta(
        source_id=f"dailymed:{set_id}",
        source_title=title,
        source_url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",
        language=Language.EN,
    )
    return RawDocument(
        source=source,
        language=Language.EN,
        title=title,
        full_text=full_text,
        format="xml",
    )
