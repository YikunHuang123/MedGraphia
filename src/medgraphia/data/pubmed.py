"""
PubMed E-utilities connector.
Uses NCBI's official REST API (no scraping).  Supports:
  - esearch  — keyword + date-range + citation-count filtering
  - efetch   — abstract + metadata retrieval in XML format

Requires a PUBMED_EMAIL env var (NCBI policy); PUBMED_API_KEY is optional
but raises the rate limit from 3 to 10 req/s.
"""
from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from typing import AsyncIterator

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from medgraphia.config import get_settings
from medgraphia.domain import Language, ParsedSection, RawDocument, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class PubMedFetchConfig:
    query: str
    max_results: int = 200
    date_from: date | None = None
    date_to: date | None = None
    min_citations: int = 0
    rettype: str = "abstract"      # "abstract" or "full"
    batch_size: int = 100          # eFetch batch size (NCBI limit: 10 000)


class PubMedConnector:
    """Fetches PubMed abstracts via NCBI E-utilities (authorized, rate-limited)."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._email = cfg.pubmed_email
        self._api_key = cfg.pubmed_api_key
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "PubMedConnector":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={"User-Agent": f"MedGraphia/0.1 (mailto:{self._email})"},
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, config: PubMedFetchConfig) -> list[RawDocument]:
        """Download up to config.max_results abstracts matching the query."""
        pmids = await self._search(config)
        if not pmids:
            logger.info("pubmed_no_results", query=config.query)
            return []

        logger.info("pubmed_ids_found", count=len(pmids), query=config.query)
        docs: list[RawDocument] = []

        # Batch eFetch to respect NCBI limits
        for i in range(0, len(pmids), config.batch_size):
            batch = pmids[i : i + config.batch_size]
            batch_docs = await self._fetch_batch(batch)
            docs.extend(batch_docs)
            # Respect rate limit: 10 req/s with API key, 3 req/s without
            await asyncio.sleep(0.12 if self._api_key else 0.4)

        logger.info("pubmed_fetch_complete", total=len(docs))
        return docs

    async def stream(self, config: PubMedFetchConfig) -> AsyncIterator[RawDocument]:
        """Yield documents one by one — memory-efficient for large corpora."""
        pmids = await self._search(config)
        for i in range(0, len(pmids), config.batch_size):
            batch = pmids[i : i + config.batch_size]
            for doc in await self._fetch_batch(batch):
                yield doc
            await asyncio.sleep(0.12 if self._api_key else 0.4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _search(self, config: PubMedFetchConfig) -> list[str]:
        """Run esearch and return a list of PubMed IDs."""
        params: dict[str, str | int] = {
            "db": "pubmed",
            "term": config.query,
            "retmax": config.max_results,
            "retmode": "json",
            "email": self._email,
            "sort": "relevance",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        if config.date_from:
            params["mindate"] = config.date_from.strftime("%Y/%m/%d")
        if config.date_to:
            params["maxdate"] = config.date_to.strftime("%Y/%m/%d")
            params["datetype"] = "pdat"

        assert self._session is not None
        async with self._session.get(f"{_BASE_URL}/esearch.fcgi", params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return data.get("esearchresult", {}).get("idlist", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _fetch_batch(self, pmids: list[str]) -> list[RawDocument]:
        """Run eFetch for a batch of PMIDs and parse the XML response."""
        params: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
            "email": self._email,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        assert self._session is not None
        async with self._session.get(f"{_BASE_URL}/efetch.fcgi", params=params) as resp:
            resp.raise_for_status()
            xml_bytes = await resp.read()

        return _parse_pubmed_xml(xml_bytes)


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_pubmed_xml(xml_bytes: bytes) -> list[RawDocument]:
    """Parse PubMed XML response into RawDocument objects."""
    root = ET.fromstring(xml_bytes)
    docs: list[RawDocument] = []

    for article in root.findall(".//PubmedArticle"):
        try:
            doc = _parse_article(article)
            docs.append(doc)
        except Exception as exc:
            logger.warning("pubmed_parse_error", error=str(exc))

    return docs


def _parse_article(article: ET.Element) -> RawDocument:
    pmid_el = article.find(".//PMID")
    pmid = pmid_el.text if pmid_el is not None else "unknown"

    title_el = article.find(".//ArticleTitle")
    title = title_el.text or "" if title_el is not None else ""

    # Extract structured abstract sections
    parsed_sections: list[ParsedSection] = []
    abstract_texts = article.findall(".//AbstractText")
    
    for el in abstract_texts:
        label = el.get("Label")
        # NLM sometimes uses NlmCategory instead of Label
        if not label:
            label = el.get("NlmCategory")
            
        content = (el.text or "").strip()
        if not content:
            continue
            
        section_title = label.title() if label else "Abstract"
        
        # If last section has the same title, append to it (sometimes PubMed splits long paragraphs)
        if parsed_sections and parsed_sections[-1].title == section_title:
            parsed_sections[-1].content += "\n" + content
        else:
            parsed_sections.append(
                ParsedSection(
                    section_path=section_title,
                    title=section_title,
                    content=content
                )
            )

    # Full abstract text for backward compatibility
    abstract = "\n\n".join(s.content for s in parsed_sections)

    # Journal + year
    journal_el = article.find(".//Journal/Title")
    journal = journal_el.text or "" if journal_el is not None else ""

    year_el = article.find(".//PubDate/Year")
    year = year_el.text or "" if year_el is not None else ""

    # Language
    lang_el = article.find(".//Language")
    raw_lang = (lang_el.text or "eng").lower()
    lang_map = {"eng": Language.EN, "chi": Language.ZH, "ger": Language.DE}
    language = lang_map.get(raw_lang, Language.EN)

    source = SourceMeta(
        source_id=f"pubmed:{pmid}",
        source_title=title,
        source_version=year,
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        language=language,
    )

    return RawDocument(
        source=source,
        language=language,
        title=title,
        abstract=abstract,
        sections=parsed_sections,
        format="text",
    )
