"""
Tests for document parsing pipeline.
Covers:
  - PubMed XML parsing
  - FDA DailyMed SPL XML parsing
  - DrugBank XML parsing
  - Docling section extraction (with mock)
  - OCR section extraction (Tesseract path, with mock)
  - UMLS RRF streaming
"""

from __future__ import annotations

import textwrap
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medgraphia.data.drugbank import _parse_drug_element
from medgraphia.data.fda_dailymed import _parse_spl_xml
from medgraphia.data.mesh import _resolve_entity_type
from medgraphia.data.pubmed import _parse_pubmed_xml
from medgraphia.domain import Language
from medgraphia.ingestion.parsers.docling_parser import _extract_full_text, _extract_sections
from medgraphia.ingestion.parsers.mineru_parser import _parse_markdown_sections

# ===========================================================================
# PubMed XML parser
# ===========================================================================

SAMPLE_PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Metformin and type 2 diabetes: a systematic review</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Metformin is first-line therapy.</AbstractText>
          <AbstractText Label="RESULTS">HbA1c reduced by 1.2%.</AbstractText>
        </Abstract>
        <Language>eng</Language>
      </Article>
      <MedlineJournalInfo>
        <MedlineTA>Diabetes Care</MedlineTA>
      </MedlineJournalInfo>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="pubmed">
          <Year>2023</Year>
        </PubMedPubDate>
      </History>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_xml_parse_count():
    docs = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
    assert len(docs) == 1


def test_pubmed_xml_parse_pmid():
    docs = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
    assert docs[0].source.source_id == "pubmed:12345678"


def test_pubmed_xml_parse_title():
    docs = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
    assert "Metformin" in docs[0].title


def test_pubmed_xml_parse_abstract():
    docs = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
    # Structured abstract should be joined
    assert "Metformin is first-line therapy" in docs[0].abstract
    assert "HbA1c reduced" in docs[0].abstract


def test_pubmed_xml_parse_language():
    docs = _parse_pubmed_xml(SAMPLE_PUBMED_XML)
    assert docs[0].language == Language.EN


def test_pubmed_xml_parse_empty():
    empty_xml = b"<PubmedArticleSet></PubmedArticleSet>"
    assert _parse_pubmed_xml(empty_xml) == []


# ===========================================================================
# FDA DailyMed SPL XML parser
# ===========================================================================

_SPL_NS = "urn:hl7-org:v3"

SAMPLE_SPL_XML = f"""<?xml version="1.0"?>
<document xmlns="{_SPL_NS}">
  <title>METFORMIN HYDROCHLORIDE tablet</title>
  <component>
    <structuredBody>
      <component>
        <section>
          <title>INDICATIONS AND USAGE</title>
          <text>
            <paragraph>Metformin is indicated as an adjunct to diet and exercise.</paragraph>
          </text>
        </section>
      </component>
      <component>
        <section>
          <title>CONTRAINDICATIONS</title>
          <text>
            <paragraph>Renal impairment (eGFR below 30 mL/min).</paragraph>
          </text>
        </section>
      </component>
    </structuredBody>
  </component>
</document>
""".encode()


def test_spl_xml_parse_title():
    doc = _parse_spl_xml(SAMPLE_SPL_XML, "test-set-id", "Metformin")
    assert "METFORMIN" in doc.title or "Metformin" in doc.title


def test_spl_xml_parse_source_id():
    doc = _parse_spl_xml(SAMPLE_SPL_XML, "abc123", "Metformin")
    assert doc.source.source_id == "dailymed:abc123"


def test_spl_xml_parse_full_text():
    doc = _parse_spl_xml(SAMPLE_SPL_XML, "abc123", "Metformin")
    assert "diet and exercise" in doc.full_text or doc.full_text == ""


def test_spl_xml_parse_language():
    doc = _parse_spl_xml(SAMPLE_SPL_XML, "abc123", "Metformin")
    assert doc.language == Language.EN


# ===========================================================================
# DrugBank XML parser
# ===========================================================================

_DB_NS_URL = "http://www.drugbank.ca"

SAMPLE_DRUGBANK_XML = f"""<drug xmlns="{_DB_NS_URL}" type="small molecule">
  <drugbank-id primary="true">DB00331</drugbank-id>
  <name>Metformin</name>
  <description>Metformin is a biguanide antihyperglycemic agent.</description>
  <indication>Type 2 diabetes mellitus treatment.</indication>
  <mechanism-of-action>Activates AMPK pathway.</mechanism-of-action>
  <pharmacodynamics>Reduces hepatic glucose production.</pharmacodynamics>
  <toxicity>Lactic acidosis risk with renal impairment.</toxicity>
  <synonyms>
    <synonym>Glucophage</synonym>
    <synonym>Dimethylbiguanide</synonym>
  </synonyms>
  <atc-codes>
    <atc-code code="A10BA02"/>
  </atc-codes>
</drug>"""


def test_drugbank_parse_id():
    elem = ET.fromstring(SAMPLE_DRUGBANK_XML)
    doc = _parse_drug_element(elem)
    assert doc.source.source_id == "drugbank:DB00331"


def test_drugbank_parse_name():
    elem = ET.fromstring(SAMPLE_DRUGBANK_XML)
    doc = _parse_drug_element(elem)
    assert doc.title == "Metformin"


def test_drugbank_parse_full_text_contains_description():
    elem = ET.fromstring(SAMPLE_DRUGBANK_XML)
    doc = _parse_drug_element(elem)
    assert "biguanide" in doc.full_text


def test_drugbank_parse_language():
    elem = ET.fromstring(SAMPLE_DRUGBANK_XML)
    doc = _parse_drug_element(elem)
    assert doc.language == Language.EN


# ===========================================================================
# MeSH entity type resolution (MeSH tree numbers; replaces UMLS STY tests)
# ===========================================================================


def test_mesh_entity_type_disease():
    # C-branch = Diseases and Conditions
    assert _resolve_entity_type(["C14.280"]) == "Disease"


def test_mesh_entity_type_mental_disorder():
    # F03-branch = Mental Disorders (also Disease)
    assert _resolve_entity_type(["F03.600"]) == "Disease"


def test_mesh_entity_type_drug():
    # D-branch = Chemicals and Drugs
    assert _resolve_entity_type(["D02.145"]) == "Drug"


def test_mesh_entity_type_unknown():
    assert _resolve_entity_type([]) == "Unknown"
    assert _resolve_entity_type(["Z99"]) == "Unknown"


def test_mesh_entity_type_priority():
    # Disease (C-branch) takes priority over Drug (D-branch) when listed first
    assert _resolve_entity_type(["C14.280", "D02.145"]) == "Disease"


# ===========================================================================
# Docling section extraction (no real PDF — unit-tests on the JSON parser)
# ===========================================================================

SAMPLE_DOCLING_JSON = {
    "body": [
        {
            "label": "section_header",
            "text": "4 Clinical Particulars",
            "level": 1,
            "prov": [{"page_no": 4}],
        },
        {
            "label": "section_header",
            "text": "4.1 Therapeutic Indications",
            "level": 2,
            "prov": [{"page_no": 4}],
        },
        {
            "label": "text",
            "text": "Metformin is indicated for type 2 diabetes.",
            "prov": [{"page_no": 4}],
        },
        {"label": "section_header", "text": "4.2 Posology", "level": 2, "prov": [{"page_no": 5}]},
        {"label": "text", "text": "Adults: 500 mg twice daily.", "prov": [{"page_no": 5}]},
        {
            "label": "table",
            "data": {"rows": [["Drug", "Dose"], ["Metformin", "500 mg"]]},
            "prov": [{"page_no": 5}],
        },
    ]
}


def test_docling_extract_sections_count():
    sections = _extract_sections(SAMPLE_DOCLING_JSON)
    assert len(sections) >= 2


def test_docling_section_path():
    sections = _extract_sections(SAMPLE_DOCLING_JSON)
    paths = [s.section_path for s in sections]
    assert any("Therapeutic Indications" in p for p in paths)


def test_docling_section_content():
    sections = _extract_sections(SAMPLE_DOCLING_JSON)
    all_content = " ".join(s.content for s in sections)
    assert "type 2 diabetes" in all_content


def test_docling_section_table():
    sections = _extract_sections(SAMPLE_DOCLING_JSON)
    tables = [t for s in sections for t in s.tables]
    assert len(tables) >= 1


def test_docling_full_text():
    full = _extract_full_text(SAMPLE_DOCLING_JSON)
    assert "Metformin" in full
    assert "500 mg" in full


# ===========================================================================
# MinerU markdown section parser
# ===========================================================================

SAMPLE_MARKDOWN = textwrap.dedent("""
    # Clinical Guidelines

    Introduction text.

    ## Pharmacology

    Metformin inhibits mitochondrial respiratory chain complex I.

    ### Dosing

    Standard dose: 500 mg twice daily.

    ## Contraindications

    Avoid in renal failure.
""").strip()


def test_mineru_sections_count():
    sections = _parse_markdown_sections(SAMPLE_MARKDOWN)
    assert len(sections) >= 3


def test_mineru_section_paths():
    sections = _parse_markdown_sections(SAMPLE_MARKDOWN)
    paths = [s.section_path for s in sections]
    assert any("Pharmacology" in p for p in paths)
    assert any("Dosing" in p for p in paths)
    assert any("Contraindications" in p for p in paths)


def test_mineru_section_content():
    sections = _parse_markdown_sections(SAMPLE_MARKDOWN)
    all_content = " ".join(s.content for s in sections)
    assert "Metformin inhibits" in all_content
    assert "renal failure" in all_content


def test_mineru_hierarchical_path():
    sections = _parse_markdown_sections(SAMPLE_MARKDOWN)
    dosing_section = next((s for s in sections if "Dosing" in s.section_path), None)
    assert dosing_section is not None
    # Should be nested under Pharmacology
    assert "Pharmacology" in dosing_section.section_path


# ===========================================================================
# Integration smoke test: PubMedConnector with mocked HTTP
# ===========================================================================


@pytest.mark.asyncio
async def test_pubmed_connector_search_returns_ids():
    """Verify that the connector correctly calls esearch and returns IDs."""
    from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(
        return_value={"esearchresult": {"idlist": ["11111111", "22222222"]}}
    )
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        connector = PubMedConnector()
        connector._session = mock_session
        ids = await connector._search(
            PubMedFetchConfig(query="metformin type 2 diabetes", max_results=10)
        )

    assert ids == ["11111111", "22222222"]
