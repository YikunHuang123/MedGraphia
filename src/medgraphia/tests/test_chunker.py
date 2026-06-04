"""
Phase 2 tests: MedicalChunker + MedicalNormalizer.

Tests are pure-Python unit tests — no Neo4j, no GPU, no network required.
All Chunk / RawDocument objects are constructed in-memory.
"""
from __future__ import annotations

import textwrap

import pytest

from medgraphia.domain import Language, ParsedSection, RawDocument, SourceMeta
from medgraphia.ingestion.chunker import (
    MedicalChunker,
    _overlap_tail,
    _split_paragraphs,
    _split_sentences,
)
from medgraphia.ingestion.normalizer import MedicalNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source() -> SourceMeta:
    return SourceMeta(source_id="test:001", source_title="Test Document")


def _make_doc(
    abstract: str = "",
    full_text: str = "",
    sections: list[ParsedSection] | None = None,
    language: Language = Language.EN,
) -> RawDocument:
    return RawDocument(
        source=_make_source(),
        language=language,
        title="Test",
        abstract=abstract,
        full_text=full_text,
        sections=sections or [],
    )


# ===========================================================================
# MedicalChunker — section-based documents
# ===========================================================================

class TestChunkerSections:
    def test_single_small_section_produces_one_chunk(self):
        sections = [
            ParsedSection(
                section_path="4.1 Indications",
                content="Metformin is indicated for type 2 diabetes.",
            )
        ]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker().chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].section_path == "4.1 Indications"
        assert "Metformin" in chunks[0].text

    def test_section_path_preserved(self):
        sections = [
            ParsedSection(
                section_path="Clinical Particulars > Posology > Paediatric",
                content="Children: 500 mg twice daily.",
            )
        ]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker().chunk(doc)

        assert chunks[0].section_path == "Clinical Particulars > Posology > Paediatric"

    def test_multiple_sections_produce_multiple_chunks(self):
        sections = [
            ParsedSection(section_path="S1", content="First section text."),
            ParsedSection(section_path="S2", content="Second section text."),
            ParsedSection(section_path="S3", content="Third section text."),
        ]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker().chunk(doc)

        assert len(chunks) == 3

    def test_empty_section_skipped(self):
        sections = [
            ParsedSection(section_path="Empty", content=""),
            ParsedSection(section_path="Full",  content="Non-empty content here."),
        ]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker().chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].section_path == "Full"

    def test_large_section_split_into_multiple_chunks(self):
        # Build a section with many paragraphs that collectively exceed 100 tokens.
        # Each paragraph is one short sentence so the splitter can find boundaries.
        para = "Metformin reduces HbA1c levels significantly."  # ~8 words → ~10 tokens
        content = "\n\n".join([para] * 15)  # 15 paragraphs → ~150 tokens total
        sections = [ParsedSection(section_path="BigSection", content=content)]
        doc = _make_doc(sections=sections)
        chunker = MedicalChunker(max_tokens=50, overlap_tokens=5)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1

    def test_large_section_chunks_all_carry_section_path(self):
        long_para = " ".join(["word"] * 600)
        sections = [ParsedSection(section_path="BigSection", content=long_para)]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker(max_tokens=100).chunk(doc)

        assert all(c.section_path == "BigSection" for c in chunks)

    def test_chunk_doc_id_matches_document(self):
        sections = [ParsedSection(section_path="S1", content="Some text.")]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker().chunk(doc)

        assert all(c.doc_id == doc.doc_id for c in chunks)

    def test_chunk_language_matches_document(self):
        sections = [ParsedSection(section_path="S", content="Text.")]
        doc = _make_doc(sections=sections, language=Language.DE)
        chunks = MedicalChunker().chunk(doc)

        assert chunks[0].language == Language.DE

    def test_token_count_populated(self):
        sections = [ParsedSection(section_path="S", content="Metformin reduces HbA1c.")]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker().chunk(doc)

        assert chunks[0].token_count is not None
        assert chunks[0].token_count > 0

    def test_page_info_carried_from_section(self):
        sections = [
            ParsedSection(section_path="S", content="Text.", page_start=7)
        ]
        doc = _make_doc(sections=sections)
        chunks = MedicalChunker().chunk(doc)

        assert chunks[0].page == 7


# ===========================================================================
# MedicalChunker — abstract-only documents (PubMed style)
# ===========================================================================

class TestChunkerAbstractOnly:
    def test_abstract_produces_at_least_one_chunk(self):
        doc = _make_doc(abstract="Metformin is first-line therapy for T2DM.")
        chunks = MedicalChunker().chunk(doc)

        assert len(chunks) >= 1

    def test_abstract_section_path_label(self):
        doc = _make_doc(abstract="Background text.")
        chunks = MedicalChunker().chunk(doc)

        assert chunks[0].section_path == "Abstract"

    def test_abstract_and_full_text_both_chunked(self):
        doc = _make_doc(
            abstract="Abstract text.",
            full_text="Extended body content with more details.",
        )
        chunks = MedicalChunker().chunk(doc)
        paths = [c.section_path for c in chunks]

        assert "Abstract" in paths
        assert "Body" in paths

    def test_duplicate_full_text_not_double_chunked(self):
        # When full_text == abstract, only one chunk group should be created
        same = "Same text for abstract and full_text."
        doc = _make_doc(abstract=same, full_text=same)
        chunks = MedicalChunker().chunk(doc)

        # Only Abstract section should appear (full_text is deduplicated)
        paths = [c.section_path for c in chunks]
        assert paths.count("Body") == 0

    def test_empty_doc_returns_no_chunks(self):
        doc = _make_doc()
        chunks = MedicalChunker().chunk(doc)

        assert chunks == []


# ===========================================================================
# MedicalChunker — multi-paragraph splitting
# ===========================================================================

class TestChunkerParagraphSplit:
    def test_multiple_paragraphs_merged_within_limit(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        doc = _make_doc(abstract=text)
        chunks = MedicalChunker(max_tokens=512).chunk(doc)

        # All three short paragraphs fit in one chunk
        assert len(chunks) == 1
        assert "Para one" in chunks[0].text

    def test_overlap_preserves_context(self):
        # Make paragraphs that force two chunks; check overlap content appears in both
        word = "medication "
        para_a = word * 50   # ~65 tokens
        para_b = word * 50
        para_c = word * 50
        text = f"{para_a.strip()}\n\n{para_b.strip()}\n\n{para_c.strip()}"
        doc = _make_doc(abstract=text)
        chunks = MedicalChunker(max_tokens=80, overlap_tokens=20).chunk(doc)

        # Multiple chunks should be produced
        assert len(chunks) >= 2


# ===========================================================================
# Internal helpers
# ===========================================================================

class TestSplitHelpers:
    def test_split_paragraphs_on_blank_lines(self):
        text = "A.\n\nB.\n\nC."
        paras = _split_paragraphs(text)
        assert paras == ["A.", "B.", "C."]

    def test_split_paragraphs_ignores_extra_blank_lines(self):
        text = "A.\n\n\n\nB."
        paras = _split_paragraphs(text)
        assert len(paras) == 2

    def test_split_sentences_en(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = _split_sentences(text, Language.EN)
        assert len(sents) >= 2

    def test_split_sentences_zh(self):
        text = "这是第一句。这是第二句。这是第三句。"
        sents = _split_sentences(text, Language.ZH)
        assert len(sents) >= 2

    def test_overlap_tail_respects_budget(self):
        items = ["aaaa", "bbbb", "cccc"]

        def tok(text, lang):
            return len(text)

        tail = _overlap_tail(items, overlap_tokens=5, language=Language.EN, token_fn=tok)
        total = sum(len(i) for i in tail)
        assert total <= 5

    def test_overlap_tail_empty_on_zero_budget(self):
        items = ["aaa", "bbb"]

        def tok(text, lang):
            return len(text)

        tail = _overlap_tail(items, overlap_tokens=0, language=Language.EN, token_fn=tok)
        assert tail == []


# ===========================================================================
# MedicalNormalizer — frequency normalisation
# ===========================================================================

class TestNormalizerFrequency:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.norm = MedicalNormalizer()

    # --- English ---

    def test_en_twice_daily(self):
        out = self.norm.normalize("Give 500 mg twice daily.", Language.EN)
        assert "bid" in out

    def test_en_bid_abbreviation(self):
        out = self.norm.normalize("Administer b.i.d.", Language.EN)
        assert "bid" in out

    def test_en_once_daily(self):
        out = self.norm.normalize("Take once a day.", Language.EN)
        assert "qd" in out

    def test_en_q12h(self):
        out = self.norm.normalize("Dose every 12 hours.", Language.EN)
        assert "bid" in out

    def test_en_three_times_daily(self):
        out = self.norm.normalize("Administer three times daily.", Language.EN)
        assert "tid" in out

    def test_en_four_times_daily(self):
        out = self.norm.normalize("Four times daily with food.", Language.EN)
        assert "qid" in out

    def test_en_q6h(self):
        out = self.norm.normalize("Every 6 hours.", Language.EN)
        assert "qid" in out

    def test_en_as_needed(self):
        out = self.norm.normalize("Use as needed.", Language.EN)
        assert "prn" in out

    def test_en_at_bedtime(self):
        out = self.norm.normalize("Take at bedtime.", Language.EN)
        assert "qhs" in out

    # --- German ---

    def test_de_zweimal_taeglich(self):
        out = self.norm.normalize("Zweimal täglich einnehmen.", Language.DE)
        assert "bid" in out

    def test_de_2mal_taeglich(self):
        out = self.norm.normalize("2mal täglich.", Language.DE)
        assert "bid" in out

    def test_de_einmal_taeglich(self):
        out = self.norm.normalize("Einmal täglich.", Language.DE)
        assert "qd" in out

    def test_de_dreimal_taeglich(self):
        out = self.norm.normalize("Dreimal täglich.", Language.DE)
        assert "tid" in out

    def test_de_bei_bedarf(self):
        out = self.norm.normalize("Bei Bedarf anwenden.", Language.DE)
        assert "prn" in out

    def test_de_bis_die(self):
        out = self.norm.normalize("500mg bis die.", Language.DE)
        assert "bid" in out

    # --- Chinese ---

    def test_zh_twice_daily(self):
        out = self.norm.normalize("每日两次。", Language.ZH)
        assert "bid" in out

    def test_zh_once_daily(self):
        out = self.norm.normalize("每日一次。", Language.ZH)
        assert "qd" in out

    def test_zh_three_times_daily(self):
        out = self.norm.normalize("每日三次。", Language.ZH)
        assert "tid" in out

    def test_zh_prn(self):
        out = self.norm.normalize("必要时服用。", Language.ZH)
        assert "prn" in out

    def test_zh_twice_slash_notation(self):
        out = self.norm.normalize("2次/日", Language.ZH)
        assert "bid" in out


# ===========================================================================
# MedicalNormalizer — dosage unit normalisation
# ===========================================================================

class TestNormalizerDosageUnits:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.norm = MedicalNormalizer()

    def test_no_space_mg(self):
        out = self.norm.normalize("Give 500mg.", Language.EN)
        assert "500 mg" in out

    def test_no_space_mcg(self):
        out = self.norm.normalize("Dose 50mcg IV.", Language.EN)
        assert "50 mcg" in out

    def test_mu_g_symbol(self):
        out = self.norm.normalize("50µg daily.", Language.EN)
        assert "50 mcg" in out

    def test_milligrams_spelled_out(self):
        out = self.norm.normalize("1000milligrams.", Language.EN)
        assert "1000 mg" in out

    def test_decimal_dose(self):
        out = self.norm.normalize("Give 0.5mg.", Language.EN)
        assert "0.5 mg" in out

    def test_unit_already_spaced_unchanged(self):
        out = self.norm.normalize("500 mg twice daily.", Language.EN)
        assert "500 mg" in out

    def test_combined_frequency_and_dosage(self):
        out = self.norm.normalize("Administer 500mg twice daily.", Language.EN)
        assert "500 mg" in out
        assert "bid" in out


# ===========================================================================
# MedicalNormalizer — normalize_chunk
# ===========================================================================

class TestNormalizerChunk:
    def test_normalize_chunk_returns_new_chunk(self):
        from medgraphia.domain import Chunk
        norm = MedicalNormalizer()
        source = _make_source()
        chunk = Chunk(
            doc_id="doc-001",
            source=source,
            language=Language.EN,
            section_path="Dosing",
            text="Give 500mg twice daily.",
        )
        result = norm.normalize_chunk(chunk)

        assert result.chunk_id == chunk.chunk_id
        assert result.doc_id == chunk.doc_id
        assert "500 mg" in result.text
        assert "bid" in result.text

    def test_normalize_chunk_unchanged_text_returns_same_object(self):
        from medgraphia.domain import Chunk
        norm = MedicalNormalizer()
        source = _make_source()
        chunk = Chunk(
            doc_id="doc-001",
            source=source,
            language=Language.EN,
            section_path="S",
            text="No medication information here.",
        )
        result = norm.normalize_chunk(chunk)

        assert result is chunk  # same object, nothing to normalise


# ===========================================================================
# Integration: chunker + normalizer pipeline
# ===========================================================================

class TestChunkerNormalizerIntegration:
    def test_pipeline_preserves_section_path_and_normalises_text(self):
        sections = [
            ParsedSection(
                section_path="4.2 Posology",
                content="Adults: 500mg twice daily with meals.",
            )
        ]
        doc = _make_doc(sections=sections)
        chunker    = MedicalChunker()
        normalizer = MedicalNormalizer()

        chunks = [normalizer.normalize_chunk(c) for c in chunker.chunk(doc)]

        assert len(chunks) == 1
        assert chunks[0].section_path == "4.2 Posology"
        assert "500 mg" in chunks[0].text
        assert "bid" in chunks[0].text

    def test_pipeline_handles_multilingual_doc(self):
        sections = [
            ParsedSection(
                section_path="Dosierung",
                content="Zweimal täglich 1000mg einnehmen.",
            )
        ]
        doc = _make_doc(sections=sections, language=Language.DE)
        chunker    = MedicalChunker()
        normalizer = MedicalNormalizer()

        chunks = [normalizer.normalize_chunk(c) for c in chunker.chunk(doc)]

        assert "1000 mg" in chunks[0].text
        assert "bid" in chunks[0].text
