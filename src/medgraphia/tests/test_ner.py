"""
Phase 3 tests: NER pipeline — MentionSpan, GLiNERNER, BertNER, MedicalNERPipeline.

All tests are pure-Python unit tests: no GPU, no network, no Neo4j required.
GLiNER and transformers are optional; tests gracefully skip when not installed.
"""
from __future__ import annotations

import pytest

from medgraphia.domain import Chunk, EntityType, Language, SourceMeta
from medgraphia.ingestion.ner._types import MentionSpan
from medgraphia.ingestion.ner.pipeline import MedicalNERPipeline, _merge_spans

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source() -> SourceMeta:
    return SourceMeta(source_id="test:ner", source_title="NER Test")


def _make_chunk(text: str, language: Language = Language.EN) -> Chunk:
    return Chunk(
        doc_id="ner-test-doc",
        source=_make_source(),
        language=language,
        section_path="Test",
        text=text,
    )


def _span(
    text: str,
    start: int,
    end: int,
    entity_type: EntityType,
    confidence: float = 0.9,
    source: str = "test",
) -> MentionSpan:
    return MentionSpan.from_text(
        text=text, start=start, end=end,
        entity_type=entity_type, confidence=confidence, source=source,
    )


# ===========================================================================
# MentionSpan
# ===========================================================================

class TestMentionSpan:
    def test_normalized_is_lowercased(self):
        s = MentionSpan.from_text("Metformin", 0, 9, EntityType.DRUG)
        assert s.normalized == "metformin"

    def test_overlaps_identical(self):
        a = _span("x", 0, 5, EntityType.DRUG)
        assert a.overlaps(a)

    def test_overlaps_partial(self):
        a = _span("diab", 0, 10, EntityType.DISEASE)
        b = _span("betes", 5, 15, EntityType.DISEASE)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_no_overlap_adjacent(self):
        a = _span("A", 0, 5, EntityType.DRUG)
        b = _span("B", 5, 10, EntityType.DISEASE)
        assert not a.overlaps(b)

    def test_no_overlap_distant(self):
        a = _span("A", 0, 5, EntityType.DRUG)
        b = _span("B", 20, 25, EntityType.DRUG)
        assert not a.overlaps(b)

    def test_containment_is_overlap(self):
        outer = _span("type 2 diabetes", 0, 15, EntityType.DISEASE)
        inner = _span("diabetes", 7, 15, EntityType.DISEASE)
        assert outer.overlaps(inner)

    def test_source_preserved(self):
        s = MentionSpan.from_text("insulin", 0, 7, EntityType.DRUG, source="gliner")
        assert s.source == "gliner"


# ===========================================================================
# _merge_spans (pipeline deduplication logic)
# ===========================================================================

class TestMergeSpans:
    def test_non_overlapping_spans_all_kept(self):
        a = _span("drug", 0, 4, EntityType.DRUG, confidence=0.9)
        b = _span("disease", 10, 17, EntityType.DISEASE, confidence=0.8)
        result = _merge_spans([a], [b])
        assert len(result) == 2

    def test_same_type_overlap_keeps_higher_confidence(self):
        low  = _span("diabetes", 0, 8, EntityType.DISEASE, confidence=0.6, source="bert")
        high = _span("type 2 diabetes", 0, 15, EntityType.DISEASE, confidence=0.9, source="gliner")
        result = _merge_spans([high], [low])
        # high-confidence span should survive; low should be dominated
        texts = [s.text for s in result]
        assert "type 2 diabetes" in texts
        # lower-confidence overlapping span of same type should be dropped
        assert "diabetes" not in texts or len(result) == 1

    def test_different_type_overlap_both_kept(self):
        # Same text region, different type → keep both (ambiguity)
        drug_span    = _span("insulin", 0, 7, EntityType.DRUG,    confidence=0.85)
        symptom_span = _span("insulin", 0, 7, EntityType.SYMPTOM, confidence=0.75)
        result = _merge_spans([drug_span], [symptom_span])
        types = {s.entity_type for s in result}
        assert EntityType.DRUG in types
        assert EntityType.SYMPTOM in types

    def test_empty_inputs(self):
        assert _merge_spans([], []) == []
        assert len(_merge_spans([_span("x", 0, 5, EntityType.DRUG)], [])) == 1

    def test_duplicates_from_same_source_deduplicated(self):
        a = _span("metformin", 0, 9, EntityType.DRUG, confidence=0.9)
        b = _span("metformin", 0, 9, EntityType.DRUG, confidence=0.7)
        result = _merge_spans([a, b], [])
        assert len(result) == 1


# ===========================================================================
# MedicalNERPipeline (no model required — tests that it runs without crashing)
# ===========================================================================

class TestNERPipelineStructure:
    """
    Tests the pipeline interface without triggering model downloads.
    GLiNER.predict and BertNER.predict are patched to return fixed spans so that
    tests run instantly without GPU / network access.
    """

    @pytest.fixture
    def pipeline(self):
        return MedicalNERPipeline(
            gliner_threshold=0.45,
            min_confidence=0.40,
        )

    @pytest.fixture
    def patched_pipeline(self, pipeline):
        """Pipeline whose underlying NER models are replaced with no-op stubs."""
        from unittest.mock import patch
        with patch.object(pipeline._gliner, "predict", return_value=[]) as _g, \
             patch.object(pipeline._bert,   "predict", return_value=[]) as _b:
            yield pipeline

    def test_extract_returns_chunk(self, patched_pipeline):
        chunk = _make_chunk("Metformin treats type 2 diabetes.")
        result = patched_pipeline.extract(chunk)
        assert result is not chunk          # new object
        assert result.chunk_id == chunk.chunk_id
        assert result.doc_id == chunk.doc_id

    def test_extract_empty_text_returns_same_chunk(self, patched_pipeline):
        chunk = _make_chunk("")
        result = patched_pipeline.extract(chunk)
        assert result is chunk

    def test_extract_entities_is_list(self, patched_pipeline):
        chunk = _make_chunk("Insulin is used for hyperglycemia.")
        result = patched_pipeline.extract(chunk)
        assert isinstance(result.entities, list)

    def test_extract_batch(self, patched_pipeline):
        chunks = [
            _make_chunk("Warfarin interacts with aspirin.", Language.EN),
            _make_chunk("高血压患者每日服用降压药。", Language.ZH),
            _make_chunk("Metformin ist ein Antidiabetikum.", Language.DE),
        ]
        results = patched_pipeline.extract_batch(chunks)
        assert len(results) == 3
        for r in results:
            assert isinstance(r.entities, list)

    def test_entities_from_gliner_have_mention_prefix(self, pipeline):
        """Directly inject GLiNER spans and verify Entity CUI prefix."""
        from unittest.mock import patch
        fake_spans = [
            MentionSpan.from_text("Metformin", 0, 9, EntityType.DRUG, 0.9, "gliner"),
            MentionSpan.from_text("type 2 diabetes", 10, 25, EntityType.DISEASE, 0.85, "gliner"),
        ]
        with patch.object(pipeline._gliner, "predict", return_value=fake_spans), \
             patch.object(pipeline._bert,   "predict", return_value=[]):
            chunk = _make_chunk("Metformin treats type 2 diabetes.")
            result = pipeline.extract(chunk)
            for entity in result.entities:
                assert entity.cui.startswith("MENTION:")

    def test_language_is_preserved(self, patched_pipeline):
        chunk = _make_chunk("Zweimal täglich einnehmen.", Language.DE)
        result = patched_pipeline.extract(chunk)
        assert result.language == Language.DE

    def test_section_path_preserved(self, patched_pipeline):
        chunk = Chunk(
            doc_id="d1",
            source=_make_source(),
            language=Language.EN,
            section_path="4.2 Posology",
            text="Take 500mg twice daily.",
        )
        result = patched_pipeline.extract(chunk)
        assert result.section_path == "4.2 Posology"


# ===========================================================================
# Entity deduplication in pipeline._spans_to_entities
# ===========================================================================

class TestSpanToEntityDeduplication:
    def test_same_mention_same_type_deduplicated(self):
        pipeline = MedicalNERPipeline(min_confidence=0.0)
        # Inject spans with same normalized text + same type manually
        from medgraphia.ingestion.ner._types import MentionSpan

        spans = [
            MentionSpan.from_text("Metformin", 0, 9, EntityType.DRUG, confidence=0.9, source="gliner"),
            MentionSpan.from_text("metformin", 10, 19, EntityType.DRUG, confidence=0.7, source="bert"),
        ]
        chunk = _make_chunk("Metformin and metformin")
        entities = pipeline._spans_to_entities(spans, chunk)

        # Should produce exactly one entity
        assert len(entities) == 1
        # Should keep the higher-confidence instance
        assert entities[0].confidence == 0.9

    def test_same_mention_different_type_kept(self):
        pipeline = MedicalNERPipeline(min_confidence=0.0)
        spans = [
            MentionSpan.from_text("insulin", 0, 7, EntityType.DRUG,    confidence=0.85, source="gliner"),
            MentionSpan.from_text("insulin", 0, 7, EntityType.GENE,    confidence=0.60, source="bert"),
        ]
        chunk = _make_chunk("insulin")
        entities = pipeline._spans_to_entities(spans, chunk)
        assert len(entities) == 2

    def test_entity_cui_has_mention_prefix(self):
        pipeline = MedicalNERPipeline(min_confidence=0.0)
        spans = [
            MentionSpan.from_text("Warfarin", 0, 8, EntityType.DRUG, confidence=0.9, source="test"),
        ]
        chunk = _make_chunk("Warfarin")
        entities = pipeline._spans_to_entities(spans, chunk)
        assert len(entities) == 1
        assert entities[0].cui == "MENTION:warfarin"

    def test_source_id_from_chunk(self):
        pipeline = MedicalNERPipeline(min_confidence=0.0)
        spans = [
            MentionSpan.from_text("aspirin", 0, 7, EntityType.DRUG, confidence=0.9, source="test"),
        ]
        chunk = _make_chunk("aspirin")
        entities = pipeline._spans_to_entities(spans, chunk)
        assert chunk.source.source_id in entities[0].source_ids


# ===========================================================================
# GLiNER availability probe (does not fail if not installed)
# ===========================================================================

class TestGLiNERAvailability:
    def test_gliner_availability_flag_is_bool(self):
        from medgraphia.ingestion.ner.gliner_ner import _GLINER_AVAILABLE
        assert isinstance(_GLINER_AVAILABLE, bool)

    def test_gliner_predict_returns_list(self):
        from unittest.mock import patch
        from medgraphia.ingestion.ner.gliner_ner import GLiNERNER
        ner = GLiNERNER()
        # Patch _load_model to return a fake model that returns empty list
        fake_model = type("FakeGLiNER", (), {
            "predict_entities": lambda self, text, labels, threshold=0.5: []
        })()
        with patch.object(ner, "_load_model", return_value=fake_model):
            result = ner.predict("Metformin treats type 2 diabetes.", Language.EN)
        assert isinstance(result, list)

    def test_gliner_predict_empty_text(self):
        from medgraphia.ingestion.ner.gliner_ner import GLiNERNER
        ner = GLiNERNER()
        assert ner.predict("", Language.EN) == []


# ===========================================================================
# BERT NER availability probe
# ===========================================================================

class TestBertNERAvailability:
    def test_bert_availability_flag_is_bool(self):
        from medgraphia.ingestion.ner.bert_ner import _TRANSFORMERS_AVAILABLE
        assert isinstance(_TRANSFORMERS_AVAILABLE, bool)

    def test_bert_predict_empty_text(self):
        from medgraphia.ingestion.ner.bert_ner import BertNER
        ner = BertNER()
        assert ner.predict("", Language.EN) == []

    def test_bert_predict_no_model_name_returns_empty(self):
        from medgraphia.ingestion.ner.bert_ner import BertNER
        ner = BertNER(en_model="")
        result = ner.predict("Insulin is a hormone.", Language.EN)
        assert result == []
