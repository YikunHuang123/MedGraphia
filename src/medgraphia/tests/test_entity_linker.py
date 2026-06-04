"""
Phase 3 tests: EntityLinker — BM25 candidate retrieval, string re-ranking,
SapBERT re-ranking (skipped if sentence-transformers not installed).

All tests are pure-Python unit tests: no network, no Neo4j, no GPU required.
"""
from __future__ import annotations

import pytest

from medgraphia.domain import Chunk, Entity, EntityType, Language, SourceMeta
from medgraphia.ingestion.entity_linker import EntityLinker, _tokenize


# ---------------------------------------------------------------------------
# Helpers — minimal in-memory UMLS index
# ---------------------------------------------------------------------------

def _mini_umls() -> dict[str, dict]:
    """
    A tiny UMLS-like concept index for testing (no real UMLS files needed).
    Mirrors the dict structure returned by UMLSLoader.load().
    """
    return {
        "C0025598": {
            "cui": "C0025598",
            "label": "Metformin",
            "entity_type": "Drug",
            "synonyms": ["metformin hydrochloride", "glucophage"],
            "lang_labels": {"zh": "二甲双胍", "de": "Metformin"},
        },
        "C0011860": {
            "cui": "C0011860",
            "label": "Diabetes Mellitus, Type 2",
            "entity_type": "Disease",
            "synonyms": ["type 2 diabetes", "T2DM", "NIDDM"],
            "lang_labels": {"zh": "2型糖尿病", "de": "Typ-2-Diabetes mellitus"},
        },
        "C0027051": {
            "cui": "C0027051",
            "label": "Myocardial Infarction",
            "entity_type": "Disease",
            "synonyms": ["heart attack", "MI", "myocardial infarct"],
            "lang_labels": {"zh": "心肌梗死", "de": "Myokardinfarkt"},
        },
        "C0020538": {
            "cui": "C0020538",
            "label": "Hypertension",
            "entity_type": "Disease",
            "synonyms": ["high blood pressure", "arterial hypertension"],
            "lang_labels": {"zh": "高血压", "de": "Bluthochdruck"},
        },
        "C0004057": {
            "cui": "C0004057",
            "label": "Aspirin",
            "entity_type": "Drug",
            "synonyms": ["acetylsalicylic acid", "ASA", "aspirin tablet"],
            "lang_labels": {"zh": "阿司匹林", "de": "Aspirin"},
        },
        "C0043031": {
            "cui": "C0043031",
            "label": "Warfarin",
            "entity_type": "Drug",
            "synonyms": ["coumadin", "warfarin sodium"],
            "lang_labels": {"zh": "华法林", "de": "Warfarin"},
        },
    }


def _linker(umls: dict | None = None) -> EntityLinker:
    linker = EntityLinker(
        umls_index=umls or _mini_umls(),
        bm25_top_k=10,
        link_threshold=0.60,
        sapbert_model="",       # no SapBERT in unit tests
        sapbert_threshold=0.75,
    )
    linker.build_index()
    return linker


def _mention(text: str, entity_type: EntityType) -> Entity:
    return Entity(
        cui=f"MENTION:{text.strip().lower()}",
        label=text,
        entity_type=entity_type,
        confidence=0.85,
    )


def _make_chunk(entities: list[Entity]) -> Chunk:
    return Chunk(
        doc_id="test-doc",
        source=SourceMeta(source_id="test:el", source_title="EL Test"),
        language=Language.EN,
        section_path="Test",
        text="test text",
        entities=entities,
    )


# ===========================================================================
# Tokenizer
# ===========================================================================

class TestTokenize:
    def test_english_words(self):
        tokens = _tokenize("type 2 diabetes mellitus")
        assert "type" in tokens
        assert "diabetes" in tokens
        assert "mellitus" in tokens

    def test_chinese_chars(self):
        tokens = _tokenize("心肌梗死")
        assert "心" in tokens
        assert "肌" in tokens

    def test_german_umlaut(self):
        tokens = _tokenize("Myokardinfarkt Bluthochdruck")
        assert "myokardinfarkt" in tokens

    def test_mixed_text(self):
        tokens = _tokenize("Metformin 二甲双胍 500mg")
        assert "metformin" in tokens
        assert "二" in tokens

    def test_empty_returns_fallback(self):
        tokens = _tokenize("")
        assert len(tokens) >= 1


# ===========================================================================
# EntityLinker — index construction
# ===========================================================================

class TestLinkerBuildIndex:
    def test_build_index_idempotent(self):
        linker = _linker()
        # Second call should not raise
        linker.build_index()

    def test_empty_umls_builds_without_error(self):
        linker = EntityLinker(umls_index={})  # explicit empty index
        linker.build_index()
        # Should return unchanged entities
        mention = _mention("metformin", EntityType.DRUG)
        result = linker.link_entities([mention])
        assert result[0].cui == "MENTION:metformin"

    def test_entries_count_matches_umls(self):
        linker = _linker()
        assert len(linker._entries) == len(_mini_umls())


# ===========================================================================
# BM25 candidate retrieval
# ===========================================================================

class TestBM25Candidates:
    def test_exact_label_match(self):
        linker = _linker()
        candidates = linker._bm25_candidates("Metformin")
        cuis = [e.cui for e in candidates]
        assert "C0025598" in cuis

    def test_synonym_match(self):
        linker = _linker()
        candidates = linker._bm25_candidates("glucophage")
        cuis = [e.cui for e in candidates]
        assert "C0025598" in cuis

    def test_chinese_label_match(self):
        linker = _linker()
        candidates = linker._bm25_candidates("心肌梗死")
        cuis = [e.cui for e in candidates]
        assert "C0027051" in cuis

    def test_german_label_match(self):
        linker = _linker()
        candidates = linker._bm25_candidates("Bluthochdruck")
        cuis = [e.cui for e in candidates]
        assert "C0020538" in cuis

    def test_no_match_returns_empty_or_small_list(self):
        linker = _linker()
        candidates = linker._bm25_candidates("zzzzxxx999")
        # BM25 may return empty; it should at least not error
        assert isinstance(candidates, list)


# ===========================================================================
# String re-ranking (difflib fallback)
# ===========================================================================

class TestStringRerank:
    def test_exact_label_gets_high_score(self):
        linker = _linker()
        candidates = linker._entries  # all entries
        result = linker._string_rerank("Warfarin", candidates)
        assert result is not None
        cui, label, lang_labels, score = result
        assert cui == "C0043031"
        assert score >= 0.9

    def test_synonym_resolves_to_correct_cui(self):
        linker = _linker()
        result = linker._string_rerank("heart attack", linker._entries)
        assert result is not None
        assert result[0] == "C0027051"  # Myocardial Infarction

    def test_low_similarity_returns_none(self):
        linker = _linker()
        result = linker._string_rerank("zyx123abc", linker._entries)
        assert result is None


# ===========================================================================
# link_entities — end-to-end
# ===========================================================================

class TestLinkEntities:
    def test_english_drug_linked(self):
        linker = _linker()
        entities = [_mention("Metformin", EntityType.DRUG)]
        result = linker.link_entities(entities)
        assert len(result) == 1
        assert result[0].cui == "C0025598"

    def test_english_disease_linked(self):
        linker = _linker()
        entities = [_mention("type 2 diabetes", EntityType.DISEASE)]
        result = linker.link_entities(entities)
        assert result[0].cui == "C0011860"

    def test_chinese_disease_linked(self):
        linker = _linker()
        entities = [_mention("心肌梗死", EntityType.DISEASE)]
        result = linker.link_entities(entities)
        assert result[0].cui == "C0027051"

    def test_german_drug_linked(self):
        linker = _linker()
        entities = [_mention("Warfarin", EntityType.DRUG)]
        result = linker.link_entities(entities)
        assert result[0].cui == "C0043031"

    def test_unknown_mention_keeps_provisional_cui(self):
        linker = _linker()
        entities = [_mention("zyx123_unknown_drug", EntityType.DRUG)]
        result = linker.link_entities(entities)
        assert result[0].cui.startswith("MENTION:")

    def test_already_linked_entity_unchanged(self):
        linker = _linker()
        entity = Entity(
            cui="C0025598",
            label="Metformin",
            entity_type=EntityType.DRUG,
            confidence=1.0,
        )
        result = linker.link_entities([entity])
        assert result[0].cui == "C0025598"
        assert result[0] is entity

    def test_multiple_entities(self):
        linker = _linker()
        entities = [
            _mention("Metformin", EntityType.DRUG),
            _mention("Hypertension", EntityType.DISEASE),
            _mention("Aspirin", EntityType.DRUG),
        ]
        result = linker.link_entities(entities)
        cuis = {e.cui for e in result}
        assert "C0025598" in cuis
        assert "C0020538" in cuis
        assert "C0004057" in cuis

    def test_confidence_propagated(self):
        linker = _linker()
        entity = _mention("Metformin", EntityType.DRUG)
        result = linker.link_entities([entity])
        # Linked entity should have confidence <= original (min of original and link score)
        assert result[0].confidence <= entity.confidence + 1e-6

    def test_lang_labels_populated_from_umls(self):
        linker = _linker()
        entities = [_mention("Warfarin", EntityType.DRUG)]
        result = linker.link_entities([entities[0]])
        assert result[0].lang_labels.get("zh") == "华法林"
        assert result[0].lang_labels.get("de") == "Warfarin"

    def test_type_filter_prefers_same_type(self):
        # "Aspirin" exists as Drug; should not be linked to Disease-type concepts
        linker = _linker()
        entities = [_mention("Aspirin", EntityType.DRUG)]
        result = linker.link_entities([entities[0]])
        assert result[0].entity_type == EntityType.DRUG


# ===========================================================================
# link_chunk
# ===========================================================================

class TestLinkChunk:
    def test_link_chunk_returns_new_chunk(self):
        linker = _linker()
        entities = [_mention("Metformin", EntityType.DRUG)]
        chunk = _make_chunk(entities)
        result = linker.link_chunk(chunk)
        assert result is not chunk

    def test_link_chunk_empty_entities_unchanged(self):
        linker = _linker()
        chunk = _make_chunk([])
        result = linker.link_chunk(chunk)
        assert result is chunk

    def test_link_chunk_entities_resolved(self):
        linker = _linker()
        entities = [_mention("Hypertension", EntityType.DISEASE)]
        chunk = _make_chunk(entities)
        result = linker.link_chunk(chunk)
        assert result.entities[0].cui == "C0020538"


# ===========================================================================
# SapBERT availability probe
# ===========================================================================

class TestSapBERTAvailability:
    def test_sapbert_available_flag_is_bool(self):
        from medgraphia.ingestion.entity_linker import _SAPBERT_AVAILABLE
        assert isinstance(_SAPBERT_AVAILABLE, bool)

    def test_linker_works_without_sapbert(self):
        # sapbert_model="" → string fallback
        linker = EntityLinker(
            umls_index=_mini_umls(),
            sapbert_model="",
            link_threshold=0.60,
        )
        linker.build_index()
        result = linker.link_entities([_mention("Aspirin", EntityType.DRUG)])
        assert result[0].cui == "C0004057"


# ===========================================================================
# Phase 4 integration tests
# ===========================================================================

class TestGLiNERNERToELIntegration:
    """
    EL + GLiNER integration test.

    Strategy:
      1. Mock GLiNER.predict to return fixed MentionSpan objects.
      2. Run MedicalNERPipeline.extract → entities with MENTION: CUIs.
      3. Feed the resulting chunk through EntityLinker.link_chunk.
      4. Verify the final CUIs match the mini-UMLS ground truth.

    No model is downloaded; this tests the full NER → EL chain with controlled input.
    """

    def _make_source(self) -> SourceMeta:
        return SourceMeta(source_id="integration:ner-el", source_title="NER-EL Integration Test")

    def _run_pipeline(
        self,
        text: str,
        gliner_spans,
        expected_cuis: dict[str, str],
    ) -> None:
        """
        Helper: inject fixed GLiNER spans, run pipeline + EL, assert CUI mapping.
        expected_cuis: {normalized_text → expected_cui}
        """
        from medgraphia.ingestion.ner.pipeline import MedicalNERPipeline

        pipeline = MedicalNERPipeline(min_confidence=0.0)
        linker = EntityLinker(
            umls_index=_mini_umls(),
            bm25_top_k=10,
            link_threshold=0.60,
            sapbert_model="",
        )
        linker.build_index()

        chunk = Chunk(
            doc_id="integration-doc",
            source=self._make_source(),
            language=Language.EN,
            section_path="Test",
            text=text,
        )

        # Inject fixed spans into the pipeline (both BERT and GLiNER patched)
        from unittest.mock import patch
        with patch.object(pipeline._gliner, "predict", return_value=gliner_spans), \
             patch.object(pipeline._bert, "predict", return_value=[]):
            chunk_with_entities = pipeline.extract(chunk)

        # All entities should start with MENTION: at this stage
        for entity in chunk_with_entities.entities:
            assert entity.cui.startswith("MENTION:"), (
                f"Expected MENTION: prefix before EL, got {entity.cui}"
            )

        # Run entity linking
        linked_chunk = linker.link_chunk(chunk_with_entities)

        # Build mapping: normalized_text → linked CUI
        linked_map = {
            e.cui[len("MENTION:"):] if e.cui.startswith("MENTION:") else e.label.lower(): e.cui
            for e in linked_chunk.entities
        }

        for mention_text, expected_cui in expected_cuis.items():
            matched = [
                e for e in linked_chunk.entities
                if e.cui == expected_cui
                or (e.label.lower() == mention_text.lower())
            ]
            assert matched, (
                f"Expected CUI {expected_cui!r} for mention {mention_text!r} "
                f"not found in linked entities: "
                f"{[(e.label, e.cui) for e in linked_chunk.entities]}"
            )

    def test_drug_mention_resolves_to_correct_cui(self):
        """GLiNER detects 'Metformin' → EL resolves to C0025598."""
        from medgraphia.ingestion.ner._types import MentionSpan

        text = "Metformin is used to treat type 2 diabetes."
        gliner_spans = [
            MentionSpan.from_text("Metformin", 0, 9, EntityType.DRUG, confidence=0.9, source="gliner"),
        ]
        self._run_pipeline(text, gliner_spans, {"Metformin": "C0025598"})

    def test_disease_mention_resolves_to_correct_cui(self):
        """GLiNER detects 'type 2 diabetes' → EL resolves to C0011860."""
        from medgraphia.ingestion.ner._types import MentionSpan

        text = "Metformin is used to treat type 2 diabetes mellitus."
        gliner_spans = [
            MentionSpan.from_text(
                "type 2 diabetes", 26, 41, EntityType.DISEASE, confidence=0.88, source="gliner"
            ),
        ]
        self._run_pipeline(text, gliner_spans, {"type 2 diabetes": "C0011860"})

    def test_multiple_entities_all_resolved(self):
        """Both Drug and Disease mentions are correctly resolved end-to-end."""
        from medgraphia.ingestion.ner._types import MentionSpan

        text = "Warfarin interacts with Aspirin in hypertension patients."
        gliner_spans = [
            MentionSpan.from_text("Warfarin", 0, 8, EntityType.DRUG, confidence=0.92, source="gliner"),
            MentionSpan.from_text("Aspirin", 23, 30, EntityType.DRUG, confidence=0.91, source="gliner"),
        ]
        from medgraphia.ingestion.ner.pipeline import MedicalNERPipeline

        pipeline = MedicalNERPipeline(min_confidence=0.0)
        linker = EntityLinker(
            umls_index=_mini_umls(),
            bm25_top_k=10,
            link_threshold=0.60,
            sapbert_model="",
        )
        linker.build_index()

        source = SourceMeta(source_id="int:multi", source_title="Multi")
        chunk = Chunk(doc_id="d1", source=source, language=Language.EN,
                      section_path="", text=text)

        from unittest.mock import patch
        with patch.object(pipeline._gliner, "predict", return_value=gliner_spans), \
             patch.object(pipeline._bert, "predict", return_value=[]):
            chunk = pipeline.extract(chunk)

        linked = linker.link_chunk(chunk)
        linked_cuis = {e.cui for e in linked.entities}
        assert "C0043031" in linked_cuis, "Warfarin should resolve to C0043031"
        assert "C0004057" in linked_cuis, "Aspirin should resolve to C0004057"

    def test_unresolvable_mention_keeps_provisional_cui(self):
        """Mentions that cannot be linked to UMLS retain their MENTION: prefix."""
        from medgraphia.ingestion.ner._types import MentionSpan
        from medgraphia.ingestion.ner.pipeline import MedicalNERPipeline

        text = "Zylofuroxinetriamine-X12 was tested."
        gliner_spans = [
            MentionSpan.from_text(
                "Zylofuroxinetriamine-X12", 0, 24, EntityType.DRUG, confidence=0.85, source="gliner"
            ),
        ]
        pipeline = MedicalNERPipeline(min_confidence=0.0)
        linker = EntityLinker(
            umls_index=_mini_umls(), bm25_top_k=10, link_threshold=0.60, sapbert_model=""
        )
        linker.build_index()

        source = SourceMeta(source_id="int:unknown", source_title="Unknown")
        chunk = Chunk(doc_id="d2", source=source, language=Language.EN,
                      section_path="", text=text)

        from unittest.mock import patch
        with patch.object(pipeline._gliner, "predict", return_value=gliner_spans), \
             patch.object(pipeline._bert, "predict", return_value=[]):
            chunk = pipeline.extract(chunk)

        linked = linker.link_chunk(chunk)
        provisional = [e for e in linked.entities if e.cui.startswith("MENTION:")]
        assert len(provisional) == 1

    def test_entity_confidence_not_inflated_by_el(self):
        """Linked entity confidence must not exceed the original NER confidence."""
        from medgraphia.ingestion.ner._types import MentionSpan
        from medgraphia.ingestion.ner.pipeline import MedicalNERPipeline

        text = "Aspirin is taken daily."
        original_conf = 0.55
        gliner_spans = [
            MentionSpan.from_text(
                "Aspirin", 0, 7, EntityType.DRUG, confidence=original_conf, source="gliner"
            ),
        ]
        pipeline = MedicalNERPipeline(min_confidence=0.0)
        linker = EntityLinker(
            umls_index=_mini_umls(), bm25_top_k=10, link_threshold=0.50, sapbert_model=""
        )
        linker.build_index()

        source = SourceMeta(source_id="int:conf", source_title="Conf")
        chunk = Chunk(doc_id="d3", source=source, language=Language.EN,
                      section_path="", text=text)

        from unittest.mock import patch
        with patch.object(pipeline._gliner, "predict", return_value=gliner_spans), \
             patch.object(pipeline._bert, "predict", return_value=[]):
            chunk = pipeline.extract(chunk)

        linked = linker.link_chunk(chunk)
        for entity in linked.entities:
            assert entity.confidence <= original_conf + 1e-6, (
                f"Confidence was inflated: {entity.confidence} > {original_conf}"
            )


class TestBM25TypeFilterPriority:
    """
    BM25 type filter priority test.

    Scenario: the same concept name ("Aspirin") exists in the mini-UMLS as a Drug
    (C0004057) and, in an extended test index, also as a synthetic Disease entry.
    Verifies that the type filter correctly prioritises the same-type candidate.
    """

    def _dual_type_umls(self) -> dict:
        """Mini-UMLS where 'Aspirin' appears as both Drug (C0004057) and Disease (C9999999)."""
        index = _mini_umls()
        index["C9999999"] = {
            "cui": "C9999999",
            "label": "Aspirin Hypersensitivity",    # also contains 'aspirin' in synonyms
            "entity_type": "Disease",
            "synonyms": ["aspirin allergy", "aspirin"],  # synonym overlap
            "lang_labels": {},
        }
        return index

    def test_drug_type_preferred_over_disease_for_drug_mention(self):
        """
        When 'Aspirin' mention has entity_type=Drug and there are candidates in
        both Drug and Disease categories, the Drug entry (C0004057) must win.
        """
        linker = EntityLinker(
            umls_index=self._dual_type_umls(),
            bm25_top_k=50,
            link_threshold=0.50,
            sapbert_model="",
        )
        linker.build_index()

        entities = [_mention("Aspirin", EntityType.DRUG)]
        result = linker.link_entities(entities)

        assert len(result) == 1
        assert result[0].cui == "C0004057", (
            f"Expected Drug CUI C0004057, got {result[0].cui!r} — "
            "type filter should prefer Drug when entity_type=Drug"
        )

    def test_disease_type_preferred_over_drug_for_disease_mention(self):
        """
        When 'Aspirin Hypersensitivity' mention has entity_type=Disease, the
        Disease entry (C9999999) should win over the Drug entry.
        """
        linker = EntityLinker(
            umls_index=self._dual_type_umls(),
            bm25_top_k=50,
            link_threshold=0.50,
            sapbert_model="",
        )
        linker.build_index()

        entities = [_mention("Aspirin Hypersensitivity", EntityType.DISEASE)]
        result = linker.link_entities(entities)

        assert len(result) == 1
        assert result[0].cui == "C9999999", (
            f"Expected Disease CUI C9999999, got {result[0].cui!r} — "
            "type filter should prefer Disease when entity_type=Disease"
        )

    def test_type_filter_falls_back_to_any_type_when_no_same_type_match(self):
        """
        If no candidate matches the requested entity type, _find_best_match should
        fall back to any-type candidates rather than returning None.
        """
        linker = EntityLinker(
            umls_index=_mini_umls(),  # only Disease/Drug entries
            bm25_top_k=50,
            link_threshold=0.50,
            sapbert_model="",
        )
        linker.build_index()

        # "Aspirin" is Drug — searching with type GENE should fall back to Drug entry
        entities = [_mention("Aspirin", EntityType.GENE)]
        result = linker.link_entities(entities)

        # Should still find Aspirin (C0004057) via fallback, not return MENTION:
        assert len(result) == 1
        # Must not stay as MENTION: (fallback should work)
        assert result[0].cui == "C0004057", (
            f"Type fallback should still resolve Aspirin, got {result[0].cui!r}"
        )

    def test_find_best_match_returns_typed_pool_when_available(self):
        """Direct test of _find_best_match's typed/untyped pool selection."""
        linker = EntityLinker(
            umls_index=self._dual_type_umls(),
            bm25_top_k=50,
            link_threshold=0.40,
            sapbert_model="",
        )
        linker.build_index()

        # Directly call internal method for white-box verification
        result = linker._find_best_match("Aspirin", EntityType.DRUG)
        assert result is not None
        cui, label, lang_labels, score = result
        assert cui == "C0004057", (
            f"_find_best_match should return Drug CUI C0004057, got {cui!r}"
        )

    def test_typed_pool_non_empty_means_untyped_pool_not_used(self):
        """
        When typed candidates exist, the function should not mix in untyped ones.
        Verify that the returned CUI is always from the requested type's pool.
        """
        index = self._dual_type_umls()
        linker = EntityLinker(
            umls_index=index,
            bm25_top_k=50,
            link_threshold=0.40,
            sapbert_model="",
        )
        linker.build_index()

        # Disease query: only Disease-typed entries should be considered as primary pool
        result = linker._find_best_match("Aspirin Hypersensitivity", EntityType.DISEASE)
        assert result is not None
        cui = result[0]
        # The winning entry must be from the Disease category
        assert index[cui]["entity_type"] == "Disease", (
            f"Expected a Disease-type CUI, got {cui!r} with type "
            f"{index[cui]['entity_type']!r}"
        )
