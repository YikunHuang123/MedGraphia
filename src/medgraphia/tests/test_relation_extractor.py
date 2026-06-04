"""
Phase 4 tests: RelationExtractor (pydantic-ai version).

All tests are pure-Python unit tests:
  - LLM calls are mocked at the pydantic-ai Agent level.
  - No litellm, no Neo4j, no GPU required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import Agent

from medgraphia.domain import Chunk, Entity, EntityType, Language, Relation, RelationType, SourceMeta
from medgraphia.ingestion.relation_extractor import (
    ExtractedRelation,
    RelationExtractionResult,
    RelationExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source() -> SourceMeta:
    return SourceMeta(source_id="test:re", source_title="RE Test")


def _make_chunk(
    text: str,
    entities: list[Entity],
    language: Language = Language.EN,
) -> Chunk:
    return Chunk(
        doc_id="test-doc-re",
        source=_source(),
        language=language,
        section_path="Test",
        text=text,
        entities=entities,
    )


def _entity(
    cui: str,
    label: str,
    entity_type: EntityType = EntityType.DRUG,
) -> Entity:
    return Entity(
        cui=cui,
        label=label,
        entity_type=entity_type,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# RelationExtractor Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestExtractChunk:
    async def test_single_entity_returns_empty(self):
        extractor = RelationExtractor(model=MagicMock())
        chunk = _make_chunk("Metformin is a drug.", [_entity("C0025598", "Metformin")])
        assert await extractor.extract_chunk(chunk) == []

    async def test_no_linked_entities_returns_empty(self):
        extractor = RelationExtractor(model=MagicMock())
        # Entities with MENTION: prefix are not yet linked
        mention = Entity(cui="MENTION:metformin", label="Metformin",
                         entity_type=EntityType.DRUG, confidence=0.9)
        chunk = _make_chunk("text", [mention])
        assert await extractor.extract_chunk(chunk) == []

    async def test_treats_relation_extracted(self):
        # Mock result from Agent
        mock_data = RelationExtractionResult(
            relations=[
                ExtractedRelation(
                    source_cui="C0025598",
                    target_cui="C0011860",
                    relation_type=RelationType.TREATS,
                    evidence_text="Metformin treats type 2 diabetes.",
                    confidence=0.92,
                )
            ]
        )
        
        mock_agent_run = AsyncMock()
        mock_agent_run.return_value = MagicMock(data=mock_data)

        extractor = RelationExtractor(model=MagicMock())
        # Replace the real agent's run with our mock
        extractor._agent.run = mock_agent_run

        entities = [
            _entity("C0025598", "Metformin", EntityType.DRUG),
            _entity("C0011860", "Diabetes Mellitus, Type 2", EntityType.DISEASE),
        ]
        chunk = _make_chunk("Metformin treats type 2 diabetes.", entities)
        relations = await extractor.extract_chunk(chunk)

        assert len(relations) == 1
        rel = relations[0]
        assert rel.source_cui == "C0025598"
        assert rel.target_cui == "C0011860"
        assert rel.relation_type == RelationType.TREATS
        assert rel.confidence == pytest.approx(0.92)
        mock_agent_run.assert_called_once()

    async def test_multiple_relations(self):
        mock_data = RelationExtractionResult(
            relations=[
                ExtractedRelation(source_cui="C0025598", target_cui="C0011860",
                                  relation_type=RelationType.TREATS, evidence_text="ev1", confidence=0.9),
                ExtractedRelation(source_cui="C0004057", target_cui="C0025598",
                                  relation_type=RelationType.INTERACTS_WITH, evidence_text="ev2", confidence=0.8),
            ]
        )
        mock_agent_run = AsyncMock(return_value=MagicMock(data=mock_data))
        extractor = RelationExtractor(model=MagicMock())
        extractor._agent.run = mock_agent_run

        entities = [
            _entity("C0025598", "Metformin", EntityType.DRUG),
            _entity("C0011860", "Diabetes", EntityType.DISEASE),
            _entity("C0004057", "Aspirin", EntityType.DRUG),
        ]
        chunk = _make_chunk("text", entities)
        relations = await extractor.extract_chunk(chunk)
        assert len(relations) == 2

    async def test_low_confidence_filtered(self):
        mock_data = RelationExtractionResult(
            relations=[
                ExtractedRelation(source_cui="C001", target_cui="C002",
                                  relation_type=RelationType.TREATS, evidence_text="ev", confidence=0.3),
            ]
        )
        mock_agent_run = AsyncMock(return_value=MagicMock(data=mock_data))
        extractor = RelationExtractor(model=MagicMock(), min_confidence=0.5)
        extractor._agent.run = mock_agent_run

        entities = [_entity("C001", "A"), _entity("C002", "B", EntityType.DISEASE)]
        chunk = _make_chunk("text", entities)
        assert await extractor.extract_chunk(chunk) == []


@pytest.mark.asyncio
class TestExtractBatch:
    async def test_batch_aggregates_all_relations(self):
        mock_data = RelationExtractionResult(
            relations=[
                ExtractedRelation(source_cui="C001", target_cui="C002",
                                  relation_type=RelationType.TREATS, evidence_text="ev", confidence=0.9),
            ]
        )
        mock_agent_run = AsyncMock(return_value=MagicMock(data=mock_data))
        extractor = RelationExtractor(model=MagicMock())
        extractor._agent.run = mock_agent_run

        entities = [_entity("C001", "A"), _entity("C002", "B", EntityType.DISEASE)]
        chunks = [_make_chunk("text", entities) for _ in range(3)]
        relations = await extractor.extract_batch(chunks)
        assert len(relations) == 3

    async def test_batch_handles_chunk_error_gracefully(self):
        mock_agent_run = AsyncMock()
        mock_agent_run.side_effect = [
            RuntimeError("Agent failed"),
            MagicMock(data=RelationExtractionResult(relations=[
                ExtractedRelation(source_cui="C001", target_cui="C002",
                                  relation_type=RelationType.TREATS, evidence_text="ev", confidence=0.9)
            ]))
        ]
        extractor = RelationExtractor(model=MagicMock())
        extractor._agent.run = mock_agent_run

        entities = [_entity("C001", "A"), _entity("C002", "B", EntityType.DISEASE)]
        chunks = [_make_chunk("text", entities) for _ in range(2)]
        relations = await extractor.extract_batch(chunks)
        assert len(relations) == 1


@pytest.mark.asyncio
class TestNeo4jWrite:
    async def test_write_empty_relations_is_noop(self):
        extractor = RelationExtractor(model=MagicMock())
        await extractor.write_relations_to_neo4j([])

    async def test_write_relations_neo4j_unavailable_no_raise(self):
        extractor = RelationExtractor(model=MagicMock())
        rel = Relation(
            source_cui="C001",
            target_cui="C002",
            relation_type=RelationType.TREATS,
            chunk_id="ck1",
        )
        with patch("medgraphia.graph.queries.create_relation", side_effect=ConnectionError("neo4j down")):
            await extractor.write_relations_to_neo4j([rel])
