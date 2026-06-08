"""
Phase 4 tests: CommunityBuilder (pydantic-ai version).

All tests are pure-Python unit tests:
  - LLM calls are mocked at the pydantic-ai Agent level.
  - No Neo4j required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medgraphia.domain import Community, Entity, EntityType, Relation, RelationType
from medgraphia.ingestion.community_builder import (
    CommunityBuilder,
    CommunitySummaryResult,
    _format_concepts,
    _format_relations,
    _stable_community_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(cui: str, label: str, entity_type: EntityType = EntityType.DRUG) -> Entity:
    return Entity(cui=cui, label=label, entity_type=entity_type, confidence=0.9)


def _relation(src: str, tgt: str, rel_type: RelationType = RelationType.TREATS) -> Relation:
    return Relation(
        source_cui=src,
        target_cui=tgt,
        relation_type=rel_type,
        confidence=1.0,
    )


def _builder(
    mock_data: CommunitySummaryResult | None = None, min_size: int = 2
) -> CommunityBuilder:
    """Helper to create a builder with a mocked agent."""
    extractor = CommunityBuilder(model=MagicMock(), min_size=min_size)

    if mock_data:
        mock_run = AsyncMock()
        mock_run.return_value = MagicMock(data=mock_data)
        extractor._agent.run = mock_run

    return extractor


# ---------------------------------------------------------------------------
# Format Helpers
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_format_concepts_contains_labels(self):
        entity_map = {"C001": _entity("C001", "Metformin")}
        text = _format_concepts({"C001"}, entity_map)
        assert "Metformin" in text
        assert "C001" in text

    def test_format_relations_contains_labels(self):
        entity_map = {
            "C001": _entity("C001", "Metformin"),
            "C002": _entity("C002", "Diabetes"),
        }
        rel = _relation("C001", "C002")
        text = _format_relations([rel], entity_map)
        assert "Metformin" in text
        assert "Diabetes" in text
        assert "TREATS" in text

    def test_format_relations_capped_at_30(self):
        relations = [_relation("C001", "C002") for _ in range(40)]
        text = _format_relations(relations, {})
        lines = [l for l in text.splitlines() if "TREATS" in l]
        assert len(lines) <= 30


# ---------------------------------------------------------------------------
# Community Identification
# ---------------------------------------------------------------------------


class TestCommunityIdentification:
    def test_stable_id_is_deterministic(self):
        cuis = {"C001", "C002", "C003"}
        id1 = _stable_community_id(cuis)
        id2 = _stable_community_id(cuis)
        assert id1 == id2
        assert id1.startswith("comm_")

    def test_stable_id_ignores_input_order(self):
        assert _stable_community_id({"A", "B"}) == _stable_community_id({"B", "A"})


# ---------------------------------------------------------------------------
# CommunityBuilder.build_from_relations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBuildFromRelations:
    async def test_no_relations_returns_empty(self):
        builder = _builder()
        assert await builder.build_from_relations([], {}) == []

    async def test_communities_returned_as_community_objects(self):
        relations = [_relation("C001", "C002"), _relation("C003", "C004")]
        builder = _builder(min_size=2)

        # Mock the summarizer to avoid LLM calls
        builder._summarise = AsyncMock(return_value="Summary")

        communities = await builder.build_from_relations(relations, {})
        assert len(communities) >= 2
        assert isinstance(communities[0], Community)

    async def test_min_size_filters_small_communities(self):
        # 2 nodes form one community of size 2
        relations = [_relation("C001", "C002")]
        builder = _builder(min_size=3)  # require at least 3 members
        builder._summarise = AsyncMock(return_value="Summary")

        communities = await builder.build_from_relations(relations, {})
        assert len(communities) == 0

    async def test_llm_summary_integrated(self):
        mock_data = CommunitySummaryResult(
            summary="Diabetes treatment group.",
            explanation="Contains metformin and related drugs.",
            clinical_relevance="High relevance for T2DM management.",
        )
        builder = _builder(mock_data=mock_data, min_size=2)
        relations = [_relation("C001", "C002")]

        communities = await builder.build_from_relations(relations, {})
        assert len(communities) == 1
        summary = communities[0].summary
        assert "Diabetes treatment group" in summary
        assert "High relevance" in summary


# ---------------------------------------------------------------------------
# Neo4j write degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNeo4jWrite:
    async def test_write_empty_communities_is_noop(self):
        builder = _builder()
        await builder.write_communities_to_neo4j([])

    async def test_write_communities_neo4j_unavailable_no_raise(self):
        builder = _builder()
        comm = Community(community_id="c1", members=["C1"], summary="S", level=0)
        with patch(
            "medgraphia.graph.queries.upsert_community", side_effect=ConnectionError("down")
        ):
            await builder.write_communities_to_neo4j([comm])
