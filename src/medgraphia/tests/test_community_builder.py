"""
CommunityBuilder tests.

All tests are pure-Python unit tests:
  - LLM summarization is bypassed via monkey-patching `_summarise`.
  - No Neo4j required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from medgraphia.domain import Chunk, Community, Entity, EntityType, Language, SourceMeta
from medgraphia.ingestion.community_builder import (
    CommunityBuilder,
    _format_concepts,
    _format_cooccurrences,
    _stable_community_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(cui: str, label: str, entity_type: EntityType = EntityType.DRUG) -> Entity:
    return Entity(cui=cui, label=label, entity_type=entity_type, confidence=0.9)


def _chunk(chunk_id: str, entities: list[Entity]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc:test",
        source=SourceMeta(source_id="test", source_title="Test"),
        language=Language.EN,
        section_path="test",
        text="test chunk text",
        entities=entities,
    )


def _builder(min_size: int = 2) -> CommunityBuilder:
    return CommunityBuilder(min_size=min_size)


# ---------------------------------------------------------------------------
# Format Helpers
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_format_concepts_contains_labels(self):
        entity_map = {"C001": _entity("C001", "Metformin")}
        text = _format_concepts({"C001"}, entity_map)
        assert "Metformin" in text
        assert "C001" in text

    def test_format_cooccurrences_contains_labels(self):
        entity_map = {
            "C001": _entity("C001", "Metformin"),
            "C002": _entity("C002", "Diabetes"),
        }
        text = _format_cooccurrences([("C001", "C002", 3)], entity_map)
        assert "Metformin" in text
        assert "Diabetes" in text
        assert "3 shared chunk" in text

    def test_format_cooccurrences_capped_at_30(self):
        pairs = [("C001", f"C{i:03d}", 1) for i in range(40)]
        text = _format_cooccurrences(pairs, {})
        lines = [l for l in text.splitlines() if "co-occurs" in l]
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
# CommunityBuilder.build_from_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBuildFromChunks:
    async def test_no_chunks_returns_empty(self):
        builder = _builder()
        assert await builder.build_from_chunks([], {}) == []

    async def test_communities_returned_as_community_objects(self):
        chunks = [
            _chunk("c1", [_entity("C001", "Metformin"), _entity("C002", "Diabetes")]),
            _chunk("c2", [_entity("C003", "Aspirin"), _entity("C004", "Headache")]),
        ]
        builder = _builder(min_size=2)
        builder._summarise = AsyncMock(return_value="Summary")

        communities = await builder.build_from_chunks(chunks, {})
        assert len(communities) >= 2
        assert isinstance(communities[0], Community)

    async def test_min_size_filters_small_communities(self):
        # 2 co-mentioned entities form one community of size 2
        chunks = [_chunk("c1", [_entity("C001", "Metformin"), _entity("C002", "Diabetes")])]
        builder = _builder(min_size=3)  # require at least 3 members
        builder._summarise = AsyncMock(return_value="Summary")

        communities = await builder.build_from_chunks(chunks, {})
        assert len(communities) == 0

    async def test_llm_summary_integrated(self):
        builder = _builder(min_size=2)
        builder._summarise = AsyncMock(
            return_value="Diabetes treatment group. High relevance for T2DM management."
        )
        chunks = [_chunk("c1", [_entity("C001", "Metformin"), _entity("C002", "Diabetes")])]

        communities = await builder.build_from_chunks(chunks, {})
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
