"""
LLM-based relation extractor using pydantic-ai (architecture doc §2.5).

Schema-guided approach: a closed set of medical relation types forces the LLM to
produce edges that are meaningful and auditable.
"""
from __future__ import annotations

import itertools
from typing import Any, Type

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from medgraphia.domain import Chunk, Entity, Relation, RelationType
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------

class ExtractedRelation(BaseModel):
    """A single relation extracted by the LLM."""
    source_cui: str = Field(description="CUI of the source entity")
    target_cui: str = Field(description="CUI of the target entity")
    relation_type: RelationType = Field(description="The semantic type of the relationship")
    evidence_text: str = Field(description="Verbatim excerpt from the text supporting this relation (max 200 chars)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")

class RelationExtractionResult(BaseModel):
    """The full set of relations extracted from a text chunk."""
    relations: list[ExtractedRelation] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# RelationExtractor
# ---------------------------------------------------------------------------

class RelationExtractor:
    """
    Extracts typed relations between linked entities using pydantic-ai.
    """

    def __init__(
        self,
        model: Any,
        min_confidence: float = 0.50,
        extracted_by: str = "",
    ) -> None:
        self._min_confidence = min_confidence
        self._extracted_by = extracted_by
        
        # Initialize the pydantic-ai Agent
        self._agent: Agent[None, RelationExtractionResult] = Agent(
            model,
            output_type=RelationExtractionResult,
            system_prompt=(
                "You are a medical knowledge graph expert specialising in pharmacology and clinical medicine. "
                "Your task is to extract semantic relations between pairs of medical entities from the provided text. "
                "Only extract relations that are explicitly or strongly implied by the evidence text. "
                "Do not create self-referential relations (source_cui == target_cui)."
            )
        )

    async def extract_chunk(self, chunk: Chunk) -> list[Relation]:
        """
        Extract relations from a single chunk using pydantic-ai.
        """
        linked = [e for e in chunk.entities if not e.cui.startswith("MENTION:")]
        if len(linked) < 2:
            return []

        # Prepare the context for the LLM
        entity_info = "\n".join([f"- {e.label} (CUI: {e.cui}, Type: {e.entity_type.value})" for e in linked])
        user_prompt = (
            f"TEXT CONTENT:\n{chunk.text}\n\n"
            f"IDENTIFIED ENTITIES:\n{entity_info}\n\n"
            "Please identify all valid relationships between these entities based ONLY on the text above."
        )

        try:
            # Run the agent - it handles JSON parsing, validation and retries automatically
            result = await self._agent.run(user_prompt)
            return self._process_result(result.output, chunk)
        except Exception as exc:
            logger.error("pydantic_ai_extraction_failed", chunk_id=chunk.chunk_id[:8], error=str(exc))
            return []

    async def extract_batch(self, chunks: list[Chunk]) -> list[Relation]:
        """Extract relations from all chunks sequentially."""
        all_relations: list[Relation] = []
        for chunk in chunks:
            relations = await self.extract_chunk(chunk)
            all_relations.extend(relations)
            if relations:
                logger.info("re_chunk_done", chunk_id=chunk.chunk_id[:8], count=len(relations))
        return all_relations

    def _process_result(self, data: RelationExtractionResult, chunk: Chunk) -> list[Relation]:
        """Convert Agent result to domain Relation objects."""
        relations: list[Relation] = []
        for item in data.relations:
            if item.source_cui == item.target_cui:
                continue
            if item.confidence < self._min_confidence:
                continue
                
            relations.append(
                Relation(
                    source_cui=item.source_cui,
                    target_cui=item.target_cui,
                    relation_type=item.relation_type,
                    evidence_text=item.evidence_text[:200],
                    source_id=chunk.source.source_id,
                    chunk_id=chunk.chunk_id,
                    confidence=item.confidence,
                    extracted_by=self._extracted_by,
                )
            )
        return relations

    async def write_relations_to_neo4j(self, relations: list[Relation]) -> None:
        """Write Relation edges to Neo4j."""
        if not relations:
            return
        try:
            from medgraphia.graph.queries import create_relation
            for rel in relations:
                await create_relation(rel)
            logger.info("re_neo4j_written", count=len(relations))
        except Exception as exc:
            logger.warning("re_neo4j_write_failed", count=len(relations), error=str(exc))

    @classmethod
    def from_settings(cls) -> "RelationExtractor":
        """Factory method to create the extractor from app settings."""
        from medgraphia.config import get_settings
        from medgraphia.llm.client import get_model

        cfg = get_settings()
        model = get_model()

        return cls(
            model=model,
            min_confidence=0.50,
            extracted_by=f"{cfg.llm_provider}/{cfg.llm_model}",
        )
