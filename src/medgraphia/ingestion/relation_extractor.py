"""
LLM-based relation extractor.
"""

from __future__ import annotations

from tqdm.asyncio import tqdm

from medgraphia.domain import Chunk, Relation, RelationType
from medgraphia.logger import get_logger
from medgraphia.prompts import ExtractedRelation

logger = get_logger(__name__)


class RelationExtractor:
    """
    Extracts typed relations between linked entities using DSPy.
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
        extracted_by: str = "",
    ) -> None:
        self._min_confidence = min_confidence
        self._extracted_by = extracted_by

    async def extract_chunk(self, chunk: Chunk) -> list[Relation]:
        """
        Extract relations from a single chunk using DSPy.
        """
        # to allow relation extraction on novel/unlinked concepts.
        entities = chunk.entities
        if len(entities) < 2:
            return []

        import dspy

        from medgraphia.llm.dspy_setup import get_lm
        from medgraphia.programs.extractor import get_extractor

        lm = get_lm("extractor")

        # Prepare labels/types context
        entity_info = "\n".join(
            [f"- {e.label} (CUI: {e.cui}, Type: {e.entity_type.value})" for e in entities]
        )
        allowed_relations = ", ".join([rt.value for rt in RelationType])

        try:
            with dspy.context(lm=lm):
                # Use the centralized extractor program (supports compiled few-shots)
                program = get_extractor()
                
                # Use dspy.asyncify to safely offload to a thread while preserving dspy.context(lm=lm).
                async_program = dspy.asyncify(program)
                prediction = await async_program(
                    text_content=chunk.text,
                    entities=entity_info,
                    allowed_relations=allowed_relations,
                )
            return self._process_result(prediction.relations, chunk)
        except Exception as exc:
            logger.error("extraction_failed", chunk_id=chunk.chunk_id[:8], error=str(exc))
            return []

    async def extract_batch(self, chunks: list[Chunk], max_workers: int = 5) -> list[Relation]:
        """Extract relations from all chunks in parallel."""
        import asyncio

        semaphore = asyncio.Semaphore(max_workers)

        async def _task(chunk: Chunk):
            async with semaphore:
                return await self.extract_chunk(chunk)

        tasks = [_task(c) for c in chunks]
        results = await tqdm.gather(*tasks, desc="Extracting relations", unit="chunk")

        all_relations: list[Relation] = []
        for relations in results:
            all_relations.extend(relations)
        return all_relations

    def _process_result(self, data: list[ExtractedRelation], chunk: Chunk) -> list[Relation]:
        """Convert DSPy output to domain Relation objects."""
        relations: list[Relation] = []
        valid_types = {rt.value for rt in RelationType}

        for item in data:
            if item.source_cui == item.target_cui:
                continue
            if item.relation_type not in valid_types:
                continue
            if item.confidence < self._min_confidence:
                continue

            relations.append(
                Relation(
                    source_cui=item.source_cui,
                    target_cui=item.target_cui,
                    relation_type=RelationType(item.relation_type),
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
            import asyncio

            # Concurrently write relations to avoid sequential N+1 network delay
            tasks = [create_relation(rel) for rel in relations]
            await asyncio.gather(*tasks)
            
            logger.info("re_neo4j_written", count=len(relations))
        except Exception as exc:
            logger.warning("re_neo4j_write_failed", count=len(relations), error=str(exc))

    @classmethod
    def from_settings(cls) -> RelationExtractor:
        from medgraphia.config import get_settings

        cfg = get_settings()
        provider = cfg.extractor_llm_provider or cfg.default_llm_provider
        model = cfg.extractor_llm_model or cfg.default_llm_model
        return cls(
            min_confidence=0.50,
            extracted_by=f"{provider}/{model}",
        )
