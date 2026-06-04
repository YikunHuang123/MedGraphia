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
from tqdm.asyncio import tqdm
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

# ---------------------------------------------------------------------------
# RelationExtractor
# ---------------------------------------------------------------------------

class RelationExtractor:
    """
    Extracts typed relations between linked entities using a robust JSON-parsing approach.
    Bypasses tool-calling to improve stability on providers like Groq.
    """

    def __init__(
        self,
        model: Any,
        min_confidence: float = 0.50,
        extracted_by: str = "",
    ) -> None:
        self._min_confidence = min_confidence
        self._extracted_by = extracted_by
        
        # Build the list of allowed relation types for the prompt
        allowed_relations = ", ".join([f"'{rt.value}'" for rt in RelationType])
        
        # Initialize the pydantic-ai Agent with str output type
        self._agent: Agent[None, str] = Agent(
            model,
            system_prompt=(
                "You are a medical knowledge graph expert specialising in pharmacology and clinical medicine.\n"
                "Your task is to extract semantic relations between pairs of medical entities from the provided text.\n\n"
                "STRICT OUTPUT FORMAT:\n"
                "Return a JSON list of objects within a ```json code block. Each object must have:\n"
                "- source_cui: (string)\n"
                "- target_cui: (string)\n"
                "- relation_type: (string, MUST be one of the allowed types)\n"
                "- evidence_text: (string, verbatim excerpt)\n"
                "- confidence: (float, 0-1)\n\n"
                f"ALLOWED RELATION TYPES: {allowed_relations}.\n"
                "NO SELF-LOOPS: source_cui must not equal target_cui.\n"
                "ONLY include relations explicitly supported by the text."
            )
        )

    async def extract_chunk(self, chunk: Chunk) -> list[Relation]:
        """
        Extract relations from a single chunk.
        """
        linked = [e for e in chunk.entities if not e.cui.startswith("MENTION:")]
        if len(linked) < 2:
            return []

        # Prepare the context for the LLM
        entity_info = "\n".join([f"- {e.label} (CUI: {e.cui}, Type: {e.entity_type.value})" for e in linked])
        user_prompt = (
            f"TEXT CONTENT:\n{chunk.text}\n\n"
            f"IDENTIFIED ENTITIES:\n{entity_info}\n\n"
            "Extract relations in JSON format:"
        )

        try:
            # Run the agent
            result = await self._agent.run(user_prompt)
            data = self._manual_parse(result.output)
            return self._process_result(data, chunk)
        except Exception as exc:
            logger.error("extraction_failed", chunk_id=chunk.chunk_id[:8], error=str(exc))
            return []

    def _manual_parse(self, text: str) -> list[dict[str, Any]]:
        """Robustly extract JSON list from markdown code blocks or raw text."""
        import json
        import re
        
        # 1. Clean up potential leading/trailing non-JSON text
        text = text.strip()
        
        # 2. Try to find content within markdown fences
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        content = match.group(1) if match else text
        
        # 3. Handle cases where the model wraps it in {"relations": [...]}
        # even though we asked for a direct list
        try:
            temp_data = json.loads(content)
            if isinstance(temp_data, dict) and "relations" in temp_data:
                return temp_data["relations"]
            if isinstance(temp_data, list):
                return temp_data
        except:
            pass
            
        # 4. Locate the first [ and last ] as a last resort
        start = content.find('[')
        end = content.rfind(']')
        
        if start != -1 and end != -1:
            json_str = content[start:end+1]
            try:
                return json.loads(json_str)
            except Exception:
                pass
        
        logger.warning("manual_json_parse_failed", snippet=text[:100])
        return []

    async def extract_batch(self, chunks: list[Chunk], max_workers: int = 5) -> list[Relation]:
        """Extract relations from all chunks in parallel with a concurrency limit."""
        import asyncio
        semaphore = asyncio.Semaphore(max_workers)

        async def _task(chunk: Chunk):
            async with semaphore:
                return await self.extract_chunk(chunk)

        # Create all tasks
        tasks = [_task(c) for c in chunks]
        
        # Run with progress bar using tqdm.gather
        results = await tqdm.gather(*tasks, desc="Extracting relations", unit="chunk")
        
        # Flatten results
        all_relations: list[Relation] = []
        for relations in results:
            all_relations.extend(relations)
            
        return all_relations

    def _process_result(self, data: list[dict[str, Any]], chunk: Chunk) -> list[Relation]:
        """Convert raw dicts to domain Relation objects with validation."""
        relations: list[Relation] = []
        valid_types = {rt.value for rt in RelationType}
        
        for item in data:
            try:
                # Validation
                src = item.get("source_cui")
                tgt = item.get("target_cui")
                rel_type = item.get("relation_type")
                
                if not all([src, tgt, rel_type]):
                    continue
                if src == tgt:
                    continue
                if rel_type not in valid_types:
                    continue
                    
                relations.append(
                    Relation(
                        source_cui=src,
                        target_cui=tgt,
                        relation_type=RelationType(rel_type),
                        evidence_text=str(item.get("evidence_text", ""))[:200],
                        source_id=chunk.source.source_id,
                        chunk_id=chunk.chunk_id,
                        confidence=float(item.get("confidence", 1.0)),
                        extracted_by=self._extracted_by,
                    )
                )
            except Exception:
                continue
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
