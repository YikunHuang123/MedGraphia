from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from medgraphia.domain.base import EntityType, RelationType

class Entity(BaseModel):
    """A medical concept node in the knowledge graph."""
    cui: str                          # UMLS Concept Unique Identifier
    label: str                        # Canonical English label
    entity_type: EntityType = EntityType.UNKNOWN
    lang_labels: dict[str, str] = Field(default_factory=dict)  # {"zh": "心肌梗死", "de": "Myokardinfarkt"}
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0

class Relation(BaseModel):
    """A directed semantic edge between two entities."""
    source_cui: str
    target_cui: str
    relation_type: RelationType
    evidence_text: str = ""
    source_id: str = ""
    chunk_id: str = ""
    confidence: float = 1.0
    extracted_by: str = ""            # model name / version that produced this edge
    properties: dict[str, Any] = Field(default_factory=dict)
