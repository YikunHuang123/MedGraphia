from __future__ import annotations

from pydantic import BaseModel, Field

from medgraphia.domain.base import EntityType


class Entity(BaseModel):
    """A medical concept node in the knowledge graph."""

    cui: str  # UMLS Concept Unique Identifier
    label: str  # Canonical English label
    entity_type: EntityType = EntityType.UNKNOWN
    lang_labels: dict[str, str] = Field(
        default_factory=dict
    )  # {"zh": "心肌梗死", "de": "Myokardinfarkt"}
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    start_char: int | None = None
    end_char: int | None = None
