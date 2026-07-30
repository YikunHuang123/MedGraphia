from __future__ import annotations

from dataclasses import dataclass, field

from medgraphia.domain import Chunk, Entity, RelationType


@dataclass
class EntityPair:
    """An intermediate representation for BERT RE inference."""
    chunk: Chunk
    source: Entity
    target: Entity
    marked_text: str  # e.g., "... [E1] Aspirin [/E1] for [E2] headache [/E2] ..."


@dataclass
class LabeledPair:
    """An EntityPair with a ground-truth relation label — used for training data."""
    marked_text: str
    label: str                     # RelationType value, or "NONE"
    lang: str
    source_cui: str
    source_label: str
    source_type: str
    target_cui: str
    target_label: str
    target_type: str
    chunk_id: str
    confidence: float = 0.0        # LLM confidence for positive labels; 0.0 for NONE

    def to_dict(self) -> dict:
        return {
            "text": self.marked_text,
            "label": self.label,
            "lang": self.lang,
            "source_cui": self.source_cui,
            "source_label": self.source_label,
            "source_type": self.source_type,
            "target_cui": self.target_cui,
            "target_label": self.target_label,
            "target_type": self.target_type,
            "chunk_id": self.chunk_id,
            "confidence": self.confidence,
        }


@dataclass
class RelationPrediction:
    """BERT RE model output for a single entity pair — used during inference."""
    source_cui: str
    target_cui: str
    relation_type: RelationType
    confidence: float
    marked_text: str = ""
