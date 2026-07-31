"""
Internal NER span type.  Intermediate representation between raw model output
and the domain Entity — lives only inside the ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from medgraphia.domain import EntityType


@dataclass(frozen=True)
class MentionSpan:
    """An entity mention span extracted from text by an NER model."""

    text: str  # Surface form
    normalized: str  # Lower-stripped form used for deduplication
    start: int  # Inclusive char offset in source text
    end: int  # Exclusive char offset
    entity_type: EntityType
    confidence: float = 1.0
    source: str = "unknown"  # "gliner" | "bert"

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text: str,
        start: int,
        end: int,
        entity_type: EntityType,
        confidence: float = 1.0,
        source: str = "unknown",
    ) -> MentionSpan:
        return cls(
            text=text,
            normalized=text.strip().lower(),
            start=start,
            end=end,
            entity_type=entity_type,
            confidence=confidence,
            source=source,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def overlaps(self, other: MentionSpan) -> bool:
        """Return True when this span's character range overlaps with other."""
        return self.start < other.end and other.start < self.end
