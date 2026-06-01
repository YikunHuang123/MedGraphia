"""
Multi-language medical NER pipeline.

Two-stage design (mirrors architecture doc §2.3):
  Stage 1 — GLiNER zero-shot coarse pass (fast, multilingual)
  Stage 2 — BERT domain model precision pass (optional; skipped if model unavailable)

Outputs Entity objects (with provisional MENTION: CUIs) attached to Chunk.entities.
Entity linking (UMLS CUI resolution) happens later in entity_linker.py.

Deduplication rule when spans from both stages overlap:
  • Same entity type → keep the span with the higher confidence score
  • Different entity type → keep both (ambiguity resolved downstream by EL)
"""
from __future__ import annotations

from medgraphia.domain import Chunk, Entity, EntityType, Language
from medgraphia.ingestion.ner._types import MentionSpan
from medgraphia.ingestion.ner.bert_ner import BertNER
from medgraphia.ingestion.ner.gliner_ner import GLiNERNER
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Provisional CUI prefix — replaced by entity_linker with real UMLS CUIs
_MENTION_PREFIX = "MENTION:"


class MedicalNERPipeline:
    """
    Combines GLiNER + optional BERT NER and converts MentionSpan → Entity.

    Usage::

        pipeline = MedicalNERPipeline()
        chunk_with_entities = pipeline.extract(chunk)
    """

    def __init__(
        self,
        gliner_model: str = "urchade/gliner_mediumv2.1",
        gliner_threshold: float = 0.45,
        bert_en_model: str = "d4data/biomedical-ner-all",
        bert_zh_model: str = "uer/roberta-base-finetuned-cluener2020-chinese",
        bert_de_model: str = "",
        min_confidence: float = 0.40,
    ) -> None:
        self._min_confidence = min_confidence
        self._gliner = GLiNERNER(model_name=gliner_model, threshold=gliner_threshold)
        self._bert = BertNER(
            en_model=bert_en_model,
            zh_model=bert_zh_model,
            de_model=bert_de_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, chunk: Chunk) -> Chunk:
        """
        Run NER on chunk.text and return a new Chunk with .entities populated.
        The original Chunk is not mutated (Pydantic model_copy).
        """
        if not chunk.text.strip():
            return chunk

        spans = self._run(chunk.text, chunk.language)
        entities = self._spans_to_entities(spans, chunk)

        if entities:
            logger.info(
                "ner_chunk_done",
                chunk_id=chunk.chunk_id,
                lang=chunk.language.value,
                entities=len(entities),
            )

        return chunk.model_copy(update={"entities": entities})

    def extract_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """Process a list of chunks.  Returns new chunks with entities populated."""
        return [self.extract(c) for c in chunks]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, text: str, language: Language) -> list[MentionSpan]:
        """Run both stages and return deduplicated, combined spans."""
        gliner_spans = self._gliner.predict(text, language)
        bert_spans = self._bert.predict(text, language)

        combined = _merge_spans(gliner_spans, bert_spans)
        
        # Final pass: merge adjacent fragments (handle subword splitting)
        final_merged = self._merge_adjacent_spans(combined)

        filtered = [s for s in final_merged if s.confidence >= self._min_confidence]
        return filtered

    def _merge_adjacent_spans(self, spans: list[MentionSpan]) -> list[MentionSpan]:
        if not spans:
            return []
        # Sort by start offset
        sorted_spans = sorted(spans, key=lambda s: s.start)
        merged = []
        
        current = sorted_spans[0]
        for next_span in sorted_spans[1:]:
            # If spans are of same type and very close (0 or 1 char gap)
            if (next_span.entity_type == current.entity_type and 
                next_span.start <= current.end + 1):
                # Merge them
                new_text = current.text + next_span.text
                current = MentionSpan.from_text(
                    text=new_text,
                    start=current.start,
                    end=next_span.end,
                    entity_type=current.entity_type,
                    confidence=max(current.confidence, next_span.confidence),
                    source=f"{current.source}+{next_span.source}"
                )
            else:
                merged.append(current)
                current = next_span
        merged.append(current)
        return merged

    def _spans_to_entities(self, spans: list[MentionSpan], chunk: Chunk) -> list[Entity]:
        """
        Convert deduplicated MentionSpan objects to Entity domain objects.

        Entities within the same chunk are deduplicated by (normalized, entity_type):
        if the same surface form appears multiple times, the highest-confidence
        instance is kept.
        """
        seen: dict[tuple[str, EntityType], float] = {}  # key → best confidence
        result: list[Entity] = []

        for span in spans:
            key = (span.normalized, span.entity_type)
            if key in seen and seen[key] >= span.confidence:
                continue
            seen[key] = span.confidence

            # Build a provisional CUI that entity_linker.py will replace
            prov_cui = f"{_MENTION_PREFIX}{span.normalized}"
            
            # Ensure the entity_type matches the domain's expected string case
            # (e.g., "drug" -> "Drug")
            etype = span.entity_type
            
            result.append(
                Entity(
                    cui=prov_cui,
                    label=span.text,
                    entity_type=etype,
                    confidence=span.confidence,
                    source_ids=[chunk.source.source_id],
                )
            )

        return result


# ---------------------------------------------------------------------------
# Span merging / deduplication helpers
# ---------------------------------------------------------------------------

def _merge_spans(
    primary: list[MentionSpan],
    secondary: list[MentionSpan],
) -> list[MentionSpan]:
    """
    Merge two span lists.  Overlapping spans of the same entity type: keep the
    one with higher confidence.  Non-overlapping and different-type overlapping
    spans are all kept.
    """
    all_spans = primary + secondary
    if not all_spans:
        return []

    # Sort by (start, -confidence) so higher-confidence comes first for same start
    all_spans.sort(key=lambda s: (s.start, -s.confidence))

    kept: list[MentionSpan] = []
    for candidate in all_spans:
        dominated = False
        for existing in kept:
            if existing.overlaps(candidate) and existing.entity_type == candidate.entity_type:
                # existing already has >= confidence (sorted order), skip candidate
                dominated = True
                break
        if not dominated:
            kept.append(candidate)

    return kept


def build_pipeline_from_settings() -> MedicalNERPipeline:
    """Convenience factory that reads config.py settings."""
    from medgraphia.config import get_settings
    cfg = get_settings()
    return MedicalNERPipeline(
        gliner_model=cfg.ner_gliner_model,
        gliner_threshold=cfg.ner_gliner_threshold,
        bert_en_model=cfg.ner_bert_en_model,
        bert_zh_model=cfg.ner_bert_zh_model,
        bert_de_model=cfg.ner_bert_de_model,
        min_confidence=cfg.ner_confidence_threshold,
    )
