"""
Query-side NER + Entity Linking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medgraphia.domain import Chunk, Entity, Language, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Placeholder source meta used when wrapping a raw query string as a Chunk
_QUERY_SOURCE = SourceMeta(
    source_id="__query__",
    source_title="User Query",
    source_version="",
)


@dataclass
class QueryEntities:
    """Result of query-side NER + entity linking."""
    query: str
    language: Language
    entities: list[Entity] = field(default_factory=list)
    # CUIs of entities that were successfully linked (no MENTION: prefix)
    linked_cuis: list[str] = field(default_factory=list)
    # Surface forms that could not be linked
    unlinked_mentions: list[str] = field(default_factory=list)

    @property
    def has_entities(self) -> bool:
        return len(self.entities) > 0

    @property
    def has_linked_entities(self) -> bool:
        return len(self.linked_cuis) > 0


class QueryNERLinker:
    """
    NER + Entity Linking for online query processing.

    Usage::

        linker = QueryNERLinker.from_settings()
        result = linker.analyze("What is the interaction between metformin and sitagliptin?")
        for cui in result.linked_cuis:
            ...  # feed into graph retriever
    """

    def __init__(
        self,
        ner_pipeline: Any | None = None,
        entity_linker: Any | None = None,
        default_language: Language = Language.EN,
    ) -> None:
        self._ner_pipeline = ner_pipeline
        self._entity_linker = entity_linker
        self._default_language = default_language
        self._initialized = False

    @classmethod
    def from_settings(cls) -> "QueryNERLinker":
        """Build from global settings; models are loaded lazily on first call."""
        return cls()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, query: str, language: Language | None = None) -> QueryEntities:
        """
        Run NER + EL on a raw query string.
        """
        lang = language or _detect_language(query) or self._default_language
        result = QueryEntities(query=query, language=lang)

        try:
            self._ensure_initialized()
            chunk = _query_to_chunk(query, lang)

            # Stage 1: NER
            chunk_with_ner = self._ner_pipeline.extract(chunk)
            if not chunk_with_ner.entities:
                logger.debug("query_ner_no_entities", query_len=len(query))
                return result

            # Stage 2: Entity Linking
            chunk_linked = self._entity_linker.link_chunk(chunk_with_ner)

            for entity in chunk_linked.entities:
                result.entities.append(entity)
                if not entity.cui.startswith("MENTION:"):
                    result.linked_cuis.append(entity.cui)
                else:
                    result.unlinked_mentions.append(entity.label)

            logger.info(
                "query_ner_done",
                query_len=len(query),
                lang=lang.value,
                entities=len(result.entities),
                linked=len(result.linked_cuis),
            )

        except Exception as exc:
            logger.warning("query_ner_failed", error=f"{type(exc).__name__}: {exc}")

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Lazily load NER pipeline and entity linker on first call."""
        if self._initialized:
            return

        if self._ner_pipeline is None:
            try:
                from medgraphia.ingestion.ner.pipeline import build_pipeline_from_settings
                self._ner_pipeline = build_pipeline_from_settings()
                logger.info("query_ner_pipeline_loaded")
            except Exception as exc:
                logger.warning("query_ner_pipeline_load_failed", error=str(exc))
                self._ner_pipeline = _NoopNER()

        if self._entity_linker is None:
            try:
                from medgraphia.ingestion.entity_linker import EntityLinker
                self._entity_linker = EntityLinker.from_settings()
                self._entity_linker.build_index()
                logger.info("query_el_loaded")
            except Exception as exc:
                logger.warning("query_el_load_failed", error=str(exc))
                self._entity_linker = _NoopLinker()

        self._initialized = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_to_chunk(text: str, language: Language) -> Chunk:
    """Wrap a raw query string into a minimal Chunk for the NER pipeline."""
    return Chunk(
        doc_id="__query__",
        source=_QUERY_SOURCE,
        language=language,
        text=text,
        section_path="query",
    )


def _detect_language(text: str) -> Language | None:
    """
    Heuristic language detection based on character set analysis.
    Returns None if indeterminate (caller falls back to default_language).
    """
    cjk_count = sum(1 for c in text if "一" <= c <= "鿿" or "㐀" <= c <= "䶿")
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return None

    cjk_ratio = cjk_count / total_alpha
    if cjk_ratio > 0.3:
        return Language.ZH

    # Check for German-specific characters
    de_chars = sum(1 for c in text.lower() if c in "äöüß")
    # Rough heuristic: German words are long and contain umlauts
    if de_chars > 0:
        return Language.DE

    return Language.EN


class _NoopNER:
    """Fallback NER that returns the chunk unchanged."""
    def extract(self, chunk: Chunk) -> Chunk:
        return chunk


class _NoopLinker:
    """Fallback entity linker that returns the chunk unchanged."""
    def link_chunk(self, chunk: Chunk) -> Chunk:
        return chunk
    def build_index(self) -> None:
        pass
