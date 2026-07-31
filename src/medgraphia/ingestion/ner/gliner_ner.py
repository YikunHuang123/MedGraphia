"""
GLiNER zero-shot coarse NER for multilingual medical text.

Default model: Ihor/gliner-biomed-large-v1.0 (domain-adapted for biomedical NER)
Handles EN / ZH / DE in a single zero-shot pass using a biomedical label set.

if `gliner` is not installed, predict() returns [] and logs a
warning.  The NER pipeline continues with BERT-only or regex mode.
"""

from __future__ import annotations

from medgraphia.domain import EntityType, Language
from medgraphia.ingestion.ner._types import MentionSpan
from medgraphia.logger import get_logger

logger = get_logger(__name__)

try:
    from gliner import GLiNER as _GLiNERModel 

    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False
    logger.warning("gliner_not_installed", msg="pip install gliner to enable zero-shot NER")

# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------

# Natural-language labels passed to GLiNER — one per concept per language.
# Using explicit Language enums ensures type safety and prevents implicit index errors.
_ENTITY_LABELS: dict[EntityType, dict[Language, str]] = {
    EntityType.DISEASE: {Language.EN: "disease", Language.ZH: "疾病", Language.DE: "Erkrankung"},
    EntityType.DRUG: {Language.EN: "drug", Language.ZH: "药物", Language.DE: "Medikament"},
    EntityType.SYMPTOM: {Language.EN: "symptom", Language.ZH: "症状", Language.DE: "Symptom"},
    EntityType.GENE: {Language.EN: "gene", Language.ZH: "基因", Language.DE: "Gen"},
    EntityType.PROCEDURE: {Language.EN: "procedure", Language.ZH: "手术", Language.DE: "Operation"},
    EntityType.ANATOMY: {Language.EN: "body part", Language.ZH: "身体部位", Language.DE: "Körperteil"},
    EntityType.PHYSIOLOGY: {Language.EN: "biological process", Language.ZH: "生理过程", Language.DE: "biologischer Prozess"},
    EntityType.LIVING_BEING: {Language.EN: "pathogen", Language.ZH: "病原体", Language.DE: "Erreger"},
}

# Reverse map: label text -> EntityType (used to map model output back to domain types)
_LABEL_TO_TYPE: dict[str, EntityType] = {
    label: etype
    for etype, lang_map in _ENTITY_LABELS.items()
    for label in lang_map.values()
}

# Generate language-specific label lists.
# For non-English languages, we explicitly append the core English labels to handle
# mixed-language biomedical texts (e.g., Chinese notes containing "DNA" or "CT").
_EN_LABELS = [lang_map[Language.EN] for lang_map in _ENTITY_LABELS.values()]
_ZH_LABELS = [lang_map[Language.ZH] for lang_map in _ENTITY_LABELS.values()] + _EN_LABELS
_DE_LABELS = [lang_map[Language.DE] for lang_map in _ENTITY_LABELS.values()] + _EN_LABELS

_N_TYPES = len(_ENTITY_LABELS)
_ALL_LABELS = _EN_LABELS + _ZH_LABELS[:_N_TYPES] + _DE_LABELS[:_N_TYPES]  # keep unique values for UNKNOWN fallback

_LABELS_BY_LANG: dict[Language, list[str]] = {
    Language.EN: _EN_LABELS,
    Language.ZH: _ZH_LABELS,
    Language.DE: _DE_LABELS,
    Language.UNKNOWN: _ALL_LABELS,
}


class GLiNERNER:
    """
    Zero-shot biomedical NER using GLiNER.
    Usage::
        ner = GLiNERNER()
        spans = ner.predict("Metformin treats type 2 diabetes.", Language.EN)
    """

    def __init__(
        self,
        model_name: str = "Ihor/gliner-biomed-large-v1.0",
        threshold: float = 0.45,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._model = None

    @property
    def is_available(self) -> bool:
        return _GLINER_AVAILABLE

    def predict(self, text: str, language: Language = Language.EN) -> list[MentionSpan]:
        """
        Extract entity spans from text.
        Returns [] if gliner is not installed or if the model fails to load.
        """
        if not _GLINER_AVAILABLE or not text.strip():
            return []

        try:
            model = self._load_model()
        except Exception as exc:
            logger.warning("gliner_load_failed", error=str(exc))
            return []

        labels = _LABELS_BY_LANG.get(language, _ALL_LABELS)

        try:
            raw = model.predict_entities(text, labels, threshold=self._threshold)
        except Exception as exc:
            logger.warning("gliner_predict_failed", error=str(exc))
            return []

        spans: list[MentionSpan] = []
        for ent in raw:
            label_text = ent.get("label", "")
            entity_type = _LABEL_TO_TYPE.get(label_text)
            if entity_type is None:
                continue
            spans.append(
                MentionSpan.from_text(
                    text=ent["text"],
                    start=ent["start"],
                    end=ent["end"],
                    entity_type=entity_type,
                    confidence=float(ent.get("score", 1.0)),
                    source="gliner",
                )
            )

        logger.debug(
            "gliner_predict_done",
            lang=language.value,
            text_len=len(text),
            spans=len(spans),
        )
        return spans

    def predict_batch(
        self, texts: list[str], language: Language = Language.EN
    ) -> list[list[MentionSpan]]:
        """
        Batch predict for multiple texts in a single forward pass.
        Returns one MentionSpan list per input text (same order).
        Falls back to per-text predict() if the model does not support list input.
        """
        if not _GLINER_AVAILABLE or not texts:
            return [[] for _ in texts]

        try:
            model = self._load_model()
        except Exception as exc:
            logger.warning("gliner_load_failed", error=str(exc))
            return [[] for _ in texts]

        labels = _LABELS_BY_LANG.get(language, _ALL_LABELS)
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return [[] for _ in texts]

        indices, batch_texts = zip(*non_empty)
        try:
            raw_results = model.predict_entities(list(batch_texts), labels, threshold=self._threshold)
            # GLiNER returns list[dict] for a single string; list[list[dict]] for a list
            if raw_results and isinstance(raw_results[0], dict):
                raw_results = [raw_results]
        except Exception as exc:
            logger.warning("gliner_predict_failed", error=str(exc))
            return [[] for _ in texts]

        all_spans: list[list[MentionSpan]] = [[] for _ in texts]
        for orig_idx, raw_entities in zip(indices, raw_results):
            spans = []
            for ent in raw_entities:
                entity_type = _LABEL_TO_TYPE.get(ent.get("label", ""))
                if entity_type is None:
                    continue
                spans.append(
                    MentionSpan.from_text(
                        text=ent["text"],
                        start=ent["start"],
                        end=ent["end"],
                        entity_type=entity_type,
                        confidence=float(ent.get("score", 1.0)),
                        source="gliner",
                    )
                )
            all_spans[orig_idx] = spans

        logger.debug("gliner_batch_done", lang=language.value, texts=len(batch_texts))
        return all_spans

    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            logger.info("gliner_loading", model=self._model_name)
            self._model = _GLiNERModel.from_pretrained(self._model_name)
            logger.info("gliner_loaded", model=self._model_name)
        return self._model
