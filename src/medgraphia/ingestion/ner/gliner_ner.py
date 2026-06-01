"""
GLiNER zero-shot coarse NER for multilingual medical text.

Default model: urchade/gliner_mediumv2.1
Handles EN / ZH / DE in a single zero-shot pass using a biomedical label set.

Graceful degradation: if `gliner` is not installed, predict() returns [] and logs a
warning.  The NER pipeline continues with BERT-only or regex mode.
"""
from __future__ import annotations

from medgraphia.domain import EntityType, Language
from medgraphia.ingestion.ner._types import MentionSpan
from medgraphia.logger import get_logger

logger = get_logger(__name__)

try:
    from gliner import GLiNER as _GLiNERModel  # type: ignore[import]
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False
    logger.warning("gliner_not_installed", msg="pip install gliner to enable zero-shot NER")

# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------

# Natural-language labels passed to GLiNER — one per concept type per language.
# Multilingual labels improve recall for DE / ZH documents processed by the same model.
_ENTITY_LABELS: dict[EntityType, list[str]] = {
    EntityType.DISEASE: ["disease", "疾病", "Erkrankung"],
    EntityType.DRUG: ["drug", "药物", "Medikament"],
    EntityType.SYMPTOM: ["symptom", "症状", "Symptom"],
    EntityType.GENE: ["gene", "基因", "Gen"],
    EntityType.PROCEDURE: ["procedure", "手术", "Operation"],
}

_ALL_LABELS: list[str] = [
    label for labels in _ENTITY_LABELS.values() for label in labels
]

# Reverse map: label text → EntityType (built at import time)
_LABEL_TO_TYPE: dict[str, EntityType] = {
    label: etype
    for etype, labels in _ENTITY_LABELS.items()
    for label in labels
}

# Language-specific subsets keep the label list shorter for focused documents.
_EN_LABELS = [l for l in _ALL_LABELS if not any("一" <= c <= "鿿" for c in l)
              and not any(c in "äöüÄÖÜß" for c in l)]
_ZH_LABELS = [l for l in _ALL_LABELS if any("一" <= c <= "鿿" for c in l)] + \
             [l for l in _EN_LABELS[:12]]  # keep core English labels too
_DE_LABELS = [l for l in _ALL_LABELS if any(c in "äöüÄÖÜß" for c in l)] + \
             [l for l in _EN_LABELS[:12]]

_LABELS_BY_LANG: dict[Language, list[str]] = {
    Language.EN:      _EN_LABELS,
    Language.ZH:      _ZH_LABELS,
    Language.DE:      _DE_LABELS,
    Language.UNKNOWN: _ALL_LABELS,
}


class GLiNERNER:
    """
    Zero-shot biomedical NER using GLiNER.

    Model is loaded lazily on the first predict() call.

    Usage::

        ner = GLiNERNER()
        spans = ner.predict("Metformin treats type 2 diabetes.", Language.EN)
        # → [MentionSpan("Metformin", ..., DRUG), MentionSpan("type 2 diabetes", ..., DISEASE)]
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_mediumv2.1",
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

    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            logger.info("gliner_loading", model=self._model_name)
            self._model = _GLiNERModel.from_pretrained(self._model_name)
            logger.info("gliner_loaded", model=self._model_name)
        return self._model
