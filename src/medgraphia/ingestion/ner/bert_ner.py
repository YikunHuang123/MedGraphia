"""
Domain-specific BERT NER — optional precision layer after GLiNER coarse pass.

Each language uses a fine-tuned BERT model for higher-precision entity detection.
Results are merged with GLiNER spans in the pipeline (pipeline.py deduplicates).

Language → default model:
  EN: d4data/biomedical-ner-all  (multi-corpus biomedical NER)
  ZH: uer/roberta-base-finetuned-cluener2020-chinese
  DE: skipped by default (ner_bert_de_model = "" in config)

Degradation: if `transformers` is unavailable or model fails to load,
predict() returns [] and the pipeline falls back to GLiNER-only results.
"""
from __future__ import annotations

from medgraphia.domain import EntityType, Language
from medgraphia.ingestion.ner._types import MentionSpan
from medgraphia.logger import get_logger

logger = get_logger(__name__)

try:
    from transformers import pipeline as _hf_pipeline  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers_not_installed",
        msg="pip install transformers torch to enable BERT NER",
    )

# ---------------------------------------------------------------------------
# BIO label → EntityType mapping
# Covers label conventions used by common biomedical NER models.
# ---------------------------------------------------------------------------

_BIO_PREFIX = {"B-", "I-", "b-", "i-"}

_LABEL_TO_TYPE: dict[str, EntityType] = {
    # Disease / condition
    "disease": EntityType.DISEASE,
    "dis": EntityType.DISEASE,
    "diso": EntityType.DISEASE,
    "condition": EntityType.DISEASE,
    "disorder": EntityType.DISEASE,
    "med_dis": EntityType.DISEASE,
    "疾病": EntityType.DISEASE,
    "病症": EntityType.DISEASE,
    "disease_disorder": EntityType.DISEASE,  # From debug logs
    # Drug / chemical
    "drug": EntityType.DRUG,
    "chemical": EntityType.DRUG,
    "chem": EntityType.DRUG,
    "medication": EntityType.DRUG,
    "med": EntityType.DRUG,
    "pharma": EntityType.DRUG,
    "substance": EntityType.DRUG,
    "medicine": EntityType.DRUG,
    "药物": EntityType.DRUG,
    "药品": EntityType.DRUG,
    "dru": EntityType.DRUG,
    "m": EntityType.DRUG, # iioSnail often outputs 'M' for all medical entities
    # Symptom / sign
    "symptom": EntityType.SYMPTOM,
    "sign": EntityType.SYMPTOM,
    "finding": EntityType.SYMPTOM,
    "phenotype": EntityType.SYMPTOM,
    "症状": EntityType.SYMPTOM,
    "体征": EntityType.SYMPTOM,
    "sym": EntityType.SYMPTOM,
    "sign_symptom": EntityType.SYMPTOM,  # From debug logs
    # Gene / protein
    "gene": EntityType.GENE,
    "protein": EntityType.GENE,
    "dna": EntityType.GENE,
    "rna": EntityType.GENE,
    "cell_type": EntityType.GENE,
    "基因": EntityType.GENE,
    "蛋白质": EntityType.GENE,
    "biological_structure": EntityType.GENE, # From debug logs, approximate mapping
    # Procedure
    "procedure": EntityType.PROCEDURE,
    "therapy": EntityType.PROCEDURE,
    "treatment": EntityType.PROCEDURE,
    "test": EntityType.PROCEDURE,
    "手术": EntityType.PROCEDURE,
    "治疗": EntityType.PROCEDURE,
    "检查": EntityType.PROCEDURE,
    "pro": EntityType.PROCEDURE,
    "ite": EntityType.PROCEDURE,  # item/test
    # Generic mappings for Chinese fallback (shibing624 models)
    "org": EntityType.DRUG,    # Medical companies or drugs often misclassified as ORG
    "misc": EntityType.DISEASE, # Symptoms/Diseases often misclassified as MISC
}


def _resolve_label(raw: str) -> EntityType | None:
    """Strip BIO prefix and map to EntityType.  Returns None for O / unknown."""
    raw = raw.strip()
    if raw in ("O", "o"):
        return None
    for pfx in _BIO_PREFIX:
        if raw.startswith(pfx):
            raw = raw[len(pfx):]
            break
    return _LABEL_TO_TYPE.get(raw.lower())


# ---------------------------------------------------------------------------
# BertNER
# ---------------------------------------------------------------------------

class BertNER:
    """
    BERT-based NER for precision refinement.  One model per language, loaded lazily.

    Usage::

        ner = BertNER(en_model="d4data/biomedical-ner-all")
        spans = ner.predict("Insulin is used in type 1 diabetes.", Language.EN)
    """

    def __init__(
        self,
        en_model: str = "d4data/biomedical-ner-all",
        zh_model: str = "uer/roberta-base-finetuned-cluener2020-chinese",
        de_model: str = "",
        device: int = -1,   # -1 = CPU; set to 0 for first GPU
    ) -> None:
        self._model_names: dict[Language, str] = {
            Language.EN: en_model,
            Language.ZH: zh_model,
            Language.DE: de_model,
        }
        self._device = device
        self._pipelines: dict[Language, object] = {}

    @property
    def is_available(self) -> bool:
        return _TRANSFORMERS_AVAILABLE

    def predict(self, text: str, language: Language = Language.EN) -> list[MentionSpan]:
        """Extract spans via the HuggingFace token-classification pipeline."""
        if not _TRANSFORMERS_AVAILABLE or not text.strip():
            return []

        model_name = self._model_names.get(language, "")
        if not model_name:
            return []

        try:
            pipe = self._load_pipeline(language, model_name)
        except Exception as exc:
            logger.warning(
                "bert_ner_load_failed",
                lang=language.value,
                model=model_name,
                error=str(exc),
            )
            return []

        try:
            # aggregation_strategy="simple" merges consecutive B/I tokens
            raw_entities = pipe(text, aggregation_strategy="simple")
        except Exception as exc:
            logger.warning("bert_ner_predict_failed", lang=language.value, error=str(exc))
            return []

        spans: list[MentionSpan] = []
        for ent in raw_entities:
            label_key = ent.get("entity_group") or ent.get("entity", "")
            entity_type = _resolve_label(label_key)
            if entity_type is None:
                # DEBUG INJECTION: Log the rejected label to understand why it failed
                logger.debug("bert_ner_rejected", lang=language.value, label=label_key, word=ent.get("word", ""))
                continue
            
            word = ent.get("word", "")
            # Clean up BERT subword fragments (e.g., "##pirin" -> "pirin")
            word = word.replace("##", "").strip()
            
            # Chinese space-cleanup: some tokenizers insert spaces between characters
            if language == Language.ZH:
                word = word.replace(" ", "")
                
            if not word:
                continue

            start = ent.get("start", 0)
            end = ent.get("end", start + len(word))
            score = float(ent.get("score", 1.0))
            spans.append(
                MentionSpan.from_text(
                    text=word,
                    start=start,
                    end=end,
                    entity_type=entity_type,
                    confidence=score,
                    source="bert",
                )
            )

        logger.debug(
            "bert_ner_done",
            lang=language.value,
            model=model_name,
            spans=len(spans),
        )
        return spans

    # ------------------------------------------------------------------

    def _load_pipeline(self, language: Language, model_name: str):
        if language not in self._pipelines:
            logger.info("bert_ner_loading", lang=language.value, model=model_name)
            self._pipelines[language] = _hf_pipeline(
                "token-classification",
                model=model_name,
                device=self._device,
            )
            logger.info("bert_ner_loaded", lang=language.value, model=model_name)
        return self._pipelines[language]
