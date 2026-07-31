"""
Domain-specific BERT NER — optional precision layer after GLiNER coarse pass.

Each language uses a fine-tuned BERT model for higher-precision NER.
Results are merged with GLiNER spans in the pipeline.py.

Language → default model:
  EN: d4data/biomedical-ner-all  (multi-corpus biomedical NER)
  ZH: Adapting/bert-base-chinese-finetuned-NER-biomedical
  DE: BachelorThesis/GerMedBERT_NER_V01_BRONCO_CARDIO

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

_BIO_PREFIX = {"B-", "I-", "b-", "i-", "E-", "S-", "e-", "s-", "B_", "I_", "b_", "i_", "E_", "S_"}

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
    "疾病和诊断": EntityType.DISEASE,  # Adapting/bert-base-chinese-finetuned-NER-biomedical
    "disease_disorder": EntityType.DISEASE,  # From debug logs
    "diag": EntityType.DISEASE,              # German GerMedBERT DIAG (Diagnosis)
    "problem": EntityType.DISEASE,           # German HUMADEX PROBLEM
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
    # Procedure
    "procedure": EntityType.PROCEDURE,
    "therapy": EntityType.PROCEDURE,
    "treatment": EntityType.PROCEDURE,
    "test": EntityType.PROCEDURE,
    "手术": EntityType.PROCEDURE,
    "治疗": EntityType.PROCEDURE,
    "检查": EntityType.PROCEDURE,
    "影像检查": EntityType.PROCEDURE,   # Adapting/bert-base-chinese-finetuned-NER-biomedical
    "实验室检验": EntityType.PROCEDURE,  # Adapting/bert-base-chinese-finetuned-NER-biomedical
    "pro": EntityType.PROCEDURE,
    "ite": EntityType.PROCEDURE,  # item/test
    "treat": EntityType.PROCEDURE,           # German GerMedBERT TREAT (Treatment)
    "diagnostic_procedure": EntityType.PROCEDURE,  # For d4data/biomedical-ner-all
    "therapeutic_procedure": EntityType.PROCEDURE, # For d4data/biomedical-ner-all
    # Anatomy
    "biological_structure": EntityType.ANATOMY,  # For d4data/biomedical-ner-all
    "anatomy": EntityType.ANATOMY,
    "解剖部位": EntityType.ANATOMY,  # Adapting/bert-base-chinese-finetuned-NER-biomedical
    # Physiology / biological process
    "physiology": EntityType.PHYSIOLOGY,
    "biological_process": EntityType.PHYSIOLOGY,
    # Living being / organism (pathogens)
    "organism": EntityType.LIVING_BEING,
    "生物体": EntityType.LIVING_BEING,

    # Generic or Unknown (Do NOT force into specific clinical types)
    "m": EntityType.UNKNOWN,  # iioSnail outputs 'M' for all medical entities. Should not be forced to DRUG.
    "org": EntityType.UNKNOWN,
    "misc": EntityType.UNKNOWN,
}


def _resolve_label(raw: str) -> EntityType | None:
    """Strip BIO prefix and map to EntityType.  Returns None for O / unknown."""
    raw = raw.strip()
    if raw in ("O", "o"):
        return None
    for pfx in _BIO_PREFIX:
        if raw.startswith(pfx):
            raw = raw[len(pfx) :]
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
        zh_model: str = "Adapting/bert-base-chinese-finetuned-NER-biomedical",
        de_model: str = "BachelorThesis/GerMedBERT_NER_V01_BRONCO_CARDIO",
        device: str | int | None = None,  # Auto-detect if None
    ) -> None:
        self._model_names: dict[Language, str] = {
            Language.EN: en_model,
            Language.ZH: zh_model,
            Language.DE: de_model,
        }

        # ── Device Auto-detection ────────────────────────────────────────────
        if device is None:
            try:
                import torch

                if torch.backends.mps.is_available():
                    self._device = "mps"  # mac
                elif torch.cuda.is_available():
                    self._device = 0  # CUDA index
                else:
                    self._device = -1  # CPU
            except ImportError:
                self._device = -1  # CPU fallback when torch is not installed
        else:
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
                logger.debug(
                    "bert_ner_rejected",
                    lang=language.value,
                    label=label_key,
                    word=ent.get("word", ""),
                )
                continue

            start = ent.get("start")
            end = ent.get("end")
            if start is None or end is None or start >= end:
                continue

            # Trust the pipeline's exact character offsets to extract the original surface form
            word = text[start:end]
            if not word.strip():
                continue

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

    def predict_batch(
        self, texts: list[str], language: Language = Language.EN
    ) -> list[list[MentionSpan]]:
        """
        Batch predict for multiple texts in a single HF pipeline call.
        Returns one MentionSpan list per input text (same order).
        """
        if not _TRANSFORMERS_AVAILABLE or not texts:
            return [[] for _ in texts]

        model_name = self._model_names.get(language, "")
        if not model_name:
            return [[] for _ in texts]

        try:
            pipe = self._load_pipeline(language, model_name)
        except Exception as exc:
            logger.warning(
                "bert_ner_load_failed", lang=language.value, model=model_name, error=str(exc)
            )
            return [[] for _ in texts]

        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return [[] for _ in texts]

        indices, batch_texts = zip(*non_empty)
        try:
            raw_results = pipe(list(batch_texts), aggregation_strategy="simple")
            # HF pipeline returns list[dict] for a single string; list[list[dict]] for a list
            if raw_results and isinstance(raw_results[0], dict):
                raw_results = [raw_results]
        except Exception as exc:
            logger.warning("bert_ner_predict_failed", lang=language.value, error=str(exc))
            return [[] for _ in texts]

        all_spans: list[list[MentionSpan]] = [[] for _ in texts]
        for orig_idx, raw_entities in zip(indices, raw_results):
            text = texts[orig_idx]
            spans = []
            for ent in raw_entities:
                label_key = ent.get("entity_group") or ent.get("entity", "")
                entity_type = _resolve_label(label_key)
                if entity_type is None:
                    continue
                start = ent.get("start")
                end = ent.get("end")
                if start is None or end is None or start >= end:
                    continue
                word = text[start:end]
                if not word.strip():
                    continue
                spans.append(
                    MentionSpan.from_text(
                        text=word,
                        start=start,
                        end=end,
                        entity_type=entity_type,
                        confidence=float(ent.get("score", 1.0)),
                        source="bert",
                    )
                )
            all_spans[orig_idx] = spans

        logger.debug("bert_ner_batch_done", lang=language.value, model=model_name, texts=len(batch_texts))
        return all_spans

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
