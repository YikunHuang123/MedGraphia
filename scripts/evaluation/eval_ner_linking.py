#!/usr/bin/env python3
"""
Evaluates NER + entity-linking accuracy on a small set covering
English, Chinese, and German medical text.

Metrics reported:
  NER    — Exact-match precision / recall / F1 at the entity-type level
  EL     — Top-1 accuracy (correct MeSH ID in best match)

Usage:
  python scripts/eval_ner_linking.py
  python scripts/eval_ner_linking.py --mesh-dir data/mesh --verbose
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.domain import Chunk, Language, SourceMeta
from medgraphia.logger import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Golden test set (MeSH Compatible)
# ---------------------------------------------------------------------------
# CUI field now contains MeSH Unique IDs (e.g., D008687 for Metformin)

GOLDEN_SET: list[dict[str, Any]] = [
    # ---------- English (Baseline) ----------
    {
        "text": "Metformin is first-line therapy for type 2 diabetes mellitus.",
        "lang": "en",
        "entities": [
            {"text": "Metformin", "type": "Drug", "cui": "D008687"},
            {"text": "type 2 diabetes mellitus", "type": "Disease", "cui": "D003924"},
        ],
    },
    {
        "text": "Warfarin interacts with aspirin and increases bleeding risk.",
        "lang": "en",
        "entities": [
            {"text": "Warfarin", "type": "Drug", "cui": "D014859"},
            {"text": "aspirin", "type": "Drug", "cui": "D001241"},
            {"text": "bleeding", "type": "Symptom", "cui": "D006470"},
        ],
    },
    {
        "text": "Patients with hypertension should be monitored for renal impairment.",
        "lang": "en",
        "entities": [
            {"text": "hypertension", "type": "Disease", "cui": "D006973"},
            {"text": "renal impairment", "type": "Disease", "cui": "D051437"},
        ],
    },
    {
        "text": "Ibuprofen is commonly used to reduce fever and inflammation.",
        "lang": "en",
        "entities": [
            {"text": "Ibuprofen", "type": "Drug", "cui": "D007052"},
            {"text": "fever", "type": "Symptom", "cui": "D005334"},
        ],
    },
    {
        "text": "Patients with asthma often experience shortness of breath.",
        "lang": "en",
        "entities": [
            {"text": "asthma", "type": "Disease", "cui": "D001249"},
            {"text": "shortness of breath", "type": "Symptom", "cui": "D004417"},
        ],
    },
    {
        "text": "The BRCA1 gene mutation increases the risk of breast cancer.",
        "lang": "en",
        "entities": [
            {"text": "BRCA1", "type": "Gene", "cui": "D019398"},
            {"text": "breast cancer", "type": "Disease", "cui": "D001943"},
        ],
    },
    {
        "text": "A colonoscopy was performed to screen for colorectal cancer.",
        "lang": "en",
        "entities": [
            {"text": "colonoscopy", "type": "Procedure", "cui": "D003113"},
            {"text": "colorectal cancer", "type": "Disease", "cui": "D015179"},
        ],
    },
    {
        "text": "Insulin therapy is essential for patients with type 1 diabetes.",
        "lang": "en",
        "entities": [
            {"text": "Insulin", "type": "Drug", "cui": "D007328"},
            {"text": "type 1 diabetes", "type": "Disease", "cui": "D003922"},
        ],
    },

    # ---------- English (Hard / Real-World Clinical) ----------
    {
        "text": "The patient presented with T1DM, complicated by ESRD secondary to diabetic nephropathy, requiring HD.",
        "lang": "en",
        "entities": [
            {"text": "T1DM", "type": "Disease", "cui": "D003922"},
            {"text": "ESRD", "type": "Disease", "cui": "D007676"},
            {"text": "diabetic nephropathy", "type": "Disease", "cui": "D003928"},
            {"text": "HD", "type": "Procedure", "cui": "D006439"},
        ],
    },
    {
        "text": "Initiated amoxicillin/clavulanate 875/125 mg PO BID for 7 days for the treatment of AECOPD.",
        "lang": "en",
        "entities": [
            {"text": "amoxicillin", "type": "Drug", "cui": "D000653"},
            {"text": "clavulanate", "type": "Drug", "cui": "D019886"},
            {"text": "AECOPD", "type": "Disease", "cui": "D029424"},
        ],
    },
    {
        "text": "The patient was diagnosed with non-small cell lung cancer and ST-segment elevation myocardial infarction.",
        "lang": "en",
        "entities": [
            {"text": "non-small cell lung cancer", "type": "Disease", "cui": "D002289"},
            {"text": "ST-segment elevation myocardial infarction", "type": "Disease", "cui": "D000072658"},
        ],
    },
    {
        "text": "Overexpression of HER2/neu and mutant TP53 was observed in the biopsy specimen.",
        "lang": "en",
        "entities": [
            {"text": "HER2/neu", "type": "Gene", "cui": "D018121"},
            {"text": "TP53", "type": "Gene", "cui": "D016159"},
            {"text": "biopsy", "type": "Procedure", "cui": "D001706"},
        ],
    },
    {
        "text": "Due to persistent MRSA bacteremia, vancomycin was switched to daptomycin.",
        "lang": "en",
        "entities": [
            {"text": "MRSA", "type": "Disease", "cui": "D016480"},
            {"text": "bacteremia", "type": "Disease", "cui": "D001438"},
            {"text": "vancomycin", "type": "Drug", "cui": "D014640"},
            {"text": "daptomycin", "type": "Drug", "cui": "D017273"},
        ],
    },
    {
        "text": "A CABG was scheduled due to severe three-vessel CAD.",
        "lang": "en",
        "entities": [
            {"text": "CABG", "type": "Procedure", "cui": "D001026"},
            {"text": "CAD", "type": "Disease", "cui": "D003324"},
        ],
    },
    {
        "text": "Administration of intravenous tPA is indicated for acute ischemic stroke.",
        "lang": "en",
        "entities": [
            {"text": "tPA", "type": "Drug", "cui": "D010959"},
            {"text": "acute ischemic stroke", "type": "Disease", "cui": "D020304"},
        ],
    },

    # ---------- Chinese (Baseline) ----------
    {
        "text": "二甲双胍是治疗2型糖尿病的一线用药。",
        "lang": "zh",
        "entities": [
            {"text": "二甲双胍", "type": "Drug", "cui": "D008687"},
            {"text": "2型糖尿病", "type": "Disease", "cui": "D003924"},
        ],
    },
    {
        "text": "患者出现心肌梗死症状，应立即给予阿司匹林。",
        "lang": "zh",
        "entities": [
            {"text": "心肌梗死", "type": "Disease", "cui": "D009203"},
            {"text": "阿司匹林", "type": "Drug", "cui": "D001241"},
        ],
    },
    {
        "text": "布洛芬常用于缓解发热和疼痛。",
        "lang": "zh",
        "entities": [
            {"text": "布洛芬", "type": "Drug", "cui": "D007052"},
            {"text": "发热", "type": "Symptom", "cui": "D005334"},
        ],
    },
    {
        "text": "哮喘患者常出现呼吸困难症状。",
        "lang": "zh",
        "entities": [
            {"text": "哮喘", "type": "Disease", "cui": "D001249"},
            {"text": "呼吸困难", "type": "Symptom", "cui": "D004417"},
        ],
    },
    {
        "text": "医生建议进行结肠镜检查以筛查结直肠癌。",
        "lang": "zh",
        "entities": [
            {"text": "结肠镜检查", "type": "Procedure", "cui": "D003113"},
            {"text": "结直肠癌", "type": "Disease", "cui": "D015179"},
        ],
    },
    {
        "text": "胰岛素治疗对1型糖尿病患者至关重要。",
        "lang": "zh",
        "entities": [
            {"text": "胰岛素", "type": "Drug", "cui": "D007328"},
            {"text": "1型糖尿病", "type": "Disease", "cui": "D003922"},
        ],
    },
    {
        "text": "患者出现头痛和恶心症状。",
        "lang": "zh",
        "entities": [
            {"text": "头痛", "type": "Symptom", "cui": "D006261"},
            {"text": "恶心", "type": "Symptom", "cui": "D009325"},
        ],
    },

    # ---------- Chinese (Hard / Real-World Clinical) ----------
    {
        "text": "患者既往有 COPD 及阵发性房颤病史，目前正服用达比加群酯预防缺血性脑卒中。",
        "lang": "zh",
        "entities": [
            {"text": "COPD", "type": "Disease", "cui": "D029424"},
            {"text": "阵发性房颤", "type": "Disease", "cui": "D001281"},
            {"text": "达比加群酯", "type": "Drug", "cui": "D000069055"},
            {"text": "缺血性脑卒中", "type": "Disease", "cui": "D020304"},
        ],
    },
    {
        "text": "急诊查体示双下肺可闻及细湿啰音，心电图提示急性广泛前壁心肌梗死伴频发室性早搏。",
        "lang": "zh",
        "entities": [
            {"text": "细湿啰音", "type": "Symptom", "cui": "D012135"},
            {"text": "心电图", "type": "Procedure", "cui": "D004562"},
            {"text": "急性广泛前壁心肌梗死", "type": "Disease", "cui": "D009203"},
            {"text": "室性早搏", "type": "Disease", "cui": "D018879"},
        ],
    },
    {
        "text": "行经皮冠状动脉介入治疗（PCI）植入两枚西罗莫司洗脱支架（DES）。",
        "lang": "zh",
        "entities": [
            {"text": "经皮冠状动脉介入治疗", "type": "Procedure", "cui": "D062645"},
            {"text": "PCI", "type": "Procedure", "cui": "D062645"},
            {"text": "西罗莫司", "type": "Drug", "cui": "D020123"},
            {"text": "洗脱支架", "type": "Procedure", "cui": "D054855"},
        ],
    },
    {
        "text": "由于持续的高尿酸血症，患者开始服用别嘌醇，但随后出现斯-约综合征（SJS）。",
        "lang": "zh",
        "entities": [
            {"text": "高尿酸血症", "type": "Disease", "cui": "D033461"},
            {"text": "别嘌醇", "type": "Drug", "cui": "D000486"},
            {"text": "斯-约综合征", "type": "Disease", "cui": "D013262"},
            {"text": "SJS", "type": "Disease", "cui": "D013262"},
        ],
    },
    {
        "text": "针对HER2阳性晚期乳腺癌，采用曲妥珠单抗联合紫杉醇方案化疗。",
        "lang": "zh",
        "entities": [
            {"text": "HER2阳性", "type": "Gene", "cui": "D018121"},
            {"text": "晚期乳腺癌", "type": "Disease", "cui": "D001943"},
            {"text": "曲妥珠单抗", "type": "Drug", "cui": "D000068878"},
            {"text": "紫杉醇", "type": "Drug", "cui": "D017239"},
            {"text": "化疗", "type": "Procedure", "cui": "D000971"},
        ],
    },

    # ---------- German (Baseline) ----------
    {
        "text": "Metformin ist die Erstlinientherapie bei Typ-2-Diabetes mellitus.",
        "lang": "de",
        "entities": [
            {"text": "Metformin", "type": "Drug", "cui": "D008687"},
            {"text": "Typ-2-Diabetes mellitus", "type": "Disease", "cui": "D003924"},
        ],
    },
    {
        "text": "Ibuprofen wird häufig zur Fiebersenkung eingesetzt.",
        "lang": "de",
        "entities": [
            {"text": "Ibuprofen", "type": "Drug", "cui": "D007052"},
            {"text": "Fieber", "type": "Symptom", "cui": "D005334"},
        ],
    },
    {
        "text": "Patienten mit Asthma leiden häufig unter Atemnot.",
        "lang": "de",
        "entities": [
            {"text": "Asthma", "type": "Disease", "cui": "D001249"},
            {"text": "Atemnot", "type": "Symptom", "cui": "D004417"},
        ],
    },
    {
        "text": "Eine Koloskopie wurde zur Früherkennung von Darmkrebs durchgeführt.",
        "lang": "de",
        "entities": [
            {"text": "Koloskopie", "type": "Procedure", "cui": "D003113"},
            {"text": "Darmkrebs", "type": "Disease", "cui": "D015179"},
        ],
    },
    {
        "text": "Insulintherapie ist bei Typ-1-Diabetes unerlässlich.",
        "lang": "de",
        "entities": [
            {"text": "Insulin", "type": "Drug", "cui": "D007328"},
            {"text": "Typ-1-Diabetes", "type": "Disease", "cui": "D003922"},
        ],
    },
    {
        "text": "Kopfschmerzen und Übelkeit wurden berichtet.",
        "lang": "de",
        "entities": [
            {"text": "Kopfschmerzen", "type": "Symptom", "cui": "D006261"},
            {"text": "Übelkeit", "type": "Symptom", "cui": "D009325"},
        ],
    },

    # ---------- German (Hard / Real-World Clinical) ----------
    {
        "text": "Der Patient stellte sich mit einer dekompensierten Linksherzinsuffizienz und neu aufgetretenem Vorhofflimmern vor.",
        "lang": "de",
        "entities": [
            {"text": "Linksherzinsuffizienz", "type": "Disease", "cui": "D006333"},
            {"text": "Vorhofflimmern", "type": "Disease", "cui": "D001281"},
        ],
    },
    {
        "text": "Verdacht auf tiefe Venenthrombose (TVT) und rezidivierende Lungenembolie (LAE) unter laufender NOAK-Therapie.",
        "lang": "de",
        "entities": [
            {"text": "tiefe Venenthrombose", "type": "Disease", "cui": "D020246"},
            {"text": "TVT", "type": "Disease", "cui": "D020246"},
            {"text": "Lungenembolie", "type": "Disease", "cui": "D011655"},
            {"text": "LAE", "type": "Disease", "cui": "D011655"},
        ],
    },
    {
        "text": "Zustand nach radikaler retropubischer Prostatektomie (RPE) bei lokal fortgeschrittenem Prostatakarzinom.",
        "lang": "de",
        "entities": [
            {"text": "retropubischer Prostatektomie", "type": "Procedure", "cui": "D011468"},
            {"text": "RPE", "type": "Procedure", "cui": "D011468"},
            {"text": "Prostatakarzinom", "type": "Disease", "cui": "D011471"},
        ],
    },
    {
        "text": "Die MRT des Schädels zeigte multiple Entmarkungsherde passend zu einer Multiplen Sklerose (MS).",
        "lang": "de",
        "entities": [
            {"text": "MRT", "type": "Procedure", "cui": "D008279"},
            {"text": "Multiplen Sklerose", "type": "Disease", "cui": "D009103"},
            {"text": "MS", "type": "Disease", "cui": "D009103"},
        ],
    },
    {
        "text": "Aufgrund einer schweren Sepsis erfolgte die kalkulierte Antibiose mit Meropenem und Linezolid auf der Intensivstation.",
        "lang": "de",
        "entities": [
            {"text": "Sepsis", "type": "Disease", "cui": "D018805"},
            {"text": "Meropenem", "type": "Drug", "cui": "D000077209"},
            {"text": "Linezolid", "type": "Drug", "cui": "D000069347"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Data classes for evaluation results
# ---------------------------------------------------------------------------


@dataclass
class NERResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class ELResult:
    total: int = 0
    top1_correct: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def top1_accuracy(self) -> float:
        denom = self.total - self.skipped
        return self.top1_correct / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, lang: str) -> Chunk:
    lang_map = {"en": Language.EN, "zh": Language.ZH, "de": Language.DE}
    return Chunk(
        doc_id="eval:golden",
        source=SourceMeta(source_id="eval:golden", source_title="Golden Set"),
        language=lang_map.get(lang, Language.EN),
        section_path="eval",
        text=text,
    )


def _spans_overlap(pred_text: str, gold_text: str) -> bool:
    """True if one text is a substring of the other, case-insensitive."""
    p = pred_text.lower().replace(" ", "").strip()
    g = gold_text.lower().replace(" ", "").strip()
    return p in g or g in p


def eval_ner(verbose: bool = False, gliner_model: str | None = None) -> NERResult:
    from medgraphia.ingestion.ner import MedicalNERPipeline, build_pipeline_from_settings

    if gliner_model:
        cfg = get_settings()
        pipeline = MedicalNERPipeline(
            gliner_model=gliner_model,
            gliner_threshold=cfg.ner_gliner_threshold,
            bert_en_model=cfg.ner_bert_en_model,
            bert_zh_model=cfg.ner_bert_zh_model,
            bert_de_model=cfg.ner_bert_de_model,
            min_confidence=cfg.ner_confidence_threshold,
        )
    else:
        pipeline = build_pipeline_from_settings()
    result = NERResult()

    for sample in GOLDEN_SET:
        chunk = _make_chunk(sample["text"], sample["lang"])
        try:
            enriched = pipeline.extract(chunk)
        except Exception as exc:
            result.errors.append(f"NER error on '{sample['text'][:40]}': {exc}")
            result.fn += len(sample["entities"])
            continue

        pred_entities = enriched.entities
        gold_entities = sample["entities"]

        matched_gold = set()
        for pred in pred_entities:
            matched = False
            for gi, gold in enumerate(gold_entities):
                if gi in matched_gold:
                    continue
                if pred.entity_type.value == gold["type"] and _spans_overlap(
                    pred.label, gold["text"]
                ):
                    result.tp += 1
                    matched_gold.add(gi)
                    matched = True
                    break
            if not matched:
                result.fp += 1
                if verbose:
                    logger.debug("ner_fp", text=pred.label, type=pred.entity_type.value)

        result.fn += len(gold_entities) - len(matched_gold)
        if verbose:
            for gi, gold in enumerate(gold_entities):
                if gi not in matched_gold:
                    logger.debug("ner_fn", text=gold["text"])
    return result


def eval_el(mesh_dir: str, verbose: bool = False) -> ELResult:
    from medgraphia.config import get_settings
    from medgraphia.ingestion.entity_linker import EntityLinker
    from medgraphia.ingestion.ner import build_pipeline_from_settings

    cfg = get_settings()
    pipeline = build_pipeline_from_settings()
    linker = EntityLinker.from_mesh(
        mesh_dir=mesh_dir,
        sapbert_model=cfg.el_sapbert_model,
        sapbert_threshold=cfg.el_sapbert_threshold,
    )
    linker.build_index()
    result = ELResult()

    for sample in GOLDEN_SET:
        chunk = _make_chunk(sample["text"], sample["lang"])
        try:
            # We need to run the full pipeline to see what links to what
            enriched = pipeline.extract(chunk)
            linked = linker.link_chunk(enriched)
        except Exception as exc:
            result.errors.append(f"EL error on '{sample['text'][:40]}': {exc}")
            result.total += len(sample["entities"])
            continue

        # Match predicted entities to golden entities based on ID
        predicted_cuis = {e.cui for e in linked.entities}

        for gold in sample["entities"]:
            result.total += 1
            gold_cui = gold.get("cui", "")
            if not gold_cui:
                result.skipped += 1
                continue

            if gold_cui in predicted_cuis:
                result.top1_correct += 1
                if verbose:
                    logger.debug("el_correct", mention=gold["text"], cui=gold_cui)
            else:
                if verbose:
                    logger.debug(
                        "el_wrong",
                        mention=gold["text"],
                        expected=gold_cui,
                        got="NOT_IN_PREDICTIONS",
                    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--mesh-dir", default=None, help="Path to MeSH storage dir")
@click.option("--skip-ner", is_flag=True)
@click.option("--skip-el", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--ner-f1-threshold", default=0.72, show_default=True)
@click.option("--el-acc-threshold", default=0.65, show_default=True)
@click.option(
    "--compare-gliner",
    default=None,
    help="GLiNER model name to compare against the configured default (e.g. Ihor/gliner-biomed-large-v1.0)",
)
def main(
    mesh_dir: str | None,
    skip_ner: bool,
    skip_el: bool,
    verbose: bool,
    ner_f1_threshold: float,
    el_acc_threshold: float,
    compare_gliner: str | None,
) -> None:
    cfg = get_settings()
    # Force DEBUG level if verbose is set
    log_level = "DEBUG" if verbose else cfg.log_level
    configure_logging(log_level)
    mesh_path = mesh_dir or cfg.mesh_dir

    passed = True
    click.echo("\n" + "=" * 60)
    click.echo("  MedGraphia — Quality Evaluation (MeSH)")
    click.echo("=" * 60 + "\n")

    if compare_gliner:
        click.echo(f"Comparing GLiNER: {cfg.ner_gliner_model} (current) vs {compare_gliner}\n")
        baseline = eval_ner(verbose=verbose)
        candidate = eval_ner(verbose=verbose, gliner_model=compare_gliner)
        click.echo(f"{'Model':<45}{'Precision':>10}{'Recall':>10}{'F1':>10}")
        click.echo(f"{cfg.ner_gliner_model:<45}{baseline.precision:>10.3f}{baseline.recall:>10.3f}{baseline.f1:>10.3f}")
        click.echo(f"{compare_gliner:<45}{candidate.precision:>10.3f}{candidate.recall:>10.3f}{candidate.f1:>10.3f}")
        click.echo()
        return

    if not skip_ner:
        click.echo("Evaluating NER...")
        ner = eval_ner(verbose=verbose)
        click.echo(f"  F1: {ner.f1:.3f} (Target: {ner_f1_threshold})")
        if ner.f1 < ner_f1_threshold:
            passed = False
        if ner.errors:
            click.echo("  NER Errors:")
            for err in ner.errors:
                click.echo(f"    - {err}")

    if not skip_el:
        click.echo(f"Evaluating EL (MeSH: {mesh_path})...")
        el = eval_el(mesh_dir=mesh_path, verbose=verbose)
        click.echo(f"  Top-1 Accuracy: {el.top1_accuracy:.3f} (Target: {el_acc_threshold})")
        if el.top1_accuracy < el_acc_threshold:
            passed = False
        if el.errors:
            click.echo("  EL Errors:")
            for err in el.errors:
                click.echo(f"    - {err}")

    click.echo()
    if passed:
        click.echo("✓ All quality thresholds met.\n")
        sys.exit(0)
    else:
        click.echo("✗ Quality thresholds NOT met.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
