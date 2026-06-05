#!/usr/bin/env python3
"""
Phase 3 quality-validation script.

Evaluates NER + entity-linking accuracy on a small annotated golden set covering
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
from medgraphia.domain import Chunk, EntityType, Language, SourceMeta
from medgraphia.logger import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Golden test set (MeSH Compatible)
# ---------------------------------------------------------------------------
# CUI field now contains MeSH Unique IDs (e.g., D008687 for Metformin)

GOLDEN_SET: list[dict[str, Any]] = [
    # ---------- English ----------
    {
        "text": "Metformin is first-line therapy for type 2 diabetes mellitus.",
        "lang": "en",
        "entities": [
            {"text": "Metformin",                  "type": "Drug",    "cui": "D008687"},
            {"text": "type 2 diabetes mellitus",    "type": "Disease", "cui": "D003924"},
        ],
    },
    {
        "text": "Warfarin interacts with aspirin and increases bleeding risk.",
        "lang": "en",
        "entities": [
            {"text": "Warfarin",  "type": "Drug", "cui": "D014859"},
            {"text": "aspirin",   "type": "Drug", "cui": "D001241"},
            {"text": "bleeding",  "type": "Symptom", "cui": "D006470"},
        ],
    },
    {
        "text": "Patients with hypertension should be monitored for renal impairment.",
        "lang": "en",
        "entities": [
            {"text": "hypertension",     "type": "Disease", "cui": "D006973"},
            {"text": "renal impairment", "type": "Disease", "cui": "D051437"},
        ],
    },
    # ---------- Chinese ----------
    {
        "text": "二甲双胍是治疗2型糖尿病的一线用药。",
        "lang": "zh",
        "entities": [
            {"text": "二甲双胍",  "type": "Drug",    "cui": "D008687"},
            {"text": "2型糖尿病", "type": "Disease", "cui": "D003924"},
        ],
    },
    {
        "text": "患者出现心肌梗死症状，应立即给予阿司匹林。",
        "lang": "zh",
        "entities": [
            {"text": "心肌梗死", "type": "Disease", "cui": "D009203"},
            {"text": "阿司匹林", "type": "Drug",    "cui": "D001241"},
        ],
    },
    # ---------- German ----------
    {
        "text": "Metformin ist die Erstlinientherapie bei Typ-2-Diabetes mellitus.",
        "lang": "de",
        "entities": [
            {"text": "Metformin",          "type": "Drug",    "cui": "D008687"},
            {"text": "Typ-2-Diabetes mellitus", "type": "Disease", "cui": "D003924"},
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


def eval_ner(verbose: bool = False) -> NERResult:
    from medgraphia.ingestion.ner import build_pipeline_from_settings
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
                if gi in matched_gold: continue
                if (pred.entity_type.value == gold["type"] and _spans_overlap(pred.label, gold["text"])):
                    result.tp += 1
                    matched_gold.add(gi)
                    matched = True
                    break
            if not matched:
                result.fp += 1
                if verbose: logger.debug("ner_fp", text=pred.label, type=pred.entity_type.value)

        result.fn += len(gold_entities) - len(matched_gold)
        if verbose:
            for gi, gold in enumerate(gold_entities):
                if gi not in matched_gold:
                    logger.debug("ner_fn", text=gold["text"])
    return result


def eval_el(mesh_dir: str, verbose: bool = False) -> ELResult:
    from medgraphia.ingestion.entity_linker import EntityLinker
    from medgraphia.ingestion.ner import build_pipeline_from_settings
    from medgraphia.config import get_settings

    cfg = get_settings()
    pipeline = build_pipeline_from_settings()
    linker = EntityLinker.from_mesh(
        mesh_dir=mesh_dir,
        sapbert_model=cfg.el_sapbert_model,
        sapbert_threshold=cfg.el_sapbert_threshold
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
                if verbose: logger.debug("el_correct", mention=gold["text"], cui=gold_cui)
            else:
                if verbose:
                    logger.debug("el_wrong", mention=gold["text"], expected=gold_cui, got="NOT_IN_PREDICTIONS")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--mesh-dir", default=None, help="Path to MeSH storage dir")
@click.option("--skip-ner", is_flag=True)
@click.option("--skip-el",  is_flag=True)
@click.option("--verbose",  is_flag=True)
@click.option("--ner-f1-threshold", default=0.72, show_default=True)
@click.option("--el-acc-threshold", default=0.65, show_default=True)
def main(mesh_dir: str | None, skip_ner: bool, skip_el: bool, verbose: bool, 
         ner_f1_threshold: float, el_acc_threshold: float) -> None:
    cfg = get_settings()
    # Force DEBUG level if verbose is set
    log_level = "DEBUG" if verbose else cfg.log_level
    configure_logging(log_level)
    mesh_path = mesh_dir or cfg.mesh_dir

    passed = True
    click.echo("\n" + "=" * 60)
    click.echo("  MedGraphia — Phase 3 Quality Evaluation (MeSH)")
    click.echo("=" * 60 + "\n")

    if not skip_ner:
        click.echo("Evaluating NER...")
        ner = eval_ner(verbose=verbose)
        click.echo(f"  F1: {ner.f1:.3f} (Target: {ner_f1_threshold})")
        if ner.f1 < ner_f1_threshold: passed = False
        if ner.errors:
            click.echo("  NER Errors:")
            for err in ner.errors:
                click.echo(f"    - {err}")

    if not skip_el:
        click.echo(f"Evaluating EL (MeSH: {mesh_path})...")
        el = eval_el(mesh_dir=mesh_path, verbose=verbose)
        click.echo(f"  Top-1 Accuracy: {el.top1_accuracy:.3f} (Target: {el_acc_threshold})")
        if el.top1_accuracy < el_acc_threshold: passed = False
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
