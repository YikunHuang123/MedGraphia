#!/usr/bin/env python3
"""
Expand data/processed/ with topic-clustered documents from the already-downloaded
local corpora (data/raw/{pubmed,fda_dailymed,huatuo,germed}) so RAGAS testset
generation has more than one drug/language to draw from.

Pure local parsing, no API calls — reuses the existing StructuredParser /
FDADailyMedConnector parsers. Samples per topic cluster instead of taking
everything, so multi-hop synthesizers (which need semantic overlap between
docs) have topically coherent groups instead of scattered unrelated files.

Usage:
  python scripts/evaluation/expand_processed_corpus.py
  python scripts/evaluation/expand_processed_corpus.py --per-cluster 6 --skip-dailymed
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.domain.document import RawDocument
from medgraphia.ingestion.parsers.structured_parser import StructuredParser
from medgraphia.logger import configure_logging, get_logger

logger = get_logger(__name__)

_OUTPUT_DIR = Path("data/processed")

# English keyword clusters for the PubMed abstract dump — matched against title,
# case-insensitive. Picked to give the multi-hop synthesizer topically coherent
# groups instead of 90k scattered, unrelated abstracts.
_PUBMED_CLUSTERS = {
    "diabetes": ["diabet", "insulin", "glycemic", "hba1c"],
    "cardiovascular": ["cardiac", "cardiovascular", "hypertension", "heart failure", "myocardial"],
    "oncology": ["cancer", "tumor", "tumour", "oncolog", "chemotherapy", "carcinoma"],
    "infectious_disease": ["infection", "antibiotic", "antimicrobial", "sepsis", "bacteria"],
}

# Huatuo departments (label field) worth sampling — internal medicine and
# oncology align with the project's drug/disease focus; neurology adds a
# distinct symptom vocabulary.
_HUATUO_LABELS = ["内科", "肿瘤科", "神经科学"]

_GERMED_MIN_CHARS = 200  # filters out one-sentence fragments too thin for QA generation


def _write_doc(doc: RawDocument, overwrite: bool) -> bool:
    out_path = _OUTPUT_DIR / f"{doc.doc_id}.json"
    if out_path.exists() and not overwrite:
        return False
    out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return True


def _expand_pubmed(per_cluster: int, overwrite: bool) -> int:
    src_dir = Path("data/raw/pubmed/clinical_general")
    if not src_dir.exists():
        logger.warning("pubmed_dir_missing", path=str(src_dir))
        return 0

    parser = StructuredParser()
    buckets: dict[str, list[RawDocument]] = defaultdict(list)
    needed = per_cluster * len(_PUBMED_CLUSTERS)

    for doc in parser.load_pubmed_batch(src_dir):
        title_lower = doc.title.lower()
        for cluster, keywords in _PUBMED_CLUSTERS.items():
            if len(buckets[cluster]) >= per_cluster:
                continue
            if any(kw in title_lower for kw in keywords):
                buckets[cluster].append(doc)
                break
        if sum(len(v) for v in buckets.values()) >= needed:
            break

    written = 0
    for cluster, docs in buckets.items():
        for doc in docs:
            if _write_doc(doc, overwrite):
                written += 1
        logger.info("pubmed_cluster_sampled", cluster=cluster, count=len(docs))
    return written


def _expand_huatuo(per_label: int, overwrite: bool) -> int:
    src_path = Path("data/raw/huatuo/huatuo_lite.jsonl")
    if not src_path.exists():
        logger.warning("huatuo_file_missing", path=str(src_path))
        return 0

    # Filter by label before parsing (parser doesn't expose label on RawDocument),
    # then reuse parse_huatuo's mapping logic per matched raw line.
    wanted_ids: dict[str, int] = {}
    counts: dict[str, int] = defaultdict(int)
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            label = data.get("label", "")
            if label in _HUATUO_LABELS and counts[label] < per_label:
                wanted_ids[str(data.get("id"))] = 1
                counts[label] += 1
            if all(counts[label_] >= per_label for label_ in _HUATUO_LABELS):
                break

    parser = StructuredParser()
    written = 0
    for doc in parser.parse_huatuo(src_path):
        raw_id = doc.doc_id.removeprefix("huatuo_")
        if raw_id in wanted_ids:
            if _write_doc(doc, overwrite):
                written += 1
            wanted_ids.pop(raw_id)
        if not wanted_ids:
            break
    logger.info("huatuo_sampled", labels=dict(counts), written=written)
    return written


def _expand_germed(count: int, overwrite: bool) -> int:
    src_path = Path("data/raw/germed/GERNERMED_dataset.json")
    if not src_path.exists():
        logger.warning("germed_file_missing", path=str(src_path))
        return 0

    parser = StructuredParser()
    written = 0
    for doc in parser.parse_germed(src_path):
        if len(doc.full_text) < _GERMED_MIN_CHARS:
            continue
        if _write_doc(doc, overwrite):
            written += 1
        if written >= count:
            break
    logger.info("germed_sampled", written=written)
    return written


def _expand_dailymed(overwrite: bool) -> int:
    """Parse every FDA DailyMed XML not yet in data/processed/ (only 12 total, no sampling needed)."""
    src_dir = Path("data/raw/fda_dailymed")
    if not src_dir.exists():
        logger.warning("dailymed_dir_missing", path=str(src_dir))
        return 0

    from medgraphia.data.fda_dailymed import _parse_spl_xml

    written = 0
    for xml_path in src_dir.glob("*.xml"):
        set_id = xml_path.stem
        try:
            doc = _parse_spl_xml(xml_path.read_bytes(), set_id=set_id, drug_name="")
        except Exception as exc:
            logger.warning("dailymed_parse_failed", file=xml_path.name, error=str(exc))
            continue
        if _write_doc(doc, overwrite):
            written += 1
    logger.info("dailymed_sampled", written=written)
    return written


@click.command()
@click.option("--per-cluster", default=8, show_default=True, help="Docs per PubMed topic cluster")
@click.option("--per-label", default=8, show_default=True, help="Docs per Huatuo department label")
@click.option("--germed-count", default=15, show_default=True, help="Docs to sample from GERNERMED")
@click.option("--skip-pubmed", is_flag=True)
@click.option("--skip-huatuo", is_flag=True)
@click.option("--skip-germed", is_flag=True)
@click.option("--skip-dailymed", is_flag=True)
@click.option("--overwrite", is_flag=True, help="Overwrite files already in data/processed/")
def main(
    per_cluster: int,
    per_label: int,
    germed_count: int,
    skip_pubmed: bool,
    skip_huatuo: bool,
    skip_germed: bool,
    skip_dailymed: bool,
    overwrite: bool,
) -> None:
    configure_logging("INFO")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    if not skip_pubmed:
        total += _expand_pubmed(per_cluster, overwrite)
    if not skip_huatuo:
        total += _expand_huatuo(per_label, overwrite)
    if not skip_germed:
        total += _expand_germed(germed_count, overwrite)
    if not skip_dailymed:
        total += _expand_dailymed(overwrite)

    click.echo(f"\nWrote {total} new file(s) to {_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
