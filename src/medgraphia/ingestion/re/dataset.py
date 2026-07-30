"""
Dataset I/O and splitting utilities for BERT RE training data.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from medgraphia.ingestion.re._types import LabeledPair


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_labeled_pairs(pairs: list[LabeledPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for lp in pairs:
            f.write(json.dumps(lp.to_dict(), ensure_ascii=False) + "\n")


def load_processed_chunk_ids(path: Path) -> set[str]:
    """Read chunk_ids already written to a JSONL file (for checkpoint/resume)."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.85,
    min_per_class: int = 5,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split records into train/val preserving (lang × label) distribution.

    Groups with fewer than min_per_class samples go entirely to train
    to avoid val sets with insufficient coverage.
    """
    rng = random.Random(seed)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        key = (rec.get("lang", "en"), rec.get("label", "NONE"))
        groups[key].append(rec)

    train_all: list[dict] = []
    val_all: list[dict] = []
    small_groups: list[tuple[str, str]] = []

    for (lang, label), items in sorted(groups.items()):
        rng.shuffle(items)
        if len(items) < min_per_class:
            small_groups.append((lang, label))
            train_all.extend(items)
            continue
        n_train = max(1, int(len(items) * train_ratio))
        n_val = len(items) - n_train
        if n_val < 1:
            n_train -= 1
        train_all.extend(items[:n_train])
        val_all.extend(items[n_train:])

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    return train_all, val_all, small_groups


def dataset_stats(records: list[dict]) -> dict:
    """Return per-language and per-label counts."""
    return {
        "total": len(records),
        "by_lang": dict(Counter(r.get("lang", "?") for r in records)),
        "by_label": dict(Counter(r.get("label", "?") for r in records)),
    }
