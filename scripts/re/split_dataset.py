"""
CLI: Split labeled_pairs.jsonl into train / val sets.

Usage:
    python scripts/re/split_dataset.py
    python scripts/re/split_dataset.py --train-ratio 0.9 --min-per-class 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.ingestion.re.dataset import (
    dataset_stats,
    load_jsonl,
    stratified_split,
    write_jsonl,
)


def print_stats(name: str, stats: dict) -> None:
    print(f"\n  {name}  ({stats['total']} samples)")
    print(f"  {'─'*38}")
    print("  By language:")
    for lang, n in sorted(stats["by_lang"].items()):
        print(f"    {lang:4s}: {n:6d}  ({100*n/stats['total']:.1f}%)")
    print("  By relation:")
    for label, n in sorted(stats["by_label"].items(), key=lambda x: -x[1]):
        print(f"    {label:<22s}: {n:6d}  ({100*n/stats['total']:.1f}%)")


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌  File not found: {input_path}")
        sys.exit(1)

    output_dir = input_path.parent
    print(f"\n{'='*60}")
    print("  MedGraphia — RE Dataset Splitter")
    print(f"{'='*60}\n")

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} samples from {input_path}\n")

    train, val, small = stratified_split(
        records,
        train_ratio=args.train_ratio,
        min_per_class=args.min_per_class,
        seed=args.seed,
    )
    if small:
        print(f"  ⚠  {len(small)} (lang, label) groups had < {args.min_per_class} samples → train only:")
        for g in small:
            print(f"     {g}")

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print_stats("TRAIN", dataset_stats(train))
    print_stats("VAL  ", dataset_stats(val))

    print(f"\n{'='*60}")
    print(f"  train → {train_path}")
    print(f"  val   → {val_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split RE labeled data into train/val.")
    parser.add_argument("--input", default="data/re/labeled_pairs.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
