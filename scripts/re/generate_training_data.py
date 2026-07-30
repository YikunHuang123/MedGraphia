"""
CLI: Generate BERT RE training data from Neo4j.

Usage:
    python scripts/re/generate_training_data.py
    python scripts/re/generate_training_data.py --limit 200 --lang en
    python scripts/re/generate_training_data.py --limit 10000 --none-ratio 1.5
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.ingestion.re.dataset import (
    append_labeled_pairs,
    dataset_stats,
    load_processed_chunk_ids,
)
from medgraphia.ingestion.re.labeler import DataLabeler, fetch_chunks
from medgraphia.logger import configure_logging


async def main(args: argparse.Namespace) -> None:
    cfg = get_settings()
    configure_logging(cfg.log_level)
    random.seed(42)

    output_path = Path(args.output)

    processed_ids = load_processed_chunk_ids(output_path)
    if processed_ids:
        print(f"Resuming — {len(processed_ids)} chunk_ids already written.")

    print(f"\n{'='*60}")
    print("  MedGraphia — RE Training Data Generator")
    print(f"{'='*60}\n")

    print(f"Fetching chunks (limit={args.limit}, lang={args.lang or 'all'})...")
    chunks = await fetch_chunks(args.lang, args.limit)
    chunks = [c for c in chunks if c.chunk_id not in processed_ids]
    print(f"✓ {len(chunks)} new chunks to process.\n")

    if not chunks:
        print("Nothing to do.")
        return

    labeler = DataLabeler(none_ratio=args.none_ratio)
    semaphore = asyncio.Semaphore(args.concurrency)
    stats = {"positives": 0, "none": 0, "llm_empty": 0, "llm_errors": 0}

    async def process(chunk):
        async with semaphore:
            result = await labeler.label_chunk(chunk)
            # None  = exception in LLM call (traceback already logged)
            # {}    = LLM ran successfully but found no relations in this chunk
            if result is None:
                stats["llm_errors"] += 1
                llm_map: dict = {}
            else:
                llm_map = result
                if not llm_map:
                    stats["llm_empty"] += 1
            return labeler.build_labeled_pairs(chunk, llm_map)

    tasks = [process(c) for c in chunks]
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        pairs = await coro
        append_labeled_pairs(pairs, output_path)
        for p in pairs:
            if p.label == "NONE":
                stats["none"] += 1
            else:
                stats["positives"] += 1
        if i % 50 == 0 or i == len(chunks):
            print(
                f"  [{i}/{len(chunks)}] "
                f"positives={stats['positives']}  none={stats['none']}  "
                f"llm_empty={stats['llm_empty']}  llm_errors={stats['llm_errors']}"
            )

    ratio = stats["none"] / max(stats["positives"], 1)
    print(f"\n{'='*60}")
    print(f"  Output          : {output_path}")
    print(f"  Positives       : {stats['positives']}")
    print(f"  NONE            : {stats['none']}  (ratio {ratio:.2f})")
    print(f"  LLM returned empty : {stats['llm_empty']}  (chunk had no detectable relations)")
    print(f"  LLM exceptions     : {stats['llm_errors']}  (check logs above for traceback)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BERT RE training data from Neo4j.")
    parser.add_argument("--output", default="data/re/labeled_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--lang", choices=["en", "zh", "de"], default=None)
    parser.add_argument("--none-ratio", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nStopped. Progress saved — re-run to resume.")
    except Exception as e:
        import traceback
        print(f"\n❌ {e}")
        traceback.print_exc()
        sys.exit(1)
