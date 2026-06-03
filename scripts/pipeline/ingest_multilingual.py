"""
Specialized ingestion script for large-scale multilingual datasets.
Processes documents in batches to ensure memory efficiency.

python scripts/pipeline/ingest_multilingual.py --limit 1000 --skip-en --skip-de --batch-size 20
python scripts/pipeline/ingest_multilingual.py --limit 1000 --skip-en --skip-zh --batch-size 30
python scripts/pipeline/ingest_multilingual.py --limit 1000 --skip-zh --skip-de --batch-size 20
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from itertools import islice

import click

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.logger import configure_logging, get_logger
from medgraphia.ingestion.parsers.structured_parser import StructuredParser
from medgraphia.ingestion.pipeline import (
    chunk_task, ner_task, link_task, extract_task, embed_task
)

from tqdm import tqdm
import logging

logger = get_logger("ingest_multilingual")

def batched(iterable, n):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

async def process_batch(batch, batch_idx, total_batches):
    """Runs the core pipeline stages for a single batch of documents."""
    # We use logger.debug to avoid cluttering the terminal when tqdm is active
    logger.debug("process_batch_start", index=batch_idx, total=total_batches, size=len(batch))
    
    # 1. Chunking (also writes to Neo4j)
    chunks = await chunk_task(batch)
    if not chunks:
        return

    # 2. NER
    chunks = await ner_task(chunks)

    # 3. Entity Linking (also writes to Neo4j)
    chunks = await link_task(chunks)

    # 5. Embedding (writes to Qdrant)
    await embed_task(chunks)

@click.command()
@click.option("--batch-size", default=20, help="Number of documents per batch (default 20 for M1)")
@click.option("--limit", default=None, type=int, help="Limit total documents for testing")
@click.option("--skip-en", is_flag=True, help="Skip English PubMed data")
@click.option("--skip-zh", is_flag=True, help="Skip Chinese Huatuo data")
@click.option("--skip-de", is_flag=True, help="Skip German GERNERMED data")
@click.option("--verbose", is_flag=True, help="Show all logs instead of just progress bar")
def main(batch_size, limit, skip_en, skip_zh, skip_de, verbose):
    cfg = get_settings()
    # If not verbose, we set log level to WARNING to hide info logs and show tqdm clearly
    log_level = cfg.log_level if verbose else "WARNING"
    configure_logging(log_level)
    
    # Silence some very noisy libraries
    if not verbose:
        for noisy_logger in ["prefect", "httpx", "neo4j", "sentence_transformers", "FlagEmbedding"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    parser = StructuredParser()
    all_docs = []

    if not skip_de:
        de_path = Path("data/raw/germed/GERNERMED_dataset.json")
        if de_path.exists():
            all_docs.append(parser.parse_germed(de_path))
            
    if not skip_zh:
        zh_path = Path("data/raw/huatuo/huatuo_lite.jsonl")
        if zh_path.exists():
            all_docs.append(parser.parse_huatuo(zh_path))

    if not skip_en:
        en_dir = Path("data/raw/pubmed/clinical_general")
        if en_dir.exists():
            all_docs.append(parser.load_pubmed_batch(en_dir))

    def doc_generator():
        count = 0
        for gen in all_docs:
            for doc in gen:
                yield doc
                count += 1
                if limit and count >= limit:
                    return

    # Collect documents to know total count for tqdm
    click.echo("🔍 Collecting documents...")
    docs_list = list(doc_generator())
    total_docs = len(docs_list)
    batches = list(batched(docs_list, batch_size))
    total_batches = len(batches)
    
    click.echo(f"🚀 Starting ingestion: {total_docs} documents | {total_batches} batches")

    async def run_all():
        with tqdm(total=total_batches, desc="Total Progress", unit="batch") as pbar:
            for i, batch in enumerate(batches):
                try:
                    await process_batch(batch, i + 1, total_batches)
                    pbar.update(1)
                except Exception as exc:
                    logger.error("batch_failed", index=i+1, error=str(exc))
                    if click.confirm(f"\nBatch {i+1} failed. Continue?"):
                        continue
                    else:
                        break

    asyncio.run(run_all())
    click.echo("\n✓ Multilingual ingestion complete.")

if __name__ == "__main__":
    main()
