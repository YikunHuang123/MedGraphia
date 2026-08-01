"""
Ingest the curated documents in data/processed/ (produced by
scripts/evaluation/expand_processed_corpus.py) into Neo4j + Qdrant.

data/processed/*.json is already RawDocument-shaped JSON, so this reuses the
same chunk/ner/link/embed task functions as build_graph.py's flow — it just
skips the fetch/load/parse stages since the documents are already parsed.

python scripts/pipeline/ingest_processed_corpus.py
python scripts/pipeline/ingest_processed_corpus.py --limit 5
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

sys.modules["prefect"] = None

from medgraphia.domain import RawDocument
from medgraphia.graph.schema import apply_schema
from medgraphia.ingestion.pipeline import chunk_task, embed_task, link_task, ner_task
from medgraphia.logger import configure_logging, get_logger

logger = get_logger("ingest_processed_corpus")


def load_docs(data_dir: str, limit: int | None) -> list[RawDocument]:
    files = sorted(Path(data_dir).glob("*.json"))
    if limit:
        files = files[:limit]
    docs: list[RawDocument] = []
    for f in files:
        try:
            docs.append(RawDocument.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("failed_to_load_doc", file=f.name, error=str(exc))
    return docs


@click.command()
@click.option("--data-dir", default="data/processed", show_default=True)
@click.option("--limit", default=None, type=int, help="Limit number of documents (for a quick test)")
def main(data_dir: str, limit: int | None) -> None:
    configure_logging("INFO")

    async def run() -> None:
        await apply_schema()

        docs = load_docs(data_dir, limit)
        click.echo(f"Loaded {len(docs)} document(s) from {data_dir}")
        if not docs:
            return

        chunks = await chunk_task(docs)
        click.echo(f"Chunked into {len(chunks)} chunk(s), written to Neo4j")

        chunks = await ner_task(chunks)
        n_entities = sum(len(c.entities) for c in chunks)
        click.echo(f"NER done: {n_entities} entity mention(s)")

        chunks = await link_task(chunks)
        click.echo("Entity linking done, written to Neo4j")

        await embed_task(chunks)
        click.echo("Embedding done, written to Qdrant")

        click.echo("\n✓ Ingestion complete.")

    asyncio.run(run())


if __name__ == "__main__":
    main()
