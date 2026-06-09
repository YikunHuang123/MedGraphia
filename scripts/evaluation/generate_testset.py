#!/usr/bin/env python3
"""
Synthetic Test Set Generation Script (Ragas 0.4.3 Compatible).

Uses RAGAS TestsetGenerator to create medical QA pairs from local processed data.
Requires an OpenAI API Key for generation.

python scripts/evaluation/generate_testset.py --test-size 10 --docs-limit 3
"""

from __future__ import annotations

import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── Monkeypatch: must execute before any ragas / langchain imports ───────────
# RAGAS 0.4.3 still references langchain_community sub-modules that have been
# split into separate packages in newer LangChain releases.
try:
    import langchain_google_vertexai

    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
except ImportError:
    _vertex_mock = types.ModuleType("vertexai")
    _vertex_mock.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _vertex_mock

try:
    import langchain_aws

    sys.modules["langchain_community.chat_models.bedrock"] = langchain_aws
except ImportError:
    _bedrock_mock = types.ModuleType("bedrock")
    _bedrock_mock.BedrockChat = None
    sys.modules["langchain_community.chat_models.bedrock"] = _bedrock_mock

try:
    from mistralai import Mistral  # noqa: F401
except ImportError:
    try:
        from mistralai.client import Mistral as _Mistral

        _mistral_mod = types.ModuleType("mistralai")
        _mistral_mod.Mistral = _Mistral
        sys.modules["mistralai"] = _mistral_mod
    except ImportError:
        pass
# ────────────────────────────────────────────────────────────────────────────

import click
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.transforms.engine import Parallel
from ragas.testset.transforms.extractors import EmbeddingExtractor, SummaryExtractor
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.logger import configure_logging, get_logger

logger = get_logger(__name__)

# Sections that confuse the LLM or carry no diagnostic value for medical QA
_SKIP_SECTION_KEYWORDS = frozenset(
    [
        "HOW SUPPLIED",
        "DESCRIPTION",
        "PACKAGE LABEL",
        "STORAGE AND HANDLING",
    ]
)

# Medical personas injected into the RAGAS generator
_MEDICAL_PERSONAS = [
    Persona(
        name="Clinical Pharmacologist",
        role_description="An expert in drug mechanisms, pharmacokinetics, and interactions.",
    ),
    Persona(
        name="Medical Researcher",
        role_description="A scientist evaluating clinical trial data and evidence-based guidelines.",
    ),
]


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def _truncate_at_sentence(text: str, max_chars: int = 3000, min_chars: int = 2000) -> str:
    """Truncate *text* near *max_chars* without splitting mid-sentence."""
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last_break = max(chunk.rfind(". "), chunk.rfind("\n"))
    if last_break > min_chars:
        return chunk[: last_break + 1]
    return chunk


def _load_single_file(json_file: Path) -> list[Document]:
    """Load and parse one processed JSON file into LangChain Documents."""
    docs: list[Document] = []
    try:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        source_id = data.get("source", {}).get("source_id", json_file.stem)
        doc_title = data.get("title", "")

        if data.get("sections"):
            for section in data["sections"]:
                section_title = section.get("title", "").upper()
                if any(k in section_title for k in _SKIP_SECTION_KEYWORDS):
                    continue
                content = section.get("content", "").strip()
                if len(content) > 100:
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": source_id,
                                "title": doc_title,
                                "section": section.get("title", ""),
                            },
                        )
                    )
        else:
            full_text = data.get("full_text", "").strip()
            if full_text:
                docs.append(
                    Document(
                        page_content=_truncate_at_sentence(full_text),
                        metadata={"source": source_id, "title": doc_title},
                    )
                )
    except Exception as exc:
        logger.warning("load_file_failed", file=json_file.name, error=str(exc))
    return docs


def load_documents(data_dir: str, limit: int = 5) -> list[Document]:
    """Load processed JSON files into LangChain Documents in parallel."""
    json_files = sorted(Path(data_dir).glob("*.json"))[:limit]
    logger.info("loading_files", count=len(json_files), limit=limit)

    if not json_files:
        return []

    with ThreadPoolExecutor(max_workers=min(len(json_files), 8)) as pool:
        batches = list(pool.map(_load_single_file, json_files))

    docs = [doc for batch in batches for doc in batch]
    logger.info("documents_loaded", count=len(docs))
    return docs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--data-dir",
    default="data/processed",
    show_default=True,
    help="Directory containing processed JSON files",
)
@click.option("--test-size", default=10, show_default=True, help="Number of QA pairs to generate")
@click.option(
    "--output",
    default="data/evaluation/synthetic_testset.csv",
    show_default=True,
    help="Output CSV path",
)
@click.option(
    "--docs-limit", default=2, show_default=True, help="Number of source documents to load"
)
@click.option(
    "--append",
    is_flag=True,
    help="Append new rows to an existing output file instead of overwriting",
)
@click.option(
    "--no-dedup",
    is_flag=True,
    help="Skip deduplication when appending (default: deduplicate by question text)",
)
@click.option(
    "--max-workers",
    default=4,
    show_default=True,
    help="RAGAS parallel workers — lower to avoid OpenAI rate limits",
)
def main(
    data_dir: str,
    test_size: int,
    output: str,
    docs_limit: int,
    append: bool,
    no_dedup: bool,
    max_workers: int,
) -> None:
    configure_logging("INFO")

    if not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException("OPENAI_API_KEY environment variable is required.")

    # Validate output path before spending time on generation
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.touch(exist_ok=True)
    except OSError as exc:
        raise click.ClickException(f"Cannot write to output path '{output}': {exc}")

    click.echo("\n" + "=" * 60)
    click.echo("  MedGraphia — Synthetic Testset Generation")
    click.echo("=" * 60 + "\n")

    # 1. Load documents
    documents = load_documents(data_dir, limit=docs_limit)
    if not documents:
        raise click.ClickException(
            f"No documents found in '{data_dir}'. "
            "Check that --data-dir points to a folder with processed JSON files."
        )
    click.echo(f"Loaded {len(documents)} section(s) from {docs_limit} document file(s).")

    # 2. Initialize LLM and embeddings
    click.echo(f"Initializing generator (target: {test_size} QA pairs, workers: {max_workers})...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=10, timeout=120)
    embeddings = OpenAIEmbeddings()

    # 3. Build generator and inject medical personas
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    assert hasattr(generator, "persona_list"), (
        "RAGAS API changed: 'persona_list' attribute not found — "
        "update persona injection for your installed RAGAS version."
    )
    generator.persona_list = _MEDICAL_PERSONAS

    # 4. Define transform pipeline
    #    Stage 1 runs SummaryExtractor and CustomNodeFilter in parallel (was serial).
    ragas_llm = generator.llm
    ragas_emb = generator.embedding_model
    stable_transforms = [
        Parallel(
            SummaryExtractor(llm=ragas_llm),
            CustomNodeFilter(llm=ragas_llm),
        ),
        Parallel(
            EmbeddingExtractor(
                embedding_model=ragas_emb,
                property_name="summary_embedding",
                embed_property_name="summary",
            ),
            ThemesExtractor(llm=ragas_llm),
            NERExtractor(llm=ragas_llm),
        ),
        Parallel(
            CosineSimilarityBuilder(
                property_name="summary_embedding",
                new_property_name="summary_similarity",
                threshold=0.7,
            ),
            OverlapScoreBuilder(threshold=0.01),
        ),
    ]

    # 5. Generate
    run_config = RunConfig(
        timeout=120,
        max_retries=5,
        max_wait=90,
        max_workers=max_workers,
    )
    click.echo("Generating questions — this may take a few minutes...")
    try:
        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=test_size,
            transforms=stable_transforms,
            run_config=run_config,
        )
    except Exception as exc:
        logger.error("generation_failed", error=str(exc))
        raise click.ClickException(f"Generation failed: {exc}")

    # 6. Convert to DataFrame and normalize column names for downstream compatibility
    #    RAGAS outputs: user_input / reference / reference_contexts
    #    Downstream scripts expect: question / ground_truth / contexts
    df = testset.to_pandas()
    column_map = {
        "user_input": "question",
        "reference": "ground_truth",
        "reference_contexts": "contexts",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    if len(df) < test_size:
        logger.warning(
            "testset_undergenerated",
            requested=test_size,
            actual=len(df),
            hint="Increase --docs-limit to provide more source material.",
        )
        click.echo(
            f"\nWARNING: Only {len(df)} of {test_size} requested questions were generated. "
            "Consider increasing --docs-limit."
        )

    # 7. Append + deduplicate
    final_df = df
    if append and output_path.exists() and output_path.stat().st_size > 0:
        click.echo(f"Appending to existing file: {output}")
        existing_df = pd.read_csv(output_path)
        missing_cols = set(existing_df.columns) - set(df.columns)
        if missing_cols:
            logger.warning("append_schema_mismatch", missing_columns=sorted(missing_cols))
            click.echo(
                f"WARNING: Schema mismatch — existing file has extra columns: {sorted(missing_cols)}"
            )
        final_df = pd.concat([existing_df, df], ignore_index=True)
        if not no_dedup and "question" in final_df.columns:
            before = len(final_df)
            final_df = final_df.drop_duplicates(subset=["question"], keep="last")
            removed = before - len(final_df)
            if removed:
                logger.info("dedup_removed", count=removed)
                click.echo(f"Removed {removed} duplicate question(s).")

    # 8. Save
    final_df.to_csv(output, index=False)

    click.echo("\n" + "=" * 60)
    click.echo(f"  SUCCESS: {len(final_df)} total QA pairs ({len(df)} new).")
    click.echo(f"  Saved to: {output}")
    click.echo("=" * 60)

    if "question" in df.columns and not df.empty:
        click.echo("\nSample questions generated:")
        for idx, question in enumerate(df["question"].head(3), start=1):
            click.echo(f"  [{idx}] {question}")

    click.echo()


if __name__ == "__main__":
    main()
