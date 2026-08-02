#!/usr/bin/env python3
"""
RAGAS Evaluation Script for MedGraphia.

Runs the full GraphRAG pipeline on a test set and evaluates the results using
RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall.

Usage:
  python scripts/evaluation/eval_rag_metrics.py
  python scripts/evaluation/eval_rag_metrics.py --input-file data/evaluation/synthetic_testset.csv --by-category
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from typing import Any
from uuid import uuid4

# ── Load .env before anything else ──────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", override=True)

# Clear non-official OpenAI proxy so the RAGAS judge always hits api.openai.com.
# Developers often export a proxy URL for the application LLM; we don't want that
# URL applied to the separate evaluation judge LLM.
if "OPENAI_BASE_URL" in os.environ:
    if "api.openai.com" not in os.environ["OPENAI_BASE_URL"].lower():
        os.environ.pop("OPENAI_BASE_URL")

# ── Monkeypatch: must run before any ragas / langchain imports ───────────────
# RAGAS 0.4.3 references langchain_community sub-modules moved to separate packages.
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
        import importlib.machinery

        from mistralai.client import Mistral as _Mistral

        _mistral_mod = types.ModuleType("mistralai")
        _mistral_mod.Mistral = _Mistral
        # A bare ModuleType has no __spec__, which makes importlib.util.find_spec()
        # raise ValueError for anything probing "mistralai" later (e.g. instructor).
        _mistral_mod.__spec__ = importlib.machinery.ModuleSpec("mistralai", loader=None)
        sys.modules["mistralai"] = _mistral_mod
    except ImportError:
        pass
# ────────────────────────────────────────────────────────────────────────────

import click
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.domain import Language
from medgraphia.generation.pipeline import GenerationPipeline
from medgraphia.logger import configure_logging, get_logger
from medgraphia.retrieval.pipeline import RetrievalPipeline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Built-in evaluation samples
# ---------------------------------------------------------------------------

EVAL_SAMPLES: list[dict[str, Any]] = [
    {
        "question": "What are the common side effects of Metformin?",
        "ground_truth": (
            "Common side effects include nausea, diarrhea, stomach pain, and a metallic "
            "taste in the mouth. Lactic acidosis is a rare but serious complication."
        ),
        "category": "safety",
    },
    {
        "question": "Can I take aspirin with warfarin? Explain the risks.",
        "ground_truth": (
            "Combining aspirin and warfarin is generally not recommended as both are blood "
            "thinners. It significantly increases the risk of serious bleeding, including "
            "gastrointestinal and intracranial hemorrhage."
        ),
        "category": "interaction",
    },
    {
        "question": "What is the recommended first-line treatment for Type 2 Diabetes?",
        "ground_truth": (
            "Metformin is typically the first-line medication for type 2 diabetes, along "
            "with lifestyle changes like diet and exercise."
        ),
        "category": "treatment",
    },
    {
        "question": "Tell me about Metformin.",
        "follow_up": "Is it safe for someone with renal impairment?",
        "ground_truth": (
            "Metformin is generally contraindicated or requires dose adjustment in patients "
            "with significant renal impairment due to the risk of lactic acidosis."
        ),
        "category": "memory_decay",
    },
]

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


async def run_evaluation(
    samples: list[dict[str, Any]],
    user_id: str | None = "eval_user",
    language: Language = Language.EN,
) -> pd.DataFrame:
    """Run the RAG pipeline on each sample and collect inputs for RAGAS scoring."""
    retrieval = RetrievalPipeline.from_settings()
    generation = GenerationPipeline.from_settings()
    results: list[dict[str, Any]] = []

    logger.info("eval_starting", count=len(samples), user_id=user_id)

    for sample_idx, sample in enumerate(samples):
        history: list[Any] = []
        question = sample["question"]

        # ── Multi-turn setup ────────────────────────────────────────────────
        # Guard against NaN values that arrive when loading CSV files with an
        # optional follow_up column (pandas reads empty cells as float NaN,
        # which is truthy and would incorrectly trigger this branch).
        follow_up = sample.get("follow_up")
        if isinstance(follow_up, str) and follow_up.strip():
            logger.info("eval_turn_1", q=question, sample=sample_idx + 1)
            try:
                ret_1 = await retrieval.execute(question, user_id=user_id, language=language)
                gen_1 = await generation.generate(
                    question=question,
                    query_type=ret_1.query_type,
                    retrieved_items=ret_1.items,
                    language=language,
                )
                from medgraphia.domain.chat import Message, Role

                session_id = str(uuid4())
                history = [
                    Message(session_id=session_id, role=Role.USER, content=question),
                    Message(session_id=session_id, role=Role.ASSISTANT, content=gen_1.answer),
                ]
                question = follow_up
            except Exception as exc:
                logger.error("eval_multi_turn_setup_failed", error=str(exc), sample=sample_idx + 1)
                continue

        logger.info("eval_processing", sample=f"{sample_idx + 1}/{len(samples)}", q=question[:50])

        # ── Main scored turn ────────────────────────────────────────────────
        try:
            ret_result = await retrieval.execute(
                query=question,
                history=history,
                user_id=user_id,
                language=language,
            )
            gen_result = await generation.generate(
                question=question,
                query_type=ret_result.query_type,
                retrieved_items=ret_result.items,
                history=history,
                language=language,
            )

            # RAGAS requires plain-text contexts for semantic scoring.
            # The [N] citation markers and source hints are UI conventions for the
            # LLM prompt; including them here causes the RAGAS judge to conflate
            # citation number alignment with factual grounding, inflating faithfulness
            # scores and adding noise to context precision/recall.
            contexts = [item.text[:800].strip() for item in ret_result.items]

            gt = sample["ground_truth"]
            if isinstance(gt, list):
                gt = " ".join(gt)

            results.append(
                {
                    "question": question,
                    "answer": gen_result.answer,
                    "contexts": contexts,
                    "ground_truth": gt,
                    "category": sample.get("category", "general"),
                }
            )

        except Exception as exc:
            logger.error("eval_sample_failed", q=question, error=str(exc), sample=sample_idx + 1)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# RAGAS scoring
# ---------------------------------------------------------------------------


# Sensible default judge model per provider — used when --judge-model is left
# at its default while switching --judge-provider.
_DEFAULT_JUDGE_MODEL = {
    "openai": "gpt-5.1",
    "deepseek": "deepseek-v4-pro",
    "gemini": "gemini-3.1-pro-preview",
}


def _build_judge(provider: str, judge_model: str) -> tuple[Any, Any]:
    """
    Build the (llm, embeddings) pair RAGAS needs for a given judge provider.

    DeepSeek has no embeddings API, so its judge falls back to a local BGE-M3
    embedding model (already used elsewhere in the project) — zero extra cost,
    no OpenAI dependency required for a DeepSeek-only run.
    """
    from medgraphia.config import get_settings

    cfg = get_settings()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise click.ClickException("OPENAI_API_KEY environment variable is required for --judge-provider openai.")
        return ChatOpenAI(model=judge_model, temperature=0), OpenAIEmbeddings()

    if provider == "deepseek":
        api_key = cfg.deepseek_api_key.get_secret_value()
        if not api_key:
            raise click.ClickException("DEEPSEEK_API_KEY (config: deepseek_api_key) is required for --judge-provider deepseek.")
        llm = ChatOpenAI(
            model=judge_model,
            temperature=0,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        return llm, embeddings

    if provider == "gemini":
        api_key = cfg.gemini_api_key.get_secret_value()
        if not api_key:
            raise click.ClickException("GEMINI_API_KEY (config: gemini_api_key) is required for --judge-provider gemini.")
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(model=judge_model, temperature=0, google_api_key=api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        return llm, embeddings

    raise click.ClickException(f"Unknown judge provider: {provider}")


def run_ragas_scoring(
    df: pd.DataFrame,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    max_workers: int = 1,
) -> Any:
    """Score the collected results with RAGAS metrics."""
    if df.empty:
        return None

    logger.info(
        "ragas_scoring_started",
        rows=len(df),
        judge_model=judge_model,
        judge_provider=judge_provider,
        max_workers=max_workers,
    )

    # Pass only the columns RAGAS expects; extra columns (e.g. 'category') can
    # trigger schema validation warnings or silent errors in RAGAS 0.4.x.
    ragas_cols = [c for c in ["question", "answer", "contexts", "ground_truth"] if c in df.columns]
    dataset = Dataset.from_pandas(df[ragas_cols])

    eval_llm, eval_embeddings = _build_judge(judge_provider, judge_model)

    # answer_relevancy asks the judge for `strictness` completions in one call
    # (n=strictness). DeepSeek's API only accepts n=1, so force single-sample
    # mode for non-OpenAI judges instead of hitting a 400 on every row.
    if judge_provider != "openai":
        answer_relevancy.strictness = 1

    run_config = RunConfig(timeout=600, max_retries=10, max_wait=180, max_workers=max_workers)

    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--input-file", default=None, help="Input CSV with test samples")
@click.option("--limit", default=None, type=int, help="Limit number of samples evaluated")
@click.option("--output", default="eval_results.csv", show_default=True, help="Output CSV path")
@click.option(
    "--user-id", default="eval_user", show_default=True, help="User ID for memory/decay testing"
)
@click.option(
    "--judge-provider",
    type=click.Choice(["openai", "deepseek", "gemini"]),
    default="openai",
    show_default=True,
    help="LLM provider for the RAGAS judge — openai is the most expensive; deepseek/gemini are cheaper alternatives.",
)
@click.option(
    "--judge-model",
    default=None,
    help="Judge model name (defaults per provider: gpt-4o-mini / deepseek-chat / gemini-2.5-flash)",
)
@click.option("--by-category", is_flag=True, help="Print per-category score breakdown")
@click.option(
    "--judge-max-workers",
    default=None,
    type=int,
    help="Concurrent RAGAS judge calls (default: 1 for openai to respect its stricter rate "
    "limits, 4 for deepseek/gemini).",
)
def main(
    input_file: str | None,
    limit: int | None,
    output: str,
    user_id: str,
    judge_provider: str,
    judge_model: str | None,
    by_category: bool,
    judge_max_workers: int | None,
) -> None:
    configure_logging("INFO")

    judge_model = judge_model or _DEFAULT_JUDGE_MODEL[judge_provider]
    if judge_max_workers is None:
        judge_max_workers = 1 if judge_provider == "openai" else 4

    click.echo("\n" + "=" * 60)
    click.echo("  MedGraphia — RAGAS Quality Evaluation")
    click.echo(f"  Judge: {judge_provider}/{judge_model}")
    click.echo("=" * 60 + "\n")

    # 1. Load samples
    if input_file:
        click.echo(f"Loading samples from {input_file}...")
        df_input = pd.read_csv(input_file)
        missing = {"question", "ground_truth"} - set(df_input.columns)
        if missing:
            raise click.ClickException(
                f"Input CSV is missing required column(s): {sorted(missing)}"
            )
        samples = df_input.to_dict("records")
    else:
        click.echo("Using built-in EVAL_SAMPLES.")
        samples = EVAL_SAMPLES

    if limit:
        samples = samples[:limit]
    click.echo(f"Evaluating {len(samples)} sample(s).\n")

    # 2. Run pipeline
    df = asyncio.run(run_evaluation(samples, user_id=user_id))
    if df.empty:
        raise click.ClickException("No data collected — all pipeline calls failed. Check logs.")

    # 3. Save detailed results (full df including category) before RAGAS scoring
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    click.echo(f"Pipeline results saved to {output_path}\n")

    # 4. RAGAS scoring
    result = run_ragas_scoring(
        df, judge_model=judge_model, judge_provider=judge_provider, max_workers=judge_max_workers
    )

    # 5. Report overall scores
    click.echo("-" * 20 + " RAGAS SCORES " + "-" * 20)
    if result:
        click.echo(result)

    # 6. Per-category breakdown (when requested and category column is present)
    if by_category and "category" in df.columns and result is not None:
        scores_df = result.to_pandas()
        if len(scores_df) == len(df):
            scores_df["category"] = df["category"].values
            click.echo("\nPer-category breakdown:")
            # Only select numeric columns — RAGAS to_pandas() also returns string
            # columns (user_input, response, etc.) that cannot be aggregated with mean().
            metric_cols = [
                c
                for c in scores_df.columns
                if c != "category" and pd.api.types.is_numeric_dtype(scores_df[c])
            ]
            grouped = scores_df.groupby("category")[metric_cols].mean()
            click.echo(grouped.to_string(float_format="{:.3f}".format))

    click.echo("-" * 54 + "\n")


if __name__ == "__main__":
    main()
