"""
RAGAS Evaluation Script for MedGraphia.

This script runs the full GraphRAG pipeline on a test set and evaluates the results
using RAGAS metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall).

It can be used to compare different strategies, such as the time-decay memory factor.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
)

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.domain import Language, QueryType
from medgraphia.generation.pipeline import GenerationPipeline
from medgraphia.logger import configure_logging, get_logger
from medgraphia.retrieval.pipeline import RetrievalPipeline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Test Dataset (Focusing on multi-turn and medical accuracy)
# ---------------------------------------------------------------------------

EVAL_SAMPLES = [
    {
        "question": "What are the common side effects of Metformin?",
        "ground_truth": "Common side effects include nausea, diarrhea, stomach pain, and a metallic taste in the mouth. Lactic acidosis is a rare but serious complication.",
        "category": "safety"
    },
    {
        "question": "Can I take aspirin with warfarin? Explain the risks.",
        "ground_truth": "Combining aspirin and warfarin is generally not recommended as both are blood thinners. It significantly increases the risk of serious bleeding, including gastrointestinal and intracranial hemorrhage.",
        "category": "interaction"
    },
    {
        "question": "What is the recommended first-line treatment for Type 2 Diabetes?",
        "ground_truth": "Metformin is typically the first-line medication for type 2 diabetes, along with lifestyle changes like diet and exercise.",
        "category": "treatment"
    },
    # Multi-turn/Decay Test Case
    {
        "question": "Tell me about Metformin.",
        "follow_up": "Is it safe for someone with renal impairment?",
        "ground_truth": "Metformin is generally contraindicated or requires dose adjustment in patients with significant renal impairment due to the risk of lactic acidosis.",
        "category": "memory_decay"
    }
]

# ---------------------------------------------------------------------------
# Evaluation Core
# ---------------------------------------------------------------------------

async def run_evaluation(
    user_id: str | None = "eval_user",
    language: Language = Language.EN,
    limit: int | None = None
) -> pd.DataFrame:
    """
    Run the RAG pipeline on samples and collect data for RAGAS.
    """
    settings = get_settings()
    retrieval = RetrievalPipeline.from_settings()
    generation = GenerationPipeline.from_settings()

    samples = EVAL_SAMPLES[:limit] if limit else EVAL_SAMPLES
    results = []

    logger.info("eval_starting", count=len(samples), user_id=user_id)

    for i, sample in enumerate(samples):
        history = []
        
        # Handle multi-turn if present
        question = sample["question"]
        if "follow_up" in sample:
            # 1. First turn to establish context/memory
            logger.info("eval_turn_1", q=question)
            ret_1 = await retrieval.execute(question, user_id=user_id, language=language)
            gen_1 = await generation.generate(question, ret_1.query_type, ret_1.items, language=language)
            
            # Update history for the second turn
            from medgraphia.domain.chat import Message, Role
            history = [
                Message(role=Role.USER, content=question),
                Message(role=Role.ASSISTANT, content=gen_1.answer)
            ]
            # Switch to follow-up for the actual evaluation point
            question = sample["follow_up"]

        logger.info(f"eval_processing_{i+1}/{len(samples)}", q=question[:50])

        # 2. Main turn (the one being scored)
        try:
            # Retrieval
            ret_result = await retrieval.execute(
                query=question,
                history=history,
                user_id=user_id,
                language=language
            )

            # Generation
            gen_result = await generation.generate(
                question=question,
                query_type=ret_result.query_type,
                retrieved_items=ret_result.items,
                history=history,
                language=language
            )

            # Prepare data for RAGAS
            # Contexts must be a list of strings
            contexts = [item.text for item in ret_result.items]
            
            results.append({
                "question": question,
                "contexts": contexts,
                "answer": gen_result.answer,
                "ground_truth": sample["ground_truth"],
                "category": sample.get("category", "general")
            })

        except Exception as exc:
            logger.error("eval_sample_failed", q=question, error=str(exc))

    return pd.DataFrame(results)


def run_ragas_scoring(df: pd.DataFrame) -> dict[str, float]:
    """
    Use the RAGAS library to score the collected results.
    """
    if df.empty:
        return {}

    logger.info("ragas_scoring_started", rows=len(df))
    
    # Convert to RAGAS dataset
    dataset = Dataset.from_pandas(df)

    # Note: RAGAS evaluate uses the environment's OPENAI_API_KEY by default.
    # To use other models (Ollama/Groq), you'd normally pass an llm/embeddings 
    # wrapper from langchain, but here we assume the environment is configured.
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall,
        ],
    )

    return result


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

@click.command()
@click.option("--limit", default=None, type=int, help="Limit number of test samples")
@click.option("--output", default="eval_results.csv", help="Output CSV path")
@click.option("--user-id", default="eval_user", help="User ID for memory/decay testing")
def main(limit: int | None, output: str, user_id: str) -> None:
    configure_logging("INFO")
    
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY_NOT_FOUND", msg="RAGAS evaluation usually requires an OpenAI key for the judge model.")

    click.echo("\n" + "=" * 60)
    click.echo("  MedGraphia — RAGAS Quality Evaluation")
    click.echo("=" * 60 + "\n")

    # 1. Run Pipeline
    df = asyncio.run(run_evaluation(limit=limit, user_id=user_id))
    
    if df.empty:
        click.echo("Error: No data collected from the pipeline.")
        return

    # 2. Run RAGAS Scoring
    scores = run_ragas_scoring(df)

    # 3. Report
    click.echo("\n" + "-" * 20 + " RAGAS SCORES " + "-" * 20)
    for metric, score in scores.items():
        click.echo(f"  {metric:20}: {score:.4f}")
    click.echo("-" * 54 + "\n")

    # 4. Save detailed results
    df.to_csv(output, index=False)
    click.echo(f"Detailed results saved to {output}")


if __name__ == "__main__":
    main()
