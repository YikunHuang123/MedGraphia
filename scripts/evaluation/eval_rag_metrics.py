"""
RAGAS Evaluation Script for MedGraphia.

This script runs the full GraphRAG pipeline on a test set and evaluates the results
using RAGAS metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall).

It can be used to compare different strategies, such as the time-decay memory factor.
"""

from __future__ import annotations

import sys
from pathlib import Path
import types
import os

# --- Load .env first and FORCE OVERRIDE environment variables ---
from dotenv import load_dotenv
# We find the .env file in the root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Clear base_url if it's set in shell but we want to use official OpenAI for RAGAS
if "OPENAI_BASE_URL" in os.environ:
    # If it points to siliconflow but we have an sk-proj- key, it will fail.
    # We clear it to default to official OpenAI for the evaluation judge.
    if "siliconflow" in os.environ["OPENAI_BASE_URL"].lower():
        del os.environ["OPENAI_BASE_URL"]

# --- MONKEYPATCH for RAGAS 0.4.3 compatibility ---
try:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
except ImportError:
    vertex_mock = types.ModuleType("vertexai")
    vertex_mock.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = vertex_mock

try:
    # Some environments use langchain_aws, others might need a mock
    import langchain_aws
    sys.modules["langchain_community.chat_models.bedrock"] = langchain_aws
except ImportError:
    bedrock_mock = types.ModuleType("bedrock")
    bedrock_mock.BedrockChat = None
    sys.modules["langchain_community.chat_models.bedrock"] = bedrock_mock

import asyncio
import json
import os
from typing import Any

import click
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
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
    samples: list[dict[str, Any]],
    user_id: str | None = "eval_user",
    language: Language = Language.EN,
) -> pd.DataFrame:
    """
    Run the RAG pipeline on samples and collect data for RAGAS.
    """
    settings = get_settings()
    retrieval = RetrievalPipeline.from_settings()
    generation = GenerationPipeline.from_settings()

    results = []

    logger.info("eval_starting", count=len(samples), user_id=user_id)

    for i, sample in enumerate(samples):
        history = []
        
        # Handle multi-turn if present (only for hardcoded samples usually)
        question = sample["question"]
        if "follow_up" in sample and sample["follow_up"]:
            # 1. First turn to establish context/memory
            logger.info("eval_turn_1", q=question)
            try:
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
            except Exception as e:
                logger.error("eval_multi_turn_setup_failed", error=str(e))
                continue

        logger.info(f"eval_processing_{i+1}/{len(samples)}", q=str(question)[:50])

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
            
            # Handle list-formatted ground truth from CSV if needed
            gt = sample["ground_truth"]
            if isinstance(gt, list):
                gt = " ".join(gt)
            
            results.append({
                "question": question,
                "contexts": contexts,
                "answer": gen_result.answer,
                "ground_truth": gt,
                "category": sample.get("category", "general")
            })

        except Exception as exc:
            logger.error("eval_sample_failed", q=question, error=str(exc))

    return pd.DataFrame(results)


def run_ragas_scoring(df: pd.DataFrame) -> Any:
    """
    Use the RAGAS library to score the collected results.
    """
    if df.empty:
        return None

    logger.info("ragas_scoring_started", rows=len(df))
    
    # RAGAS 0.4.3 might need certain columns in the dataset
    dataset = Dataset.from_pandas(df)
    
    # Explicitly provide LLM and Embeddings to avoid 401/env issues
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.run_config import RunConfig
    
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_embeddings = OpenAIEmbeddings()
    
    # Increase timeout and reduce parallelism to avoid rate limits/timeouts
    # Sequential execution (max_workers=1) is safest for debugging/small tests
    run_config = RunConfig(timeout=600, max_retries=10, max_wait=180, max_workers=1)
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config
    )

    return result


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

@click.command()
@click.option("--input", default=None, help="Input CSV file with test samples")
@click.option("--limit", default=None, type=int, help="Limit number of test samples")
@click.option("--output", default="eval_results.csv", help="Output CSV path")
@click.option("--user-id", default="eval_user", help="User ID for memory/decay testing")
def main(input: str | None, limit: int | None, output: str, user_id: str) -> None:
    configure_logging("INFO")
    
    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY_NOT_FOUND", msg="RAGAS evaluation requires an OpenAI key.")

    click.echo("\n" + "=" * 60)
    click.echo("  MedGraphia — RAGAS Quality Evaluation")
    click.echo("=" * 60 + "\n")

    # Load samples
    if input:
        click.echo(f"Loading samples from {input}...")
        df_input = pd.read_csv(input)
        # Ensure minimum columns exist
        if "question" not in df_input.columns or "ground_truth" not in df_input.columns:
            click.echo("Error: Input CSV must contain 'question' and 'ground_truth' columns.")
            return
        samples = df_input.to_dict("records")
    else:
        click.echo("Using hardcoded EVAL_SAMPLES.")
        samples = EVAL_SAMPLES

    if limit:
        samples = samples[:limit]

    # 1. Run Pipeline
    df = asyncio.run(run_evaluation(samples, user_id=user_id))
    
    if df.empty:
        click.echo("Error: No data collected from the pipeline.")
        return

    # 2. Run RAGAS Scoring
    result = run_ragas_scoring(df)

    # 3. Report
    click.echo("\n" + "-" * 20 + " RAGAS SCORES " + "-" * 20)
    if result:
        click.echo(result)
    click.echo("-" * 54 + "\n")

    # 4. Save detailed results
    df.to_csv(output, index=False)
    click.echo(f"Detailed results saved to {output}")


if __name__ == "__main__":
    main()
