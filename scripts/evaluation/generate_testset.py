#!/usr/bin/env python3
"""
Synthetic Test Set Generation Script (Ragas 0.4.3 Compatible).

Uses RAGAS TestsetGenerator to create medical QA pairs from local processed data.
Requires an OpenAI API Key for generation.

python scripts/evaluation/generate_testset.py --test-size 10 --docs-limit 3
"""

from __future__ import annotations

import sys
from pathlib import Path
import types

# --- MONKEYPATCH for RAGAS 0.4.3 compatibility with latest LangChain ---
try:
    import langchain_google_vertexai
    sys.modules["langchain_community.chat_models.vertexai"] = langchain_google_vertexai
except ImportError:
    vertex_mock = types.ModuleType("vertexai")
    vertex_mock.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = vertex_mock

try:
    import langchain_aws
    sys.modules["langchain_community.chat_models.bedrock"] = langchain_aws
except ImportError:
    bedrock_mock = types.ModuleType("bedrock")
    bedrock_mock.BedrockChat = None
    sys.modules["langchain_community.chat_models.bedrock"] = bedrock_mock

try:
    from mistralai import Mistral
except ImportError:
    try:
        from mistralai.client import Mistral
        mistral_mod = types.ModuleType("mistralai")
        mistral_mod.Mistral = Mistral
        sys.modules["mistralai"] = mistral_mod
    except ImportError:
        pass
# ----------------------------------------------------------------------

import asyncio
import json
import os
import click
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.logger import configure_logging, get_logger

logger = get_logger(__name__)

def load_documents(data_dir: str, limit: int = 5) -> list[Document]:
    """Load processed JSON files into LangChain Documents."""
    processed_path = Path(data_dir)
    docs = []
    
    # Sections to skip because they often confuse the LLM or aren't useful for medical QA
    SKIP_KEYWORDS = ["HOW SUPPLIED", "DESCRIPTION", "PACKAGE LABEL", "STORAGE AND HANDLING"]
    
    json_files = list(processed_path.glob("*.json"))
    logger.info("loading_files", count=len(json_files), limit=limit)
    
    for i, json_file in enumerate(json_files[:limit]):
        logger.info(f"processing_file: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if "sections" in data and data["sections"]:
                    for section in data["sections"]:
                        title = section.get("title", "").upper()
                        # Skip physical description sections that cause JSON parsing errors
                        if any(k in title for k in SKIP_KEYWORDS):
                            continue
                            
                        content = section.get("content", "").strip()
                        if len(content) > 100: # Slightly longer minimum length
                            docs.append(Document(
                                page_content=content,
                                metadata={
                                    "source": data["source"]["source_id"],
                                    "title": data.get("title", ""),
                                    "section": section.get("title", "")
                                }
                            ))
                else:
                    # Fallback for simple files
                    full_text = data.get("full_text", "").strip()
                    if full_text:
                        docs.append(Document(
                            page_content=full_text[:3000],
                            metadata={
                                "source": data["source"]["source_id"],
                                "title": data.get("title", "")
                            }
                        ))
        except Exception as exc:
            logger.warning("load_file_failed", file=json_file.name, error=str(exc))
            
    logger.info("documents_loaded", count=len(docs))
    return docs

@click.command()
@click.option("--data-dir", default="data/processed", help="Directory containing processed JSONs")
@click.option("--test-size", default=10, help="Number of questions to generate")
@click.option("--output", default="data/evaluation/synthetic_testset.csv", help="Output CSV path")
@click.option("--docs-limit", default=2, help="Number of documents to sample from")
def main(data_dir: str, test_size: int, output: str, docs_limit: int) -> None:
    configure_logging("INFO")
    
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("Error: OPENAI_API_KEY environment variable is required.")
        return

    # 1. Load data
    documents = load_documents(data_dir, limit=docs_limit)
    if not documents:
        click.echo("No documents found to generate questions from.")
        return

    # 2. Initialize Generator (Ragas 0.4.3 style)
    click.echo(f"Initializing RAGAS TestsetGenerator for {test_size} questions...")
    
    # Use gpt-4o-mini: faster, cheaper, and better at following JSON schema for extraction
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings
    )

    # 3. Generate with RunConfig for robustness
    from ragas.run_config import RunConfig
    run_config = RunConfig(timeout=120, max_retries=3, max_wait=60)

    click.echo("Generating questions (this may take a few minutes)...")
    try:
        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=test_size,
            run_config=run_config
        )
        
        # 4. Save and Report
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = testset.to_pandas()
        df.to_csv(output_path, index=False)
        
        click.echo("\n" + "="*40)
        click.echo(f"SUCCESS: Generated {len(df)} questions.")
        click.echo(f"Saved to: {output_path}")
        click.echo("="*40 + "\n")
        
        click.echo("Sample questions:")
        for i, row in df.head(3).iterrows():
            click.echo(f"[{i+1}] {row['question']}")
            
    except Exception as exc:
        logger.error("generation_failed", error=str(exc))
        click.echo(f"Error during generation: {exc}")

if __name__ == "__main__":
    main()
