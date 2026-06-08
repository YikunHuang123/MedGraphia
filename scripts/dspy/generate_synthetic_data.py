"""
Synthetic data factory for DSPy training.

Connects to Qdrant to fetch real medical chunks, then uses DeepSeek as a
teacher model to produce TWO parallel training datasets in a single generation pass:

  1. synthetic_generator_data.json  — augments ANSWER_DATA in optimize.py
     Fields: context, history, question, answer, disclaimer, tier

  2. synthetic_rewriter_data.json   — augments REWRITE_DATA in optimize.py
     Fields: history, latest_message, expected_rewritten_query, expected_tier

Usage:
    python scripts/dspy/generate_synthetic_data.py [N_EXAMPLES]
    (N_EXAMPLES defaults to 50.)
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import dspy 
from qdrant_client import AsyncQdrantClient 
from qdrant_client.http import models as qmodels

from medgraphia.config import get_settings  
from medgraphia.domain import Language  
from medgraphia.generation.llm_router import ModelTier  
from medgraphia.logger import get_logger  
from medgraphia.prompts import GenerateSyntheticMedicalQA  

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------

_LANG_MAP = {
    Language.ZH: "Chinese",
    Language.EN: "English",
    Language.DE: "German",
}


# ---------------------------------------------------------------------------
# Chunk fetcher
# ---------------------------------------------------------------------------


async def fetch_real_chunks(
    collection: str,
    client: AsyncQdrantClient,
    n_per_lang: int = 60,
    min_text_len: int = 80,
) -> list[dict[str, str]]:
    """
    Randomly sample medical chunks from Qdrant, stratified by language.

    Strategy: scroll up to 300 + 300 points per language, then random.sample
    to get diversity across the collection without loading everything.
    """
    all_chunks: list[dict[str, str]] = []

    for lang_enum in (Language.EN, Language.ZH, Language.DE):
        lang = lang_enum.value
        lang_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="language", match=qmodels.MatchValue(value=lang))]
        )

        pool: list[Any] = []

        # First scroll page
        points, next_offset = await client.scroll(
            collection_name=collection,
            scroll_filter=lang_filter,
            limit=300,
            with_payload=True,
            with_vectors=False,
        )
        pool.extend(points)

        # Second page when the collection is large enough
        if next_offset and len(pool) >= 300:
            more, _ = await client.scroll(
                collection_name=collection,
                scroll_filter=lang_filter,
                limit=300,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            pool.extend(more)

        usable = [
            {
                "text": p.payload.get("text", ""),
                "language": lang,
                "source_title": p.payload.get("source_title", ""),
                "section_path": p.payload.get("section_path", ""),
            }
            for p in pool
            if len(p.payload.get("text", "")) >= min_text_len
        ]

        sampled = random.sample(usable, min(n_per_lang, len(usable)))
        all_chunks.extend(sampled)
        logger.info(
            "sampled_chunks",
            extra={"lang": lang, "usable": len(usable), "sampled": len(sampled)},
        )

    random.shuffle(all_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Context window builder
# ---------------------------------------------------------------------------


def make_context_groups(
    chunks: list[dict[str, str]],
    group_size: int = 3,
    total_groups: int = 50,
) -> list[str]:
    """
    Assemble numbered context strings from groups of 2-3 chunks.

    Chunks are shuffled before grouping so that cross-language context
    windows appear naturally (EN + ZH + DE within one group).
    """
    pool = chunks.copy()
    groups: list[str] = []

    for _ in range(total_groups):
        if len(pool) < group_size:
            pool = chunks.copy()
            random.shuffle(pool)

        selected = pool[:group_size]
        pool = pool[group_size:]

        context = "\n".join(f"[{i + 1}] {c['text'].strip()}" for i, c in enumerate(selected))
        groups.append(context)

    return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERSONAS = [
    "anxious patient speaking in plain colloquial language",
    "emergency department physician using clinical terminology",
    "medical student looking up clinical guidelines",
    "elderly patient asking about long-term medication safety",
    "caregiver managing a family member on a complex multi-drug regimen",
]

_TIER_WEIGHTS = (
    [ModelTier.SMALL.value] * 3 + [ModelTier.MEDIUM.value] * 4 + [ModelTier.LARGE.value] * 3
)


def _sample_tier() -> str:
    return random.choice(_TIER_WEIGHTS).upper()


def _extract_citation_indices(text: str) -> list[int]:
    """Parse [1], [2], [3] markers from generated answer text."""
    return sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", text)))


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


async def generate_dataset(
    n_examples: int = 50,
    n_per_lang: int = 60,
    group_size: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (generator_data, rewriter_data) produced in a single generation pass.
    Both lists have the same length — every example contributes to both datasets.
    """
    cfg = get_settings()

    # Get Teacher model credentials from Config
    model_id = cfg.synthetic_data_llm_model
    provider = cfg.synthetic_data_llm_provider
    api_base = cfg.synthetic_data_llm_base_url

    # Handle API Key retrieval with fallback to the global DeepSeek key
    api_key_val = (
        cfg.synthetic_data_llm_api_key.get_secret_value()
        if cfg.synthetic_data_llm_api_key
        else ""
    )
    if not api_key_val and cfg.deepseek_api_key:
        api_key_val = cfg.deepseek_api_key.get_secret_value()

    if not api_key_val:
        raise RuntimeError(
            f"No API Key found for synthetic data teacher model ({model_id}).\n"
            "Set SYNTHETIC_DATA_LLM_API_KEY or DEEPSEEK_API_KEY in your .env."
        )

    # Instantiate the teacher model via DSPy using configured settings.
    # We use the provider/model string format (e.g. "openai/deepseek-chat")
    # which LiteLLM/DSPy understands.
    teacher_lm = dspy.LM(
        f"{provider}/{model_id}",
        api_key=api_key_val,
        api_base=api_base or None,
        max_tokens=1200,
        temperature=0.7,
    )
    dspy.configure(lm=teacher_lm)

    qdrant_client = AsyncQdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key or None,
        timeout=30,
    )

    logger.info("fetch_chunks_start", extra={"n_per_lang": n_per_lang})
    chunks = await fetch_real_chunks(
        collection=cfg.qdrant_collection_chunks,
        client=qdrant_client,
        n_per_lang=n_per_lang,
    )
    logger.info("fetch_chunks_done", extra={"total": len(chunks)})

    if len(chunks) < group_size:
        raise RuntimeError(
            f"Only {len(chunks)} usable chunks in Qdrant — need at least {group_size}.\n"
            "Ingest medical documents first before generating synthetic training data."
        )

    context_groups = make_context_groups(chunks, group_size=group_size, total_groups=n_examples)

    teacher = dspy.ChainOfThought(GenerateSyntheticMedicalQA)

    generator_data: list[dict] = []
    rewriter_data: list[dict] = []

    # Use the supported languages defined in the Enum
    supported_langs = [Language.ZH, Language.EN, Language.DE]

    logger.info("generation_start", extra={"n_examples": n_examples})

    for i, context in enumerate(context_groups):
        tier = _sample_tier()
        persona = random.choice(PERSONAS)
        use_history = random.choice([True, False])

        # Select language enum and map to full name for the prompt
        lang_enum = supported_langs[i % len(supported_langs)]
        lang_name = _LANG_MAP[lang_enum]

        try:
            pred = teacher(
                real_context=context,
                target_tier=tier,
                target_language=lang_name,
                simulated_persona=persona,
                requires_history=use_history,
            )

            citations = _extract_citation_indices(pred.grounded_answer)
            is_standalone = not use_history

            # Generator training record — matches ANSWER_DATA format in optimize.py
            generator_data.append(
                {
                    "context": context,
                    "history": pred.synthetic_history,
                    "question": pred.expected_rewritten_query,
                    "target_language": lang_name,
                    "answer": pred.grounded_answer,
                    "citations": citations,
                    "disclaimer": pred.disclaimer,
                }
            )

            # Rewriter training record — matches REWRITE_DATA format in optimize.py
            rewriter_data.append(
                {
                    "history": pred.synthetic_history,
                    "latest_message": pred.synthetic_latest_message,
                    "result": {
                        "is_standalone": is_standalone,
                        "rewritten_query": pred.expected_rewritten_query,
                    },
                    "expected_tier": tier,
                }
            )

            logger.info(
                f"generated_example [{i + 1}/{n_examples}]",
                extra={
                    "tier": tier,
                    "standalone": is_standalone,
                    "lang": lang_name,
                },
            )

        except Exception:
            logger.error(
                f"generation_failed [{i + 1}/{n_examples}]",
                exc_info=True,
                extra={"tier": tier, "lang": lang_name},
            )

    return generator_data, rewriter_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    n_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    generator_data, rewriter_data = asyncio.run(generate_dataset(n_examples=n_examples))

    if not generator_data:
        print("\nNo examples generated — check errors above.")
        sys.exit(1)

    out_dir = ROOT / "data" / "dspy"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_path = out_dir / "synthetic_generator_data.json"
    rew_path = out_dir / "synthetic_rewriter_data.json"

    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(generator_data, f, ensure_ascii=False, indent=2)

    with open(rew_path, "w", encoding="utf-8") as f:
        json.dump(rewriter_data, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"  Generator data : {len(generator_data)} examples → {gen_path}")
    print(f"  Rewriter data  : {len(rewriter_data)} examples  → {rew_path}")
    print("\nNext step: run scripts/dspy/optimize.py to compile both modules.")


if __name__ == "__main__":
    main()
