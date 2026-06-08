"""
MedGraphia End-to-End Test: From Query to Cited Answer.
"""

import asyncio

# Set logging level
import logging
import sys

from medgraphia.domain.base import Language
from medgraphia.generation.pipeline import GenerationPipeline
from medgraphia.retrieval.pipeline import RetrievalPipeline

logging.getLogger("medgraphia").setLevel(logging.INFO)


async def ask_medgraphia(query: str):
    # Dynamically detect language from the query text
    lang = Language.detect(query)

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print(f"TARGET LANGUAGE: {lang.value} (Detected)")
    print("=" * 80)

    # 1. Initialize core pipelines
    retrieval_pl = RetrievalPipeline.from_settings()
    generation_pl = GenerationPipeline.from_settings()

    # 2. Retrieval
    print("\n[1/3] Retrieving from Medical Knowledge Base...")
    ret_result = await retrieval_pl.execute(query)
    print(f"  - Intent: {ret_result.query_type.value}")
    print(f"  - Evidence Found: {len(ret_result.items)} items (Graph/Vector/Community)")

    # 3. Generation
    print("\n[2/3] Orchestrating LLM Generation...")

    # DEBUG: Show what we are sending to the LLM
    from medgraphia.generation.citation import build_numbered_context

    context_str = build_numbered_context(ret_result.items)

    # We can peek at the system prompt by calling the internal helper
    from medgraphia.prompts import get_system_prompt as _get_system_prompt

    sys_prompt = _get_system_prompt(ret_result.query_type, lang)

    print("\n" + "-" * 30 + " LLM INPUT DEBUG " + "-" * 30)
    print(f"[SYSTEM PROMPT]\n{sys_prompt}")
    print(f"\n[CONTEXT PASSAGES]\n{context_str}")
    print("-" * 77 + "\n")

    gen_result = await generation_pl.generate(
        question=query,
        query_type=ret_result.query_type,
        retrieved_items=ret_result.items,
        language=lang,
    )

    # 4. Results
    print("\n[3/3] Final Response:")
    print("-" * 40)
    if gen_result.routing:
        print(
            f"MODEL ROUTING: {gen_result.routing.provider.value} / {gen_result.routing.model_name} ({gen_result.routing.tier.value} tier)"
        )

    print(f"\nANSWER:\n{gen_result.answer}")

    if gen_result.disclaimer:
        print(f"\nDISCLAIMER: {gen_result.disclaimer}")

    if gen_result.citations:
        print("\nCITATIONS:")
        for cit in gen_result.citations:
            print(f"  [{cit.citation_number}] {cit.source_title} - {cit.section_path}")

    print(f"\nLATENCY: {gen_result.latency_ms:.0f}ms")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example question
    test_query = "What are the side effects of Metformin?"
    if len(sys.argv) > 1:
        test_query = sys.argv[1]

    asyncio.run(ask_medgraphia(test_query))
