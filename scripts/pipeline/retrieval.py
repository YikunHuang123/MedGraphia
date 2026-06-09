import asyncio
import logging
import sys
from pathlib import Path

# Add src to sys.path to allow importing medgraphia
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure basic logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from medgraphia.retrieval.pipeline import RetrievalPipeline
from medgraphia.domain.base import Language


async def main():
    print("=== MedGraphia Retrieval Pipeline Test (UI Simulation Mode) ===")

    # 1. Initialize Pipeline
    pipeline = RetrievalPipeline.from_settings()

    # 2. Define test queries
    test_queries = [
        # "什么是肾衰竭?",
        # "What is the interaction between metformin and sitagliptin?",
        # "How to treat type 2 diabetes?",
        # "Give me an overview of cardiovascular disease prevalence.",
        # "Is aspirin safe for a patient with stomach ulcers?",
        "What are the risk factors and management strategies for metformin-associated lactic acidosis?"
    ]

    for query in test_queries:
        print(f"\n>>> Query: {query}")
        print("-" * 50)

        try:
            # 3. Detect language as UI does
            language = Language.detect(query)
            if language == Language.UNKNOWN:
                language = Language.EN
            print(f"[*] Detected Language: {language.value}")

            # 4. Execute Pipeline
            # We use default top_k=10 to match UI, and pass empty history for first turn
            result = await pipeline.execute(
                query=query,
                history=[],
                language=language,
                user_id="anonymous"
            )

            # 5. Display Results
            print(f"[*] Complexity Tier: {getattr(result, 'complexity_tier', 'N/A')}")
            print(f"[*] Total Reranked Items: {len(result.items)}")

            if not result.items:
                print("No context items found (databases might be empty or unreachable).")
            else:
                for i, item in enumerate(result.items, 1):
                    score_str = f"RRF: {item.rrf_score:.4f}"
                    if "reranker_score" in item.metadata:
                        score_str += f" | Rerank: {item.metadata['reranker_score']:.4f}"

                    print(f"\n{i}. [{item.source.value}] {score_str}")
                    # Print first 300 chars to check for truncation
                    text_snippet = item.text.replace("\n", " ").strip()
                    print(f"   Text: {text_snippet[:1000]}...")


        except Exception as e:
            print(f"❌ Error executing pipeline: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
