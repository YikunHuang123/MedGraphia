
import asyncio
import sys
import logging
from pathlib import Path

# Add src to sys.path to allow importing medgraphia
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure basic logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

from medgraphia.retrieval.pipeline import RetrievalPipeline
from medgraphia.config import get_settings

async def main():
    print("=== MedGraphia Retrieval Pipeline Test ===")
    
    # 1. Initialize Pipeline
    # We use from_settings() which uses lazy loading internally
    pipeline = RetrievalPipeline.from_settings()
    
    # 2. Define test queries
    test_queries = [
        "什么是肾衰竭?",
        "What is the interaction between metformin and sitagliptin?",
        "How to treat type 2 diabetes?",
        "Give me an overview of cardiovascular disease prevalence.",
        "Is aspirin safe for a patient with stomach ulcers?"
    ]
    
    for query in test_queries:
        print(f"\n>>> Query: {query}")
        print("-" * 50)
        
        try:
            # 3. Execute Pipeline
            # This will trigger: NER -> Routing -> Retrieval -> Fusion -> Reranking
            result = await pipeline.execute(query, top_k=3)
            
            # 4. Display Results
            print(f"Reranked: {result.reranked}")
            
            if not result.items:
                print("No context items found (databases might be empty or unreachable).")
            else:
                for i, item in enumerate(result.items, 1):
                    score_str = f"RRF: {item.rrf_score:.4f}"
                    if "reranker_score" in item.metadata:
                        score_str += f" | Rerank: {item.metadata['reranker_score']:.4f}"
                        
                    print(f"{i}. [{item.source.value}] {score_str}")
                    # Print first 200 chars of text
                    text_snippet = item.text.replace('\n', ' ')[:200]
                    print(f"   Text: {text_snippet}...")
                    
        except Exception as e:
            print(f"❌ Error executing pipeline: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
