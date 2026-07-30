import asyncio
import sys
from pathlib import Path

# Add src to path to allow importing medgraphia module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.graph.queries import get_chunks_from_db
from medgraphia.ingestion.re.pair_builder import PairBuilder


async def main():
    print("Connecting to Neo4j and fetching 2 random chunks...")
    try:
        # get_chunks_from_db returns chunks that haven't had relations extracted yet,
        # along with all their linked entities.
        chunks = await get_chunks_from_db(limit=2)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    if not chunks:
        print("No chunks found in the database. Please make sure the DB is populated.")
        return
        
    builder = PairBuilder()
    
    for chunk in chunks:
        print(f"\n{'='*60}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Original Text: {chunk.text[:200]}... (truncated)")
        print(f"Number of entities: {len(chunk.entities)}")
        
        pairs = builder.build_pairs(chunk)
        print(f"Generated {len(pairs)} permutations:")
        print(f"{'='*60}\n")
        
        # Display up to 5 pairs to avoid console spam
        for i, p in enumerate(pairs[:5]):
            print(f"Pair {i+1}: Source={p.source.label}, Target={p.target.label}")
            print(f"Marked Text: {p.marked_text}\n")
            
        if len(pairs) > 5:
            print(f"... and {len(pairs) - 5} more pairs omitted.")

if __name__ == "__main__":
    asyncio.run(main())
