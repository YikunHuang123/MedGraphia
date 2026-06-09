import asyncio
from qdrant_client import AsyncQdrantClient
from medgraphia.config import get_settings

async def check_duplicates():
    cfg = get_settings()
    client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None)
    collection_name = cfg.qdrant_collection_chunks
    
    print(f"Checking collection: {collection_name}")
    
    # 1. Get total count
    count_result = await client.count(collection_name=collection_name)
    print(f"Total points: {count_result.count}")
    
    # 2. Scroll through points to find duplicate texts
    points, _ = await client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False
    )
    
    texts = {}
    duplicates = 0
    for p in points:
        text = p.payload.get("text", "")
        if text in texts:
            duplicates += 1
            texts[text].append(p.id)
        else:
            texts[text] = [p.id]
            
    if duplicates > 0:
        print(f"Found approximately {duplicates} duplicates in the first 100 points.")
        # Print one example
        for text, ids in texts.items():
            if len(ids) > 1:
                print(f"\nExample Duplicate Text (truncated): {text[:100]}...")
                print(f"IDs: {ids}")
                break
    else:
        print("No duplicates found in the first 100 points.")

if __name__ == "__main__":
    asyncio.run(check_duplicates())
