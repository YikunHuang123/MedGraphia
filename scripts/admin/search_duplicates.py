import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from medgraphia.config import get_settings

async def search_duplicates():
    cfg = get_settings()
    client = AsyncQdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key or None)
    collection_name = cfg.qdrant_collection_chunks
    
    target_text = "Postmarketing cases of metformin-associated lactic acidosis"
    print(f"Searching for text starting with: '{target_text}'")
    
    # Use a filter to find exact matches or starts-with if supported, 
    # but here we'll just scroll and check manually for a larger batch
    points, _ = await client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
    )
    
    matches = []
    for p in points:
        text = p.payload.get("text", "")
        if target_text in text:
            matches.append({
                "id": p.id,
                "doc_id": p.payload.get("doc_id"),
                "text_len": len(text),
                "preview": text[:100]
            })
            
    print(f"Found {len(matches)} matches.")
    for m in matches:
        print(f"ID: {m['id']} | DocID: {m['doc_id']} | Len: {m['text_len']} | Preview: {m['preview']}...")

if __name__ == "__main__":
    asyncio.run(search_duplicates())
