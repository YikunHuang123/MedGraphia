"""
Clean Short Chunks

Deletes "junk" chunks (e.g., those containing only an entity name or extremely short text)
from both Qdrant and Neo4j.

CRITICAL: This script deletes `Chunk` nodes in Neo4j but preserves `Entity` nodes
and their relationships, keeping the Knowledge Graph intact
"""

from __future__ import annotations

import asyncio
import re

from qdrant_client import QdrantClient

from medgraphia.config import settings
from medgraphia.graph.client import get_driver
from medgraphia.logger import get_logger

logger = get_logger("admin.clean_short_chunks")


def is_junk_text(text: str) -> bool:
    """
    Determine if a chunk of text is considered "junk" based on length heuristics.
    """
    if not text:
        return True

    # Remove all punctuation, whitespace, and numbers
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    cleaned_text = re.sub(r"\s+", "", cleaned_text)
    cleaned_text = re.sub(r"\d+", "", cleaned_text)

    # Check if text contains Chinese characters
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", cleaned_text))

    if has_chinese:
        # For Chinese, a valid chunk should generally have more than 8 meaningful characters
        return len(cleaned_text) < 8
    else:
        # For English/German, a valid chunk should generally have more than 20 meaningful letters (e.g. 3-4 words)
        return len(cleaned_text) < 20


async def main():
    logger.info("Starting short chunk cleanup process...")

    # 1. Connect to Qdrant
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    collection_name = settings.qdrant_collection_chunks

    # 2. Connect to Neo4j
    driver = await get_driver()

    junk_ids = []
    total_processed = 0

    try:
        # Scroll through all points in Qdrant
        logger.info(f"Scanning Qdrant collection: {collection_name}")
        offset = None
        while True:
            records, next_offset = qdrant.scroll(
                collection_name=collection_name, limit=1000, offset=offset, with_payload=True
            )

            for record in records:
                total_processed += 1
                text = record.payload.get("text", "") if record.payload else ""

                if is_junk_text(text):
                    junk_ids.append(record.id)
                    # Print an example of what's being deleted (every 50th for brevity)
                    if len(junk_ids) % 50 == 1:
                        logger.info(f"Marked for deletion: '{text[:30]}...'")

            offset = next_offset
            if offset is None:
                break

        logger.info(
            f"Scan complete. Total chunks: {total_processed}, Junk chunks found: {len(junk_ids)}"
        )

        if not junk_ids:
            logger.info("No junk chunks found. Exiting.")
            return

        # 3. Delete from Qdrant
        batch_size = 500
        for i in range(0, len(junk_ids), batch_size):
            batch = junk_ids[i : i + batch_size]
            qdrant.delete(collection_name=collection_name, points_selector=batch)
        logger.info(f"✅ Deleted {len(junk_ids)} points from Qdrant.")

        # 4. Delete from Neo4j (Chunk nodes only, NOT Entity nodes)
        async with driver.session() as session:
            for i in range(0, len(junk_ids), batch_size):
                batch = junk_ids[i : i + batch_size]
                # DETACH DELETE ensures relationships to Entities are removed, but Entities remain
                query = """
                MATCH (c:Chunk) 
                WHERE c.id IN $ids 
                DETACH DELETE c
                """
                await session.run(query, ids=batch)
        logger.info(f"✅ Deleted {len(junk_ids)} Chunk nodes from Neo4j.")

        logger.info(
            "Cleanup process finished successfully! Knowledge Graph entities remain intact."
        )

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
