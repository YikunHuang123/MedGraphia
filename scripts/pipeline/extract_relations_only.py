
import sys
import asyncio
import os
from pathlib import Path

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.config import get_settings
from medgraphia.logger import configure_logging, get_logger
from medgraphia.graph.queries import get_chunks_from_db
from medgraphia.ingestion.relation_extractor import RelationExtractor

logger = get_logger(__name__)

async def main():
    cfg = get_settings()
    configure_logging(cfg.log_level)
    
    print("\n" + "="*60)
    print("  MedGraphia — Standalone Relation Extraction")
    print("="*60 + "\n")
    
    # 1. Fetch UNPROCESSED chunks from DB
    print("Step 1: Fetching unprocessed chunks from Neo4j...")
    from medgraphia.graph.client import get_session
    
    # We query specifically for chunks where re_processed is not true
    cypher = """
    MATCH (c:Chunk)
    WHERE c.re_processed IS NULL OR c.re_processed = false
    OPTIONAL MATCH (c)-[:FROM_DOC]->(d:Document)
    OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c)
    RETURN c, d, collect(e) AS entities
    LIMIT 40000
    """
    
    # Re-using the reconstruction logic from get_chunks_from_db but with our custom query
    from medgraphia.domain import SourceMeta, Language, Entity, EntityType, Chunk as ChunkObj
    chunks = []
    async with get_session() as session:
        result = await session.run(cypher)
        async for record in result:
            c_node = record["c"]
            d_node = record["d"]
            e_nodes = record["entities"]
            source = SourceMeta(
                source_id=c_node.get("source_id", "unknown"),
                source_title=d_node.get("source_title", "Unknown Document") if d_node else "Unknown",
                source_version=c_node.get("source_version", "1.0")
            )
            chunk_entities = []
            for e in e_nodes:
                if e.get("cui") and e.get("label"):
                    e_type_str = list(set(e.labels) - {"Entity"})[0] if e.labels else "Disease"
                    chunk_entities.append(Entity(cui=e["cui"], label=e["label"], entity_type=EntityType(e_type_str)))
            
            chunks.append(ChunkObj(
                chunk_id=c_node["chunk_id"], doc_id=c_node["doc_id"], text=c_node["text"],
                language=Language(c_node.get("language", "en")), source=source, entities=chunk_entities
            ))

    print(f"✓ Found {len(chunks)} unprocessed chunks.\n")
    
    if not chunks:
        print("✅ All chunks have been processed. Nothing to do!")
        return

    # 2. Initialize Extractor
    print("Step 2: Initializing Relation Extractor from settings...")
    
    # Debug/Force settings (Sensitive info is redacted in logs)
    api_key = cfg.openai_api_key.get_secret_value() if hasattr(cfg.openai_api_key, "get_secret_value") else str(cfg.openai_api_key)
    if api_key and api_key != "dummy":
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = cfg.openai_base_url
        print(f"  → Environment variables forced: OPENAI_BASE_URL={cfg.openai_base_url}")
        print(f"  → API Key verified (prefix): {api_key[:10]}...")
    else:
        print("  ⚠️ Warning: OPENAI_API_KEY is empty or 'dummy' in settings!")

    extractor = RelationExtractor.from_settings()
    print(f"✓ Extractor initialized: {extractor._extracted_by}\n")

    # 3. Run Extraction in Batches (Checkpointing)
    CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))
    BATCH_SIZE = 200
    
    print(f"Step 3: Processing {len(chunks)} chunks in batches of {BATCH_SIZE} (Concurrency: {CONCURRENCY})...")
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n--- Batch {current_batch_num}/{total_batches} ({len(batch)} chunks) ---")
        
        # Extract
        relations = await extractor.extract_batch(batch, max_workers=CONCURRENCY)
        
        # Write relations
        if relations:
            await extractor.write_relations_to_neo4j(relations)
            print(f"  ✓ Wrote {len(relations)} relations.")
        
        # Mark chunks as processed (Checkpoint)
        chunk_ids = [c.chunk_id for c in batch]
        async with get_session() as session:
            await session.run(
                "MATCH (c:Chunk) WHERE c.chunk_id IN $ids SET c.re_processed = true",
                ids=chunk_ids
            )
        print(f"  ✓ Checkpoint saved: {len(batch)} chunks marked as processed.")

    print(f"\n" + "="*60)
    print(f"  SUCCESS: All {len(chunks)} chunks processed.")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
