import asyncio
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.graph.client import close_driver, get_driver


async def count_nodes():
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run("MATCH (n) RETURN labels(n) as label, count(*) as count")
        records = await result.data()
        print("\n--- 📊 Neo4j Node Counts ---")
        for record in records:
            print(f"{record['label']}: {record['count']}")

        result = await session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
        records = await result.data()
        print("\n--- 🔗 Neo4j Relation Counts ---")
        for record in records:
            print(f"{record['type']}: {record['count']}")

    await close_driver()


if __name__ == "__main__":
    asyncio.run(count_nodes())
