import asyncio
import sys
import types
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mock structlog and other potential missing deps before importing medgraphia
mock_structlog = types.ModuleType("structlog")
sys.modules["structlog"] = mock_structlog


class MockLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


import medgraphia.logger

medgraphia.logger.get_logger = lambda name: MockLogger()

from medgraphia.config import get_settings
from medgraphia.graph.client import close_driver, get_driver, ping


async def check_connection():
    settings = get_settings()
    print(f"Connecting to Neo4j at: {settings.neo4j_uri}")
    print(f"User: {settings.neo4j_user}")

    is_up = await ping()

    if is_up:
        print("✅ Success! Neo4j is reachable and authentication passed.")
        # Optional: Check if we can run a simple query
        driver = await get_driver()
        async with driver.session() as session:
            result = await session.run("RETURN 'Hello from Neo4j!' AS message")
            record = await result.single()
            print(f"Query test: {record['message']}")
    else:
        print("❌ Failed! Could not connect to Neo4j.")
        print("Check if Neo4j is running and if the credentials in .env or config.py are correct.")

    await close_driver()


if __name__ == "__main__":
    asyncio.run(check_connection())
