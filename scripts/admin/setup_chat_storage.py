#!/usr/bin/env python3
"""
Admin utility: Setup Neo4j constraints and indexes for chat persistence.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.graph.client import get_session


async def setup_chat_constraints() -> None:
    print("Setting up Neo4j constraints for chat storage...")

    queries = [
        # Session ID must be unique
        "CREATE CONSTRAINT chat_session_id IF NOT EXISTS FOR (s:ChatSession) REQUIRE s.session_id IS UNIQUE",
        # Message ID must be unique
        "CREATE CONSTRAINT chat_message_id IF NOT EXISTS FOR (m:ChatMessage) REQUIRE m.message_id IS UNIQUE",
        # Index on created_at for sorting performance
        "CREATE INDEX chat_message_created_at IF NOT EXISTS FOR (m:ChatMessage) ON (m.created_at)",
        # Index on user_id for session filtering
        "CREATE INDEX chat_session_user_id IF NOT EXISTS FOR (s:ChatSession) ON (s.user_id)",
    ]

    async with get_session() as session:
        for query in queries:
            try:
                await session.run(query)
                print(f"  ✓ Executed: {query}")
            except Exception as exc:
                print(f"  ❌ Failed: {query}\n    Error: {exc}")


if __name__ == "__main__":
    asyncio.run(setup_chat_constraints())
    print("\n✨ Neo4j chat constraints initialized successfully.\n")
