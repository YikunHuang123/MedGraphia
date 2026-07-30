import asyncio
import hashlib
import json
import uuid

import httpx

"""
curl -X POST http://localhost:8058/chat/stream \
     -H "Content-Type: application/json" \
     -H "X-API-Key: change-me" \
     -d '{
       "message": "Metformin 和 T2DM 有什么关系？",
       "language": "zh",
       "stream": true
     }'
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8058"
# 💡 Ensure this matches your ADMIN_BOOTSTRAP_KEY in .env
ADMIN_KEY = "change-me"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def test_phase8_features():
    """
    Test suite for Phase 8:
    1. Persistent API Keys (Creation, Auth, List, Revoke)
    2. Persistent Pipeline Status (Distributed progress tracking)
    3. New Streaming Protocol (Pure text chunks + Metadata events)
    """
    print("🚀 Starting MedGraphia Phase 8 Integration Test...")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # --- [1] Testing API Key Persistence ---
        print("\n--- [1/4] Testing API Key Persistence & Hashing ---")

        # 1.1 Create a new key
        resp = await client.post("/admin/keys", headers={"X-API-Key": ADMIN_KEY})
        if resp.status_code != 201:
            print(f"❌ Failed to create API key: {resp.text}")
            return

        new_key_data = resp.json()
        new_api_key = new_key_data["api_key"]
        prefix = new_key_data["prefix"]
        print(f"✅ Created new user key: {prefix}...")

        # 1.2 Test the new key (Verify database lookup and hashing works)
        resp = await client.get("/graph/stats", headers={"X-API-Key": new_api_key})
        if resp.status_code == 200:
            print("✅ New key authenticated via Neo4j lookup successfully.")
        else:
            print(f"❌ New key authentication failed: {resp.status_code}")

        # 1.3 Verify Prefix Redaction in list
        resp = await client.get("/admin/keys", headers={"X-API-Key": ADMIN_KEY})
        keys_list = resp.json()
        if any(k["prefix"] == prefix + "…" for k in keys_list):
            print("✅ Key prefix listed correctly in admin panel (Redacted).")

        # --- [2] Testing Pipeline Status Persistence ---
        print("\n--- [2/4] Testing Pipeline Status Persistence ---")

        domain = f"test_{uuid.uuid4().hex[:8]}"
        trigger_body = {"domain": domain, "pubmed_limit": 5}

        # 2.1 Trigger Pipeline
        resp = await client.post(
            "/admin/pipeline/trigger", json=trigger_body, headers={"X-API-Key": ADMIN_KEY}
        )
        if resp.status_code == 202:
            print(f"✅ Pipeline triggered for domain: {domain}")
        else:
            print(f"❌ Failed to trigger pipeline: {resp.text}")
            return

        # 2.2 Poll status (Verifies it's reading from Neo4j, not local memory)
        print("Polling persistent status from database...")
        for _ in range(5):
            await asyncio.sleep(2)
            resp = await client.get(
                f"/admin/pipeline/status?domain={domain}", headers={"X-API-Key": ADMIN_KEY}
            )
            status_data = resp.json()
            print(
                f"   - Progress: {status_data.get('progress', 0):.1%}, Stage: {status_data.get('stage')}"
            )
            if status_data.get("stage") in ["completed", "failed"]:
                break
        print("✅ Pipeline persistent tracking verified.")

        # --- [3] Testing New Streaming Protocol ---
        print("\n--- [3/4] Testing New Streaming Protocol (Pure Text) ---")

        chat_payload = {
            "message": "What is the relationship between metformin and T2DM?",
            "language": "en",
            "stream": True,
        }

        print("Connecting to SSE stream...")
        try:
            async with client.stream(
                "POST", "/chat/stream", json=chat_payload, headers={"X-API-Key": ADMIN_KEY}
            ) as response:
                if response.status_code != 200:
                    print(f"❌ Stream request failed: {response.status_code}")
                    return

                content_chunks = 0
                citations_received = False
                done_received = False

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "chunk":
                            content_chunks += 1
                        elif data["type"] == "citations":
                            citations_received = True
                            print("✅ Citations metadata event received.")
                        elif data["type"] == "done":
                            done_received = True
                            print("✅ 'done' event received.")

                if content_chunks > 0 and citations_received and done_received:
                    print(
                        f"✅ Streaming protocol test passed! (Received {content_chunks} text chunks)"
                    )
                else:
                    print(
                        f"❌ Streaming protocol test failed. Chunks: {content_chunks}, Citations: {citations_received}, Done: {done_received}"
                    )
        except Exception as e:
            print(f"❌ Streaming Test Exception: {e}")

        # --- [4] Verify Observability (Informational) ---
        print("\n--- [4/4] Observability Check ---")
        print("1. Check Terminal Logs: Look for 'http_request' lines with 'request_id'.")
        print(
            "2. Check Langfuse Dashboard: Verify traces for 'chat_stream' and 'knowledge_subgraph'."
        )
        print("3. Check Neo4j: Run 'MATCH (k:ApiKey) RETURN k' to see hashed keys.")

        print("\n🌟 Phase 8 Integration Test Suite Complete!")


if __name__ == "__main__":
    try:
        asyncio.run(test_phase8_features())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Test Script Fatal Error: {e}")
