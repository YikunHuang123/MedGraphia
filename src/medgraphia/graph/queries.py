"""
Cypher query library.  All graph read/write operations live here so that
the rest of the codebase never constructs Cypher strings inline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from medgraphia.domain import Chunk, Entity, EntityType, Message, RawDocument, Relation, Session
from medgraphia.graph.client import get_session
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Matches any known entity-type node label (excludes Unknown), generated from
# EntityType so it stays in sync when categories are added or removed.
_ENTITY_LABEL_DISJUNCTION = " OR ".join(
    f"e:{t.value}" for t in EntityType if t is not EntityType.UNKNOWN
)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def upsert_document(doc: RawDocument) -> None:
    """Merge a Document node (update if doc_id already exists)."""
    cypher = """
    MERGE (d:Document {doc_id: $doc_id})
    SET d.title        = $title,
        d.source_id    = $source_id,
        d.source_title = $source_title,
        d.language     = $language,
        d.format       = $format,
        d.retrieved_at = $retrieved_at
    """
    async with get_session() as session:
        await session.run(
            cypher,
            doc_id=doc.doc_id,
            title=doc.title,
            source_id=doc.source.source_id,
            source_title=doc.source.source_title,
            language=doc.language.value,
            format=doc.format,
            retrieved_at=doc.source.retrieved_at.isoformat(),
        )


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


async def create_chunk(chunk: Chunk) -> None:
    """Create a Chunk node and link it to its parent Document."""
    cypher = """
    MERGE (c:Chunk {chunk_id: $chunk_id})
    SET c.doc_id       = $doc_id,
        c.text         = $text,
        c.section_path = $section_path,
        c.language     = $language,
        c.token_count  = $token_count,
        c.page         = $page,
        c.char_offset  = $char_offset,
        c.source_id    = $source_id,
        c.source_version = $source_version
    WITH c
    MATCH (d:Document {doc_id: $doc_id})
    MERGE (c)-[:FROM_DOC]->(d)
    """
    async with get_session() as session:
        await session.run(
            cypher,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            section_path=chunk.section_path,
            language=chunk.language.value,
            token_count=chunk.token_count,
            page=chunk.page,
            char_offset=chunk.char_offset,
            source_id=chunk.source.source_id,
            source_version=chunk.source.source_version,
        )


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


async def merge_entity(entity: Entity) -> None:
    """
    MERGE entity node on CUI.  Updates label, type, and lang_labels.
    Works for any EntityType via dynamic label.
    """
    label = entity.entity_type.value  # e.g. "Disease"
    cypher = """
    MERGE (e:{label} {cui: $cui})
    SET e.label      = $label,
        e.lang_zh    = $lang_zh,
        e.lang_de    = $lang_de,
        e.confidence = $confidence
    """.replace("{label}", label)
    async with get_session() as session:
        await session.run(
            cypher,
            cui=entity.cui,
            label=entity.label,
            lang_zh=entity.lang_labels.get("zh", ""),
            lang_de=entity.lang_labels.get("de", ""),
            confidence=entity.confidence,
        )


async def link_entity_to_chunk(cui: str, entity_type: str, chunk_id: str) -> None:
    """Create a MENTIONED_IN edge between an entity node and a Chunk node."""
    cypher = """
    MATCH (e:{entity_type} {cui: $cui})
    MATCH (c:Chunk {chunk_id: $chunk_id})
    MERGE (e)-[:MENTIONED_IN]->(c)
    """.replace("{entity_type}", entity_type)
    async with get_session() as session:
        await session.run(cypher, cui=cui, chunk_id=chunk_id)


async def batch_upsert_entities_and_links(chunks: list[Chunk]) -> None:
    """
    High-performance UNWIND batch insertion for Entities and MENTIONED_IN links.
    Groups entities by their EntityType to process each label dynamically.
    Reduces N+1 queries from hundreds of thousands down to a few queries per batch.
    """
    from collections import defaultdict

    # Group payload by entity type because Neo4j UNWIND doesn't support dynamic labels
    payload_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    
    for chunk in chunks:
        for ent in chunk.entities:
            # Derive mention surface form from NER span if available.
            # Stored on the MENTIONED_IN edge so RE training can reconstruct exact
            # marker positions without re-scanning chunk text.
            mention_text = ""
            if ent.start_char is not None and ent.end_char is not None:
                mention_text = chunk.text[ent.start_char:ent.end_char]

            payload_by_type[ent.entity_type.value].append(
                {
                    "cui": ent.cui,
                    "label": ent.label,
                    "lang_zh": ent.lang_labels.get("zh", ""),
                    "lang_de": ent.lang_labels.get("de", ""),
                    "confidence": ent.confidence,
                    "chunk_id": chunk.chunk_id,
                    "mention_text": mention_text,
                    "start_char": ent.start_char if ent.start_char is not None else -1,
                    "end_char": ent.end_char if ent.end_char is not None else -1,
                }
            )

    async with get_session() as session:
        for label, payload in payload_by_type.items():
            if label == "Unknown":
                continue  # Never persist UNKNOWN entities to the graph
            cypher = f"""
            UNWIND $payload AS row
            MERGE (e:{label} {{cui: row.cui}})
            SET e.label      = row.label,
                e.lang_zh    = row.lang_zh,
                e.lang_de    = row.lang_de,
                e.confidence = row.confidence
            WITH e, row
            MATCH (c:Chunk {{chunk_id: row.chunk_id}})
            MERGE (e)-[r:MENTIONED_IN]->(c)
            SET r.mention_text = row.mention_text,
                r.start_char   = row.start_char,
                r.end_char     = row.end_char
            """
            await session.run(cypher, payload=payload)


async def get_all_entities() -> list[dict[str, Any]]:
    """
    Query Neo4j for all entity nodes and return them as plain dicts.
    Used by the embedding layer to sync graph entities to the vector store.
    """
    cypher = f"""
    MATCH (e)
    WHERE ({_ENTITY_LABEL_DISJUNCTION})
      AND e.cui IS NOT NULL AND e.label IS NOT NULL
    RETURN e.cui        AS cui,
           e.label      AS label,
           labels(e)[0] AS entity_type,
           coalesce(e.lang_zh, '') AS lang_zh,
           coalesce(e.lang_de, '') AS lang_de
    ORDER BY e.cui
    """
    entities: list[dict[str, Any]] = []
    try:
        async with get_session() as session:
            result = await session.run(cypher)
            async for record in result:
                if record["cui"] and record["label"]:
                    entities.append(
                        {
                            "cui": record["cui"],
                            "label": record["label"],
                            "entity_type": record["entity_type"] or "Unknown",
                            "lang_zh": record["lang_zh"] or "",
                            "lang_de": record["lang_de"] or "",
                        }
                    )
        logger.info("entity_fetch_done", count=len(entities))
    except Exception as exc:
        logger.warning("entity_fetch_failed", error=str(exc))
    return entities


async def get_chunks_from_db(limit: int | None = None) -> list[Chunk]:
    """
    Fetch stored chunks and their associated entities from Neo4j.
    Enables resuming the pipeline at the Relation Extraction stage.
    """
    from medgraphia.domain import Entity, EntityType, Language, SourceMeta

    cypher = """
    MATCH (c:Chunk)
    WHERE c.relations_extracted IS NULL
    OPTIONAL MATCH (c)-[:FROM_DOC]->(d:Document)
    OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c)
    RETURN c, d, collect(e) AS entities
    """
    if limit is not None:
        cypher += " LIMIT $limit"
        
    chunks: list[Chunk] = []
    async with get_session() as session:
        result = await session.run(cypher, limit=limit)
        async for record in result:
            c_node = record["c"]
            d_node = record["d"]
            e_nodes = record["entities"]

            # Reconstruct SourceMeta
            source = SourceMeta(
                source_id=c_node.get("source_id", "unknown"),
                source_title=d_node.get("source_title", "Unknown Document")
                if d_node
                else "Unknown",
                source_version=c_node.get("source_version", "1.0"),
            )

            # Reconstruct Entity objects
            chunk_entities = []
            for e in e_nodes:
                if e.get("cui") and e.get("label"):
                    # Determine entity type from labels (Disease, Drug, etc.)
                    e_type_str = list(set(e.labels) - {"Entity"})[0] if e.labels else "Disease"
                    chunk_entities.append(
                        Entity(
                            cui=e["cui"],
                            label=e["label"],
                            entity_type=EntityType(e_type_str),
                            confidence=e.get("confidence", 1.0),
                            lang_labels={
                                "zh": e.get("lang_zh", ""),
                                "de": e.get("lang_de", ""),
                            },
                        )
                    )

            # Reconstruct Chunk
            chunk = Chunk(
                chunk_id=c_node["chunk_id"],
                doc_id=c_node["doc_id"],
                text=c_node["text"],
                section_path=c_node.get("section_path", ""),
                language=Language(c_node.get("language", "en")),
                token_count=c_node.get("token_count", 0),
                page=c_node.get("page"),
                char_offset=c_node.get("char_offset", 0),
                source=source,
                entities=chunk_entities,
            )
            chunks.append(chunk)

    logger.info("chunks_fetched_from_db", count=len(chunks), with_entities=True)
    return chunks


async def get_entity_mention_counts(cuis: list[str]) -> dict[str, int]:
    """Return each CUI's total MENTIONED_IN count across the whole graph."""
    if not cuis:
        return {}
    cypher = """
    UNWIND $cuis AS cui
    MATCH (e {cui: cui})
    OPTIONAL MATCH (e)-[r:MENTIONED_IN]->(:Chunk)
    RETURN cui, count(r) AS mention_count
    """
    counts: dict[str, int] = {}
    async with get_session() as session:
        result = await session.run(cypher, cuis=cuis)
        async for record in result:
            counts[record["cui"]] = record["mention_count"]
    return counts


async def mark_chunks_extracted(chunk_ids: list[str]) -> None:
    """Mark chunks as having their relations extracted to support resuming."""
    if not chunk_ids:
        return
    cypher = """
    UNWIND $chunk_ids AS cid
    MATCH (c:Chunk {chunk_id: cid})
    SET c.relations_extracted = true
    """
    async with get_session() as session:
        await session.run(cypher, chunk_ids=chunk_ids)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


async def create_relation(relation: Relation) -> None:
    """
    Create a typed edge between two entity nodes.
    Uses MERGE to avoid duplicate edges for the same (source, target, type, chunk).
    """
    rel_type = relation.relation_type.value
    cypher = """
    MATCH (src {cui: $source_cui})
    MATCH (tgt {cui: $target_cui})
    MERGE (src)-[r:{rel_type} {chunk_id: $chunk_id}]->(tgt)
    SET r.evidence_text  = $evidence_text,
        r.source_id      = $source_id,
        r.confidence     = $confidence,
        r.extracted_by   = $extracted_by
    """.replace("{rel_type}", rel_type)
    async with get_session() as session:
        await session.run(
            cypher,
            source_cui=relation.source_cui,
            target_cui=relation.target_cui,
            chunk_id=relation.chunk_id,
            evidence_text=relation.evidence_text,
            source_id=relation.source_id,
            confidence=relation.confidence,
            extracted_by=relation.extracted_by,
        )


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------


async def upsert_community(community_id: str, summary: str, member_cuis: list[str]) -> None:
    """Create or update a Community node and link member entities."""
    cypher = """
    MERGE (com:Community {community_id: $community_id})
    SET com.summary = $summary,
        com.size    = $size
    """
    async with get_session() as session:
        await session.run(
            cypher,
            community_id=community_id,
            summary=summary,
            size=len(member_cuis),
        )
        for cui in member_cuis:
            await session.run(
                """
                MATCH (e {cui: $cui})
                MATCH (com:Community {community_id: $community_id})
                MERGE (e)-[:MEMBER_OF]->(com)
                """,
                cui=cui,
                community_id=community_id,
            )


# ---------------------------------------------------------------------------
# Query-time reads
# ---------------------------------------------------------------------------


async def get_subgraph(cui: str, hops: int = 2) -> dict[str, Any]:
    """
    Expand a 1- or 2-hop subgraph starting from a given CUI.

    Optimisations:
      - Excludes 'Chunk' nodes to avoid supernode path explosions.
      - Returns only essential serialisable properties to reduce payload size.
    """
    # ── Cypher: exclude Chunk nodes and filter paths ─────────────────────────
    cypher = """
    MATCH path = (start {cui: $cui})-[*1..{hops}]-(neighbor)
    WHERE ALL(n IN nodes(path) WHERE NOT n:Chunk)
    RETURN path
    LIMIT 300
    """.replace("{hops}", str(hops))

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    async with get_session() as session:
        result = await session.run(cypher, cui=cui)
        async for record in result:
            path = record["path"]
            for node in path.nodes:
                nid = str(node.element_id)
                if nid not in nodes:
                    # Only return UI-essential fields (no raw text)
                    nodes[nid] = {
                        "id": nid,
                        "cui": node.get("cui"),
                        "label": node.get("label"),
                        "labels": list(node.labels),
                        "lang_zh": node.get("lang_zh", ""),
                        "lang_de": node.get("lang_de", ""),
                    }
            for rel in path.relationships:
                edges.append(
                    {
                        "type": rel.type,
                        "source": str(rel.start_node.element_id),
                        "target": str(rel.end_node.element_id),
                        "confidence": rel.get("confidence", 1.0),
                    }
                )

    return {"nodes": list(nodes.values()), "edges": edges}


async def get_graph_stats() -> dict[str, int]:
    """
    Return counts of nodes and relationships.
    Uses fast APOC metadata or database statistics where possible.
    """
    async with get_session() as session:
        # ── Path A: APOC Metadata (Fastest) ──────────────────────────────────
        try:
            result = await session.run("CALL apoc.meta.stats() YIELD nodeCount, relCount")
            record = await result.single()
            if record:
                return {"nodes": int(record["nodeCount"]), "relations": int(record["relCount"])}
        except Exception:
            pass

        # ── Path B: Fast Count (O(1) on most modern Neo4j versions) ──────────
        try:
            # COUNT(*) on a label is usually optimized to use internal counters
            n_res = await session.run("MATCH (n) RETURN count(n) AS cnt")
            n_rec = await n_res.single()
            r_res = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            r_rec = await r_res.single()
            return {
                "nodes": n_rec["cnt"] if n_rec else 0,
                "relations": r_rec["cnt"] if r_rec else 0,
            }
        except Exception as exc:
            logger.warning("graph_stats_fallback_failed", error=str(exc))
            return {"nodes": 0, "relations": 0}


# ---------------------------------------------------------------------------
# Chat Persistence
# ---------------------------------------------------------------------------


async def upsert_chat_session(session: Session) -> None:
    """Create or update a ChatSession node."""
    cypher = """
    MERGE (s:ChatSession {session_id: $session_id})
    SET s.user_id    = $user_id,
        s.language   = $language,
        s.domain     = $domain,
        s.created_at = $created_at,
        s.updated_at = $updated_at
    """
    async with get_session() as g_session:
        await g_session.run(
            cypher,
            session_id=session.session_id,
            user_id=session.user_id,
            language=session.language.value,
            domain=session.domain,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
        )


async def save_chat_message(message: Message) -> None:
    """
    Persist a ChatMessage and link it to its Session.
    """
    import json

    cypher = """
    MATCH (s:ChatSession {session_id: $session_id})
    MERGE (m:ChatMessage {message_id: $message_id})
    SET m.role                 = $role,
        m.content              = $content,
        m.model_used           = $model_used,
        m.retrieval_paths_used = $retrieval_paths_used,
        m.created_at           = $created_at,
        m.citations_json       = $citations_json
    MERGE (s)-[:HAS_MESSAGE]->(m)
    """
    async with get_session() as g_session:
        await g_session.run(
            cypher,
            session_id=message.session_id,
            message_id=str(message.message_id),
            role=message.role,
            content=message.content,
            model_used=message.model_used,
            retrieval_paths_used=message.retrieval_paths_used,
            created_at=message.created_at.isoformat(),
            citations_json=json.dumps([c.model_dump() for c in message.citations]),
        )


async def get_chat_session(session_id: str) -> Session | None:
    """Retrieve a full Session with all its Messages from Neo4j."""
    import json

    from medgraphia.domain import Citation, Language, Message, Session

    cypher = """
    MATCH (s:ChatSession {session_id: $session_id})
    OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:ChatMessage)
    RETURN s, collect(m) AS messages
    """
    async with get_session() as g_session:
        result = await g_session.run(cypher, session_id=session_id)
        record = await result.single()
        if not record:
            return None

        s_node = record["s"]
        m_nodes = record["messages"]

        # Sort messages by created_at (since collect() order is non-deterministic)
        m_nodes = sorted(m_nodes, key=lambda x: x.get("created_at", ""))

        messages = []
        for m in m_nodes:
            # Parse citations back from JSON
            c_data = json.loads(m.get("citations_json", "[]"))
            citations = [Citation(**c) for c in c_data]

            messages.append(
                Message(
                    message_id=m["message_id"],
                    session_id=session_id,
                    role=m["role"],
                    content=m["content"],
                    citations=citations,
                    model_used=m.get("model_used", ""),
                    retrieval_paths_used=m.get("retrieval_paths_used", []),
                    created_at=datetime.fromisoformat(m["created_at"]),
                )
            )

        return Session(
            session_id=s_node["session_id"],
            user_id=s_node.get("user_id", "anonymous"),
            language=Language(s_node.get("language", "en")),
            domain=s_node.get("domain", ""),
            messages=messages,
            created_at=datetime.fromisoformat(s_node["created_at"]),
            updated_at=datetime.fromisoformat(s_node["updated_at"]),
        )


async def list_chat_sessions(user_id: str = "anonymous") -> list[dict[str, Any]]:
    """Return a summary of all chat sessions for a user."""
    cypher = """
    MATCH (s:ChatSession {user_id: $user_id})
    OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:ChatMessage)
    WITH s, m
    ORDER BY m.created_at ASC
    WITH s, collect(m) AS msgs
    RETURN s.session_id AS session_id,
           s.language   AS language,
           s.updated_at AS updated_at,
           size(msgs)   AS message_count,
           msgs[0].content AS first_message
    ORDER BY s.updated_at DESC
    """
    sessions = []
    async with get_session() as g_session:
        result = await g_session.run(cypher, user_id=user_id)
        async for record in result:
            sessions.append(dict(record))
    return sessions


# ---------------------------------------------------------------------------
# Semantic Memory (User Interests)
# ---------------------------------------------------------------------------


async def update_user_interests(user_id: str, cuis: list[str], decay_factor: float = 0.9) -> None:
    """
    Update the user's interest in a list of entities.
    ONLY updates existing Entity nodes. Does NOT create new medical nodes.
    """
    if not cuis:
        return

    # Filter for real CUIs only (in case any mentions leaked through)
    valid_cuis = [c for c in cuis if not c.startswith("MENTION:")]
    if not valid_cuis:
        return

    cypher = """
    MERGE (u:User {id: $user_id})
    WITH u
    UNWIND $cuis AS cui
    MATCH (e:Entity {cui: cui})
    MERGE (u)-[r:INTERESTED_IN]->(e)
    ON CREATE SET r.weight = 1.0, r.last_accessed = $now
    ON MATCH SET r.weight = (r.weight * $decay) + 1.0, r.last_accessed = $now
    """
    async with get_session() as g_session:
        await g_session.run(
            cypher,
            user_id=user_id,
            cuis=valid_cuis,
            decay=decay_factor,
            now=datetime.now().isoformat(),
        )
    logger.info("user_interests_updated", user_id=user_id, count=len(valid_cuis))


async def get_user_top_interests(user_id: str, limit: int = 10) -> list[str]:
    """Return the CUIs of the entities a user is most interested in."""
    cypher = """
    MATCH (u:User {id: $user_id})-[r:INTERESTED_IN]->(e:Entity)
    RETURN e.cui AS cui
    ORDER BY r.weight DESC
    LIMIT $limit
    """
    async with get_session() as g_session:
        result = await g_session.run(cypher, user_id=user_id, limit=limit)
        return [record["cui"] async for record in result]


# ---------------------------------------------------------------------------
# Admin & Auth Persistence
# ---------------------------------------------------------------------------


async def create_api_key_node(key_hash: str, prefix: str, role: str) -> None:
    """Persist a hashed API key and its metadata."""
    cypher = """
    MERGE (k:ApiKey {key_hash: $key_hash})
    SET k.prefix = $prefix,
        k.role   = $role,
        k.active = True,
        k.created_at = datetime()
    """
    async with get_session() as session:
        await session.run(cypher, key_hash=key_hash, prefix=prefix, role=role)


async def get_api_key_node(key_hash: str) -> dict[str, Any] | None:
    """Retrieve key metadata by its hash."""
    cypher = """
    MATCH (k:ApiKey {key_hash: $key_hash, active: True})
    RETURN k.role AS role, k.prefix AS prefix
    """
    async with get_session() as session:
        result = await session.run(cypher, key_hash=key_hash)
        record = await result.single()
        return dict(record) if record else None


async def list_api_keys() -> list[dict[str, Any]]:
    """List all active API key metadata (redacted)."""
    cypher = """
    MATCH (k:ApiKey {active: True})
    RETURN k.prefix AS prefix, k.role AS role
    ORDER BY k.created_at DESC
    """
    keys = []
    async with get_session() as session:
        result = await session.run(cypher)
        async for record in result:
            keys.append(dict(record))
    return keys


async def delete_api_keys_by_prefix(prefix: str) -> int:
    """Revoke (soft-delete) API keys matching a prefix."""
    cypher = """
    MATCH (k:ApiKey)
    WHERE k.prefix STARTS WITH $prefix
    SET k.active = False
    RETURN count(k) AS cnt
    """
    async with get_session() as session:
        result = await session.run(cypher, prefix=prefix)
        record = await result.single()
        return record["cnt"] if record else 0


async def upsert_pipeline_status(domain: str, status_data: dict[str, Any]) -> None:
    """Persist the current state of a pipeline run."""
    cypher = """
    MERGE (ps:PipelineStatus {domain: $domain})
    SET ps += $status_data,
        ps.updated_at = datetime()
    """
    async with get_session() as session:
        await session.run(cypher, domain=domain, status_data=status_data)


async def get_pipeline_status(domain: str) -> dict[str, Any] | None:
    """Retrieve the latest pipeline status for a domain."""
    cypher = """
    MATCH (ps:PipelineStatus {domain: $domain})
    RETURN ps
    """
    async with get_session() as session:
        result = await session.run(cypher, domain=domain)
        record = await result.single()
        if record:
            return _convert_neo4j_types(dict(record["ps"]))
        return None


def _convert_neo4j_types(data: Any) -> Any:
    """Recursively convert Neo4j-specific types (like DateTime) to standard Python types."""
    from datetime import datetime

    from neo4j.time import DateTime

    if isinstance(data, dict):
        return {k: _convert_neo4j_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_neo4j_types(i) for i in data]
    elif isinstance(data, DateTime):
        # Neo4j DateTime -> Standard Python datetime
        return datetime(
            data.year,
            data.month,
            data.day,
            data.hour,
            data.minute,
            int(data.second),
            tzinfo=data.tzinfo,
        )
    return data
