"""
Cypher query library.  All graph read/write operations live here so that
the rest of the codebase never constructs Cypher strings inline.
"""
from __future__ import annotations

from typing import Any

from medgraphia.domain import Chunk, Entity, Relation, RawDocument
from medgraphia.graph.client import get_session
from medgraphia.logger import get_logger

logger = get_logger(__name__)


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
    Works for Disease / Drug / Symptom / Gene / Procedure via dynamic label.
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
    Returns nodes and edges as serialisable dicts.
    """
    cypher = """
    MATCH path = (start {cui: $cui})-[*1..{hops}]-(neighbor)
    RETURN path
    LIMIT 500
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
                    nodes[nid] = {"id": nid, "labels": list(node.labels), **dict(node)}
            for rel in path.relationships:
                edges.append(
                    {
                        "type": rel.type,
                        "source": str(rel.start_node.element_id),
                        "target": str(rel.end_node.element_id),
                        **dict(rel),
                    }
                )

    return {"nodes": list(nodes.values()), "edges": edges}


async def get_graph_stats() -> dict[str, int]:
    """Return counts of nodes and relationships for the admin panel."""
    cypher = """
    CALL apoc.meta.stats()
    YIELD nodeCount, relCount
    RETURN nodeCount, relCount
    """
    async with get_session() as session:
        try:
            result = await session.run(cypher)
            record = await result.single()
            if record:
                return {"nodes": record["nodeCount"], "relations": record["relCount"]}
        except Exception:
            pass
        # Fallback without APOC
        n_result = await session.run("MATCH (n) RETURN count(n) AS cnt")
        n_record = await n_result.single()
        r_result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        r_record = await r_result.single()
        return {
            "nodes": n_record["cnt"] if n_record else 0,
            "relations": r_record["cnt"] if r_record else 0,
        }
