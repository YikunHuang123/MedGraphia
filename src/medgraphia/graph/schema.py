"""
Neo4j schema definitions: node labels, relationship types, property constraints,
and index creation statements.  Run apply_schema() once on a fresh database.
"""

from __future__ import annotations

from medgraphia.domain import EntityType
from medgraphia.graph.client import get_session
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------
ENTITY_LABELS = [t.value for t in EntityType]
NODE_LABELS = ENTITY_LABELS + ["Chunk", "Document", "Community"]

# ---------------------------------------------------------------------------
# Relationship types (closed schema — no arbitrary types allowed)
# ---------------------------------------------------------------------------
RELATION_TYPES = [
    "TREATS",
    "CAUSES",
    "INTERACTS_WITH",
    "DOSAGE_FOR",
    "SYMPTOM_OF",
    "COMPLICATION_OF",
    "CODED_AS",
    "CONTRAINDICATED_IN",
    "MENTIONED_IN",  # Entity  → Chunk
    "FROM_DOC",  # Chunk   → Document
    "MEMBER_OF",  # Entity  → Community
]

# ---------------------------------------------------------------------------
# DDL statements (Cypher)
# Each tuple: (description, cypher)
# ---------------------------------------------------------------------------
_CONSTRAINTS: list[tuple[str, str]] = [
    # Unique CUI per medical entity type, generated from EntityType so every
    # category gets a constraint without hand-writing one per type.
    *[
        (
            f"unique_{label.lower()}_cui",
            f"CREATE CONSTRAINT unique_{label.lower()}_cui IF NOT EXISTS FOR (n:{label}) REQUIRE n.cui IS UNIQUE",
        )
        for label in ENTITY_LABELS
    ],
    # Chunk and Document use UUID-based IDs
    (
        "unique_chunk_id",
        "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (n:Chunk)         REQUIRE n.chunk_id IS UNIQUE",
    ),
    (
        "unique_doc_id",
        "CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (n:Document)        REQUIRE n.doc_id IS UNIQUE",
    ),
    (
        "unique_community_id",
        "CREATE CONSTRAINT unique_community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.community_id IS UNIQUE",
    ),
    # Admin / Auth persistence
    (
        "unique_api_key_hash",
        "CREATE CONSTRAINT unique_api_key_hash IF NOT EXISTS FOR (n:ApiKey)    REQUIRE n.key_hash IS UNIQUE",
    ),
    (
        "unique_pipeline_dom",
        "CREATE CONSTRAINT unique_pipeline_dom IF NOT EXISTS FOR (n:PipelineStatus) REQUIRE n.domain IS UNIQUE",
    ),
]

_INDEXES: list[tuple[str, str]] = [
    # Full-text search index on entity labels (enables BM25-style lexical lookup),
    # generated from EntityType — one (label, lang_zh, lang_de) index triple per type.
    *[
        (
            f"idx_{label.lower()}_{prop}",
            f"CREATE INDEX idx_{label.lower()}_{prop} IF NOT EXISTS FOR (n:{label}) ON (n.{col})",
        )
        for label in ENTITY_LABELS
        for prop, col in (("label", "label"), ("zh", "lang_zh"), ("de", "lang_de"))
    ],
    (
        "idx_chunk_section",
        "CREATE INDEX idx_chunk_section IF NOT EXISTS FOR (n:Chunk)     ON (n.section_path)",
    ),
    (
        "idx_chunk_doc_id",
        "CREATE INDEX idx_chunk_doc_id IF NOT EXISTS FOR (n:Chunk)      ON (n.doc_id)",
    ),
    (
        "idx_doc_source_id",
        "CREATE INDEX idx_doc_source_id IF NOT EXISTS FOR (n:Document)  ON (n.source_id)",
    ),
]


async def apply_schema() -> None:
    """Apply all constraints and indexes to Neo4j (idempotent via IF NOT EXISTS)."""
    async with get_session() as session:
        for name, cypher in _CONSTRAINTS:
            try:
                await session.run(cypher)
                logger.debug("schema_constraint_applied", name=name)
            except Exception as exc:
                logger.warning("schema_constraint_skipped", name=name, error=str(exc))

        for name, cypher in _INDEXES:
            try:
                await session.run(cypher)
                logger.debug("schema_index_applied", name=name)
            except Exception as exc:
                logger.warning("schema_index_skipped", name=name, error=str(exc))

    logger.info("neo4j_schema_ready")
