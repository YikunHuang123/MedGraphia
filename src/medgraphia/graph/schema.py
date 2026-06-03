"""
Neo4j schema definitions: node labels, relationship types, property constraints,
and index creation statements.  Run apply_schema() once on a fresh database.
"""
from __future__ import annotations

from medgraphia.graph.client import get_session
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------
NODE_LABELS = [
    "Disease",
    "Drug",
    "Symptom",
    "Gene",
    "Procedure",
    "Chunk",
    "Document",
    "Community",
]

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
    "MENTIONED_IN",   # Entity  → Chunk
    "FROM_DOC",       # Chunk   → Document
    "MEMBER_OF",      # Entity  → Community
]

# ---------------------------------------------------------------------------
# DDL statements (Cypher)
# Each tuple: (description, cypher)
# ---------------------------------------------------------------------------
_CONSTRAINTS: list[tuple[str, str]] = [
    # Unique CUI per medical entity type
    ("unique_disease_cui",    "CREATE CONSTRAINT unique_disease_cui IF NOT EXISTS FOR (n:Disease)   REQUIRE n.cui IS UNIQUE"),
    ("unique_drug_cui",       "CREATE CONSTRAINT unique_drug_cui IF NOT EXISTS FOR (n:Drug)         REQUIRE n.cui IS UNIQUE"),
    ("unique_symptom_cui",    "CREATE CONSTRAINT unique_symptom_cui IF NOT EXISTS FOR (n:Symptom)   REQUIRE n.cui IS UNIQUE"),
    ("unique_gene_cui",       "CREATE CONSTRAINT unique_gene_cui IF NOT EXISTS FOR (n:Gene)         REQUIRE n.cui IS UNIQUE"),
    ("unique_procedure_cui",  "CREATE CONSTRAINT unique_procedure_cui IF NOT EXISTS FOR (n:Procedure) REQUIRE n.cui IS UNIQUE"),
    # Chunk and Document use UUID-based IDs
    ("unique_chunk_id",       "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (n:Chunk)         REQUIRE n.chunk_id IS UNIQUE"),
    ("unique_doc_id",         "CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (n:Document)        REQUIRE n.doc_id IS UNIQUE"),
    ("unique_community_id",   "CREATE CONSTRAINT unique_community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.community_id IS UNIQUE"),
    # Admin / Auth persistence
    ("unique_api_key_hash",   "CREATE CONSTRAINT unique_api_key_hash IF NOT EXISTS FOR (n:ApiKey)    REQUIRE n.key_hash IS UNIQUE"),
    ("unique_pipeline_dom",   "CREATE CONSTRAINT unique_pipeline_dom IF NOT EXISTS FOR (n:PipelineStatus) REQUIRE n.domain IS UNIQUE"),
]

_INDEXES: list[tuple[str, str]] = [
    # Full-text search index on entity labels (enables BM25-style lexical lookup)
    ("idx_disease_label",    "CREATE INDEX idx_disease_label IF NOT EXISTS FOR (n:Disease)   ON (n.label)"),
    ("idx_disease_zh",       "CREATE INDEX idx_disease_zh    IF NOT EXISTS FOR (n:Disease)   ON (n.lang_zh)"),
    ("idx_disease_de",       "CREATE INDEX idx_disease_de    IF NOT EXISTS FOR (n:Disease)   ON (n.lang_de)"),

    ("idx_drug_label",       "CREATE INDEX idx_drug_label    IF NOT EXISTS FOR (n:Drug)      ON (n.label)"),
    ("idx_drug_zh",          "CREATE INDEX idx_drug_zh       IF NOT EXISTS FOR (n:Drug)      ON (n.lang_zh)"),
    ("idx_drug_de",          "CREATE INDEX idx_drug_de       IF NOT EXISTS FOR (n:Drug)      ON (n.lang_de)"),

    ("idx_symptom_label",    "CREATE INDEX idx_symptom_label IF NOT EXISTS FOR (n:Symptom)   ON (n.label)"),
    ("idx_symptom_zh",       "CREATE INDEX idx_symptom_zh    IF NOT EXISTS FOR (n:Symptom)   ON (n.lang_zh)"),
    ("idx_symptom_de",       "CREATE INDEX idx_symptom_de    IF NOT EXISTS FOR (n:Symptom)   ON (n.lang_de)"),

    ("idx_gene_label",       "CREATE INDEX idx_gene_label    IF NOT EXISTS FOR (n:Gene)      ON (n.label)"),
    ("idx_gene_zh",          "CREATE INDEX idx_gene_zh       IF NOT EXISTS FOR (n:Gene)      ON (n.lang_zh)"),
    ("idx_gene_de",          "CREATE INDEX idx_gene_de       IF NOT EXISTS FOR (n:Gene)      ON (n.lang_de)"),

    ("idx_proc_label",       "CREATE INDEX idx_proc_label    IF NOT EXISTS FOR (n:Procedure) ON (n.label)"),
    ("idx_proc_zh",          "CREATE INDEX idx_proc_zh       IF NOT EXISTS FOR (n:Procedure) ON (n.lang_zh)"),
    ("idx_proc_de",          "CREATE INDEX idx_proc_de       IF NOT EXISTS FOR (n:Procedure) ON (n.lang_de)"),

    ("idx_chunk_section",    "CREATE INDEX idx_chunk_section IF NOT EXISTS FOR (n:Chunk)     ON (n.section_path)"),
    ("idx_chunk_doc_id",     "CREATE INDEX idx_chunk_doc_id IF NOT EXISTS FOR (n:Chunk)      ON (n.doc_id)"),
    ("idx_doc_source_id",    "CREATE INDEX idx_doc_source_id IF NOT EXISTS FOR (n:Document)  ON (n.source_id)"),
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
