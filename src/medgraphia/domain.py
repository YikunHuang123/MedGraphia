"""
Core domain models shared across the entire pipeline.
All models are immutable Pydantic dataclasses / BaseModel subclasses.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    DISEASE = "Disease"
    DRUG = "Drug"
    SYMPTOM = "Symptom"
    GENE = "Gene"
    PROCEDURE = "Procedure"
    UNKNOWN = "Unknown"


class RelationType(str, Enum):
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    INTERACTS_WITH = "INTERACTS_WITH"
    DOSAGE_FOR = "DOSAGE_FOR"
    SYMPTOM_OF = "SYMPTOM_OF"
    COMPLICATION_OF = "COMPLICATION_OF"
    CODED_AS = "CODED_AS"
    CONTRAINDICATED_IN = "CONTRAINDICATED_IN"
    MENTIONED_IN = "MENTIONED_IN"
    FROM_DOC = "FROM_DOC"


class Language(str, Enum):
    EN = "en"
    ZH = "zh"
    DE = "de"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    CLINICAL_DECISION = "clinical_decision"
    DRUG_INTERACTION = "drug_interaction"
    LITERATURE_MULTIHOP = "literature_multihop"
    CROSS_CORPUS = "cross_corpus"
    PATIENT_FAQ = "patient_faq"


# ---------------------------------------------------------------------------
# Provenance / source metadata
# ---------------------------------------------------------------------------

class SourceMeta(BaseModel):
    """Tracks where a document or chunk came from."""
    source_id: str
    source_title: str = ""
    source_version: str = ""
    source_url: str = ""
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    language: Language = Language.EN


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A medical concept node in the knowledge graph."""
    cui: str                          # UMLS Concept Unique Identifier
    label: str                        # Canonical English label
    entity_type: EntityType = EntityType.UNKNOWN
    lang_labels: dict[str, str] = Field(default_factory=dict)  # {"zh": "心肌梗死", "de": "Myokardinfarkt"}
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------

class Relation(BaseModel):
    """A directed semantic edge between two entities."""
    source_cui: str
    target_cui: str
    relation_type: RelationType
    evidence_text: str = ""
    source_id: str = ""
    chunk_id: str = ""
    confidence: float = 1.0
    extracted_by: str = ""            # model name / version that produced this edge
    properties: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Document & Chunk
# ---------------------------------------------------------------------------

class ParsedSection(BaseModel):
    """A structural section extracted from a parsed document."""
    section_path: str                 # e.g. "4.4 Special warnings > Renal function"
    title: str = ""
    content: str = ""
    page_start: int | None = None
    page_end: int | None = None
    tables: list[dict[str, Any]] = Field(default_factory=list)


class RawDocument(BaseModel):
    """A document as received from a data source, before chunking."""
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    source: SourceMeta
    language: Language = Language.EN
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    sections: list[ParsedSection] = Field(default_factory=list)
    file_path: str | None = None      # local path to the raw file (PDF / XML)
    format: str = "text"              # "pdf" | "xml" | "html" | "text"


class Chunk(BaseModel):
    """A text chunk ready for NER, embedding, and graph insertion."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    source: SourceMeta
    language: Language = Language.EN
    section_path: str = ""
    text: str
    token_count: int | None = None
    page: int | None = None
    char_offset: int | None = None
    entities: list[Entity] = Field(default_factory=list)
    embedding: list[float] | None = None
    sparse_embedding: dict[int, float] | None = None   # token_id → weight


# ---------------------------------------------------------------------------
# Community (Leiden cluster)
# ---------------------------------------------------------------------------

class Community(BaseModel):
    """A Leiden-detected entity cluster with an LLM-generated summary."""
    community_id: str = Field(default_factory=lambda: str(uuid4()))
    members: list[str] = Field(default_factory=list)  # list of CUIs
    summary: str = ""
    size: int = 0
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Session & Message (online query pipeline)
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    citation_number: int
    source_title: str
    source_version: str = ""
    section_path: str = ""
    content_snippet: str = ""
    chunk_id: str = ""


class Message(BaseModel):
    message_id: UUID = Field(default_factory=uuid4)
    session_id: str
    role: str                         # "user" | "assistant"
    content: str
    citations: list[Citation] = Field(default_factory=list)
    model_used: str = ""
    faithfulness_score: float | None = None
    retrieval_paths_used: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "anonymous"
    language: Language = Language.EN
    domain: str = ""
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
