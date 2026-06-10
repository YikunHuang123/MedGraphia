from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from medgraphia.domain.base import Language

if TYPE_CHECKING:
    from medgraphia.domain.medical import Entity


class SourceMeta(BaseModel):
    """Tracks where a document or chunk came from."""

    source_id: str
    source_title: str = ""
    source_version: str = ""
    source_url: str = ""
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    language: Language = Language.EN


class ParsedSection(BaseModel):
    """A structural section extracted from a parsed document."""

    section_path: str  # e.g. "4.4 Special warnings > Renal function"
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
    file_path: str | None = None  # local path to the raw file (PDF / XML)
    format: str = "text"  # "pdf" | "xml" | "html" | "text"


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
    parent_text: str = ""  # full section text; set by chunker for parent-child retrieval
    entities: list[Entity] = Field(default_factory=list)
    embedding: list[float] | None = None
    sparse_embedding: dict[int, float] | None = None  # token_id → weight
