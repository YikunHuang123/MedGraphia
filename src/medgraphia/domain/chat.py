from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from medgraphia.domain.base import Language


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


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
    role: Role  # Changed from str to Role Enum
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
    title: str = "New Session"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    messages: list[Message] = Field(default_factory=list)
