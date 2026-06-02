from __future__ import annotations
from pydantic import BaseModel, Field
from uuid import uuid4

class Community(BaseModel):
    """A Leiden-detected entity cluster with an LLM-generated summary."""
    community_id: str = Field(default_factory=lambda: str(uuid4()))
    members: list[str] = Field(default_factory=list)  # list of CUIs
    summary: str = ""
    size: int = 0
    embedding: list[float] | None = None
