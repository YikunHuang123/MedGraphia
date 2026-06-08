from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field


class Community(BaseModel):
    """A Leiden-detected entity cluster with an LLM-generated summary."""

    community_id: str = Field(default_factory=lambda: str(uuid4()))
    members: list[str] = Field(default_factory=list)  # list of CUIs
    summary: str = ""
    size: int = 0
    embedding: list[float] | None = None
