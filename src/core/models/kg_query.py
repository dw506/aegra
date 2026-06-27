"""Query models for the knowledge graph module."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.kg_enums import EntityStatus


class QueryFilter(BaseModel):
    """Reusable filter object for node and edge queries."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = None
    status: EntityStatus | None = None
    tags: set[str] = Field(default_factory=set)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
