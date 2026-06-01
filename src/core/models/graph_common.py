"""Shared graph primitives used by AG and attack-process models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def stable_node_id(prefix: str, payload: dict[str, Any]) -> str:
    """Build a deterministic ID from a small structured payload."""

    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}::{digest}"


class GraphRef(BaseModel):
    """Reference to a source object in KG, AG, TG or a derived query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph: Literal["kg", "ag", "tg", "query"]
    ref_id: str = Field(min_length=1)
    ref_type: str | None = None
    label: str | None = None

    def key(self) -> str:
        """Return a stable key for indexing."""

        return f"{self.graph}:{self.ref_id}"
