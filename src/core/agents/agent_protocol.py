"""Shared graph-reference primitives.

This module defines the lightweight `GraphRef`/`GraphScope` primitives used by
the API and orchestrator. The multi-agent input/output envelope
(`AgentInput`/`AgentOutput`/`AgentContext`/`AgentKind`) and the permission-gated
`BaseAgent` abstraction from the multi-agent era have been removed; the live path
drives the single executor and the planner directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class GraphScope(str, Enum):
    """Graph or state ownership scopes."""

    KG = "kg"
    AG = "ag"
    RUNTIME = "runtime"


class GraphRef(BaseModel):
    """Lightweight reference to a graph entity or runtime object.

    Attributes:
        graph: Target graph or state namespace for this reference.
        ref_id: Stable identifier of the referenced object.
        ref_type: Optional type name for consumers that need stronger typing.
        metadata: Extra structured context about the reference.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    graph: GraphScope = Field(
        description="Graph or state namespace that owns the referenced object."
    )
    ref_id: str = Field(
        min_length=1,
        description="Stable identifier of the referenced object.",
    )
    ref_type: str | None = Field(
        default=None,
        description="Optional object type associated with the reference.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata attached to the reference.",
    )


__all__ = [
    "GraphRef",
    "GraphScope",
    "utc_now",
]
