"""Capability model for stage-level graph-driven operations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


CapabilityType = Literal[
    "network_reachability",
    "credential",
    "authenticated_context",
    "session",
    "privilege_context",
    "pivot_route",
    "internal_service_access",
    "goal_access",
]


class Capability(BaseModel):
    """Execution capability gained or required by a stage result."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    capability_id: str = Field(min_length=1)
    capability_type: CapabilityType
    source_task_id: str = Field(min_length=1)
    host_id: str | None = None
    service_id: str | None = None
    identity: str | None = None
    runtime_ref: str | None = None
    enables: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["Capability", "CapabilityType"]
