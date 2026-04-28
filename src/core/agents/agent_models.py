"""Shared record models for the agent layer.

These models provide a common exchange format for perception, planner, worker,
critic and other agents so the system does not need separate ad-hoc payload
shapes for each component.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import GraphRef, GraphScope


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def new_record_id(prefix: str) -> str:
    """Return a compact identifier for one shared agent record."""

    return f"{prefix}-{uuid4().hex}"


class BaseAgentRecord(BaseModel):
    """Base class for all shared agent exchange records."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(
        min_length=1,
        description="Stable or generated identifier for this record.",
    )
    source_agent: str = Field(
        min_length=1,
        description="Concrete agent name that emitted this record.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        description="Creation timestamp of the record.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emitter confidence in the correctness or usefulness of the record.",
    )
    summary: str = Field(
        min_length=1,
        description="Human-readable summary of the record.",
    )
    refs: list[GraphRef] = Field(
        default_factory=list,
        description="Graph or runtime references associated with the record.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured machine-readable payload carried by the record.",
    )

    def to_agent_output_fragment(self) -> dict[str, Any]:
        """Return a JSON-safe fragment suitable for `AgentOutput` lists."""

        return self.model_dump(mode="json")


class ObservationRecord(BaseAgentRecord):
    """Non-authoritative observation emitted by an agent."""

    id: str = Field(
        default_factory=lambda: new_record_id("obs"),
        description="Unique identifier for the observation record.",
    )


class EvidenceRecord(BaseAgentRecord):
    """Evidence summary or pointer emitted by an agent."""

    id: str = Field(
        default_factory=lambda: new_record_id("evidence"),
        description="Unique identifier for the evidence record.",
    )
    payload_ref: str | None = Field(
        default=None,
        description="Optional external reference to the raw evidence payload.",
    )


class OutcomeRecord(BaseAgentRecord):
    """Structured outcome generated from task or decision execution."""

    id: str = Field(
        default_factory=lambda: new_record_id("outcome"),
        description="Unique identifier for the outcome record.",
    )
    task_id: str = Field(
        min_length=1,
        description="Task identifier associated with this outcome.",
    )
    outcome_type: str = Field(
        min_length=1,
        description="Outcome category such as execution_result or validation_result.",
    )
    success: bool = Field(
        description="Whether the underlying operation completed successfully.",
    )
    raw_result_ref: str | None = Field(
        default=None,
        description="Optional reference to the raw underlying execution result.",
    )


class DecisionRecord(BaseAgentRecord):
    """Structured decision emitted by planner, scheduler or critic-style agents."""

    id: str = Field(
        default_factory=lambda: new_record_id("decision"),
        description="Unique identifier for the decision record.",
    )
    decision_type: str = Field(
        min_length=1,
        description="Decision category such as schedule, prune or prioritize.",
    )
    score: float = Field(
        default=0.0,
        description="Decision score or ranking signal.",
    )
    target_refs: list[GraphRef] = Field(
        default_factory=list,
        description="Primary target references of this decision.",
    )
    rationale: str = Field(
        min_length=1,
        description="Human-readable rationale for the decision.",
    )


class StateDeltaRecord(BaseAgentRecord):
    """Scoped graph or runtime delta proposed by an agent."""

    id: str = Field(
        default_factory=lambda: new_record_id("delta"),
        description="Unique identifier for the state delta record.",
    )
    graph_scope: GraphScope = Field(
        description="Scope that owns the target of the proposed delta.",
    )
    delta_type: str = Field(
        min_length=1,
        description="Delta type such as upsert, patch, supersede or expire.",
    )
    target_ref: GraphRef = Field(
        description="Primary target reference of the proposed delta.",
    )
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured patch payload proposed by the agent.",
    )

    def to_agent_output_fragment(self) -> dict[str, Any]:
        """Return a JSON-safe state delta fragment with explicit scope fields."""

        data = self.model_dump(mode="json")
        data["scope"] = self.graph_scope.value
        data.setdefault("write_type", "state")
        return data


class ReplanRequestRecord(BaseAgentRecord):
    """Structured replanning request emitted by agents."""

    id: str = Field(
        default_factory=lambda: new_record_id("replan"),
        description="Unique identifier for the replan request record.",
    )
    trigger_task_id: str = Field(
        min_length=1,
        description="Task whose outcome or state triggered replanning.",
    )
    reason: str = Field(
        min_length=1,
        description="Reason that replanning is requested.",
    )
    affected_refs: list[GraphRef] = Field(
        default_factory=list,
        description="Graph or runtime references impacted by the requested replan.",
    )
    severity: str = Field(
        min_length=1,
        description="Severity hint such as low, medium or high.",
    )


__all__ = [
    "DecisionRecord",
    "EvidenceRecord",
    "ObservationRecord",
    "OutcomeRecord",
    "ReplanRequestRecord",
    "StateDeltaRecord",
    "new_record_id",
    "utc_now",
]
