"""Shared agent protocol models.

This module defines the common input/output envelope (`AgentInput`/`AgentOutput`/
`AgentContext`) and the lightweight `GraphRef`/`GraphScope`/`AgentKind` primitives
used across the agent layer. The permission-gated `BaseAgent`/`WritePermission`
abstraction from the multi-agent era has been removed; the live path drives the
single executor and the planner directly rather than through a generic
`BaseAgent.run` dispatch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class AgentKind(str, Enum):
    """Top-level categories for orchestration agents.

    PLANNER/WORKER/STATE_WRITER are produced by the live path; CRITIC/SUPERVISOR
    remain only as decision-history labels (the observer and settings still
    branch on them, though no critic/supervisor advisor is wired).
    """

    STATE_WRITER = "state_writer"
    PLANNER = "planner"
    WORKER = "worker"
    CRITIC = "critic"
    SUPERVISOR = "supervisor"


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


class AgentContext(BaseModel):
    """Execution context shared across one agent invocation.

    Attributes:
        operation_id: Operation identifier for the current orchestration run.
        runtime_state_ref: Optional reference to the Runtime State snapshot or
            store key that the agent should treat as current execution context.
        budget_ref: Optional reference to the runtime budget entry or snapshot.
        checkpoint_ref: Optional reference to the active recovery checkpoint.
        extra: Additional structured context reserved for future extensions.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str = Field(
        min_length=1,
        description="Operation identifier for the current invocation.",
    )
    runtime_state_ref: str | None = Field(
        default=None,
        description="Reference to the runtime state snapshot or store key.",
    )
    budget_ref: str | None = Field(
        default=None,
        description="Reference to the runtime budget state used by this agent.",
    )
    checkpoint_ref: str | None = Field(
        default=None,
        description="Reference to the active checkpoint or recovery anchor.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured context for agent-specific use.",
    )


class AgentInput(BaseModel):
    """Standardized input envelope for one agent invocation.

    Attributes:
        graph_refs: Graph objects or runtime refs that the agent may inspect.
        task_ref: Optional execution task identifier associated with the invocation.
        decision_ref: Optional planner decision identifier.
        context: Shared execution context for this invocation.
        raw_payload: Free-form structured payload reserved for agent-specific
            data that does not justify a dedicated top-level field yet.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    graph_refs: list[GraphRef] = Field(
        default_factory=list,
        description="Graph and runtime references made available to the agent.",
    )
    task_ref: str | None = Field(
        default=None,
        description="Associated execution task identifier, when applicable.",
    )
    decision_ref: str | None = Field(
        default=None,
        description="Associated planner decision identifier.",
    )
    context: AgentContext = Field(
        description="Execution context shared across the invocation.",
    )
    raw_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured agent-specific payload content.",
    )


class AgentOutput(BaseModel):
    """Standardized output payload emitted by one agent.

    Attributes:
        observations: Non-authoritative observations produced by the agent.
        evidence: Evidence references or summaries emitted by the agent.
        outcomes: Structured outcome records intended for downstream consumers.
        decisions: Planner or critic-style structured decisions.
        state_deltas: Proposed state deltas gated by agent permissions.
        replan_requests: Structured replanning requests for downstream owners.
        emitted_events: Structured emitted events, when permitted.
        logs: Human-readable execution log lines.
        errors: Human-readable error lines collected during execution.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    observations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Observed facts or signals not yet committed to a graph.",
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence summaries or references produced by the agent.",
    )
    outcomes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured outcomes returned to downstream components.",
    )
    decisions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured planner or critic decisions.",
    )
    state_deltas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Proposed scoped state or graph changes.",
    )
    replan_requests: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured local or full replanning requests.",
    )
    emitted_events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured events emitted for event-driven flows.",
    )
    logs: list[str] = Field(
        default_factory=list,
        description="Human-readable execution log lines.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Human-readable error messages captured during execution.",
    )


__all__ = [
    "AgentContext",
    "AgentInput",
    "AgentKind",
    "AgentOutput",
    "GraphRef",
    "GraphScope",
    "utc_now",
]
