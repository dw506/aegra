"""Shared agent protocol models and base abstractions.

This module defines the common input/output envelope used across the agent
layer. The protocol is intentionally generic and ownership-aware so that each
agent can operate within a clear permission boundary:

- KG / AG / TG / Runtime writes are permission-gated.
- Agents communicate through structured inputs and outputs.
- Concrete business agents are expected to subclass :class:`BaseAgent`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class AgentKind(str, Enum):
    """Top-level categories for all orchestration agents."""

    PERCEPTION = "perception"
    STATE_WRITER = "state_writer"
    GRAPH_PROJECTION = "graph_projection"
    PLANNER = "planner"
    TASK_BUILDER = "task_builder"
    SCHEDULER = "scheduler"
    WORKER = "worker"
    CRITIC = "critic"
    SUPERVISOR = "supervisor"


class GraphScope(str, Enum):
    """Graph or state ownership scopes used by permission checks."""

    KG = "kg"
    AG = "ag"
    TG = "tg"
    RUNTIME = "runtime"


class WritePermission(BaseModel):
    """Declared write permissions for one agent.

    Attributes:
        scopes: Graph scopes that the agent is allowed to touch.
        allow_structural_write: Whether the agent may create, replace or remove
            structural graph elements inside its permitted scopes.
        allow_state_write: Whether the agent may update stateful or runtime-like
            data inside its permitted scopes.
        allow_event_emit: Whether the agent may emit downstream events.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    scopes: list[GraphScope] = Field(
        default_factory=list,
        description="Scopes that this agent is allowed to write to.",
    )
    allow_structural_write: bool = Field(
        default=False,
        description="Whether structural graph mutations are permitted.",
    )
    allow_state_write: bool = Field(
        default=False,
        description="Whether state-level updates are permitted.",
    )
    allow_event_emit: bool = Field(
        default=True,
        description="Whether this agent may emit structured events.",
    )


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
        task_ref: Optional TG task identifier associated with the invocation.
        decision_ref: Optional planner or scheduler decision identifier.
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
        description="Associated TG task identifier, when applicable.",
    )
    decision_ref: str | None = Field(
        default=None,
        description="Associated planner or scheduler decision identifier.",
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
        decisions: Planner, scheduler or critic-style structured decisions.
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
        description="Structured planner, scheduler or critic decisions.",
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


class AgentExecutionResult(BaseModel):
    """Execution envelope returned by :class:`BaseAgent.run`.

    Attributes:
        agent_name: Concrete agent instance name.
        agent_kind: High-level agent category.
        success: Whether execution completed without protocol-level failure.
        output: Structured output emitted by the agent.
        started_at: Invocation start timestamp.
        finished_at: Invocation finish timestamp.
        duration_ms: End-to-end execution duration in milliseconds.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    agent_name: str = Field(
        min_length=1,
        description="Concrete agent instance name.",
    )
    agent_kind: AgentKind = Field(
        description="High-level agent category.",
    )
    success: bool = Field(
        description="Whether the agent run completed successfully.",
    )
    output: AgentOutput = Field(
        description="Structured output emitted by the agent.",
    )
    started_at: datetime = Field(
        description="Invocation start timestamp.",
    )
    finished_at: datetime = Field(
        description="Invocation finish timestamp.",
    )
    duration_ms: int = Field(
        ge=0,
        description="End-to-end execution duration in milliseconds.",
    )


class BaseAgent(ABC):
    """Abstract base class for all orchestration agents.

    Concrete agents should declare their `name`, `kind` and `write_permission`,
    then implement :meth:`execute`. The default :meth:`run` method validates the
    input, executes the agent, enforces the minimal permission rules and returns
    a structured execution result.
    """

    name: str
    kind: AgentKind
    write_permission: WritePermission

    def __init__(self, name: str, kind: AgentKind, write_permission: WritePermission) -> None:
        """Initialize the base agent with explicit identity and permissions."""

        if not name:
            raise ValueError("name must not be empty")
        self.name = name
        self.kind = kind
        self.write_permission = write_permission

    def validate_input(self, agent_input: AgentInput) -> None:
        """Validate the common agent input contract.

        Subclasses may extend this method, but should keep the common checks.
        """

        if not agent_input.context.operation_id:
            raise ValueError("context.operation_id must not be empty")

    def can_write_scope(
        self,
        scope: GraphScope,
        *,
        structural: bool = False,
        state_write: bool = False,
    ) -> bool:
        """Return True when this agent may write the requested scope."""

        if scope not in self.write_permission.scopes:
            return False
        if structural and not self.write_permission.allow_structural_write:
            return False
        if state_write and not self.write_permission.allow_state_write:
            return False
        return True

    def describe_capabilities(self) -> dict[str, Any]:
        """Return a compact machine-readable capability summary."""

        return {
            "name": self.name,
            "kind": self.kind.value,
            "write_permission": self.write_permission.model_dump(mode="json"),
        }

    def run(self, agent_input: AgentInput) -> AgentExecutionResult:
        """Standard execution entrypoint for agent subclasses.

        This method validates input, calls :meth:`execute`, applies minimal
        permission checks and wraps the result with execution metadata.
        """

        started_at = utc_now()
        started_perf = perf_counter()
        success = False
        try:
            self.validate_input(agent_input)
            output = self.execute(agent_input)
            self._validate_output_permissions(output)
            success = len(output.errors) == 0
        except Exception as exc:
            output = AgentOutput(errors=[str(exc)])
            success = False
        finished_at = utc_now()
        duration_ms = max(0, int((perf_counter() - started_perf) * 1000))
        return AgentExecutionResult(
            agent_name=self.name,
            agent_kind=self.kind,
            success=success,
            output=output,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )

    @abstractmethod
    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Execute one agent invocation and return structured output."""

    def _validate_output_permissions(self, output: AgentOutput) -> None:
        """Enforce minimal write-permission checks on agent output."""

        if output.state_deltas:
            self._validate_state_deltas(output.state_deltas)
        if output.emitted_events and not self.write_permission.allow_event_emit:
            raise PermissionError(f"{self.name} is not allowed to emit events")

    def _validate_state_deltas(self, state_deltas: list[dict[str, Any]]) -> None:
        """Validate scoped state deltas against the declared permissions."""

        if not self.write_permission.scopes:
            raise PermissionError(f"{self.name} cannot emit state_deltas without write scopes")

        for delta in state_deltas:
            raw_scope = delta.get("scope")
            if raw_scope is None:
                raise PermissionError("state_deltas must declare a scope")
            scope = raw_scope if isinstance(raw_scope, GraphScope) else GraphScope(str(raw_scope).lower())
            write_type = str(delta.get("write_type", "state")).lower()
            structural = write_type == "structural"
            state_write = write_type == "state"
            if not self.can_write_scope(scope, structural=structural, state_write=state_write):
                raise PermissionError(
                    f"{self.name} is not allowed to emit {write_type} delta for scope {scope.value}"
                )


__all__ = [
    "AgentContext",
    "AgentExecutionResult",
    "AgentInput",
    "AgentKind",
    "AgentOutput",
    "BaseAgent",
    "GraphRef",
    "GraphScope",
    "WritePermission",
    "utc_now",
]
