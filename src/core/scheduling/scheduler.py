"""Deterministic TG/Runtime scheduling function.

This module is the non-agent scheduling boundary for the main execution path.
The legacy ``SchedulerAgent`` remains as a compatibility implementation while
callers migrate away from registering a scheduler in the agent pipeline.
"""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import AgentContext, AgentInput, AgentOutput, GraphRef, GraphScope
from src.core.agents.scheduler_agent import SchedulerAgent, SchedulingContext
from src.core.models.runtime import RuntimeState
from src.core.models.tg import TaskGraph


class SchedulingResult(BaseModel):
    """Canonical output of one deterministic scheduler tick."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    decisions: list[dict[str, Any]] = Field(default_factory=list)
    state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    emitted_events: list[dict[str, Any]] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def accepted_decisions(self) -> list[dict[str, Any]]:
        return [
            decision
            for decision in self.decisions
            if bool(decision.get("accepted")) and str(decision.get("action")) == "assign"
        ]

    def to_agent_output(self) -> AgentOutput:
        return AgentOutput(
            decisions=list(self.decisions),
            state_deltas=list(self.state_deltas),
            emitted_events=list(self.emitted_events),
            logs=list(self.logs),
            errors=list(self.errors),
        )

    @classmethod
    def from_agent_output(cls, output: AgentOutput) -> "SchedulingResult":
        return cls(
            decisions=list(output.decisions),
            state_deltas=list(output.state_deltas),
            emitted_events=list(output.emitted_events),
            logs=list(output.logs),
            errors=list(output.errors),
        )


def schedule_ready_tasks(
    *,
    task_graph: TaskGraph,
    runtime_state: RuntimeState | None,
    context: SchedulingContext | dict[str, Any] | None = None,
    operation_id: str = "operation",
    graph_refs: Sequence[GraphRef] | None = None,
    payload: dict[str, Any] | None = None,
) -> SchedulingResult:
    """Find schedulable TG tasks and emit assignment decisions.

    The implementation intentionally returns scheduler-shaped data instead of
    an agent execution envelope. The legacy ``SchedulerAgent`` is only used as
    a transitional compatibility implementation behind this functional API.
    """

    scheduling_context = (
        context
        if isinstance(context, SchedulingContext)
        else SchedulingContext.model_validate(dict(context or {}))
    )
    raw_payload = {
        **dict(payload or {}),
        "tg_graph": task_graph.to_dict(),
        "scheduling_context": scheduling_context.model_dump(mode="json"),
    }
    if runtime_state is not None:
        raw_payload["runtime_state"] = runtime_state.model_dump(mode="json")
    refs = list(graph_refs or [GraphRef(graph=GraphScope.TG, ref_id="task_graph", ref_type="TaskGraph")])
    scheduler_input = AgentInput(
        graph_refs=refs,
        context=AgentContext(operation_id=operation_id),
        raw_payload=raw_payload,
    )
    output = SchedulerAgent().execute(scheduler_input)
    return SchedulingResult.from_agent_output(output)
