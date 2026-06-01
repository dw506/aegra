"""Deterministic TG/Runtime scheduling function.

This module is a non-agent compatibility service for deterministic scheduling
helpers. It must not be used as the final decision owner for LLM agent flows.
"""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import AgentOutput, GraphRef
from src.core.agents.scheduler_agent import SchedulingContext
from src.core.models.runtime import RuntimeState
from src.core.models.tg import TaskGraph
from src.core.runtime.scheduler import RuntimeScheduler


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
    """Find schedulable TG tasks and emit deterministic service decisions."""

    scheduling_context = (
        context
        if isinstance(context, SchedulingContext)
        else SchedulingContext.model_validate(dict(context or {}))
    )
    del graph_refs, payload, scheduling_context
    if runtime_state is None:
        decisions = [
            {"task_id": task.id, "worker_id": None, "action": "assign", "accepted": True}
            for task in task_graph.find_schedulable_tasks()
        ]
        return SchedulingResult(decisions=decisions, logs=[f"deterministic scheduler selected {len(decisions)} task(s)"])
    tick = RuntimeScheduler().tick(task_graph=task_graph, runtime_state=runtime_state)
    return SchedulingResult(
        decisions=[decision.model_dump(mode="json") for decision in tick.decisions],
        logs=[
            f"deterministic scheduler considered {len(tick.candidate_task_ids)} task(s)",
            f"deterministic scheduler selected {len(tick.selected_task_ids)} task(s)",
        ],
    )
