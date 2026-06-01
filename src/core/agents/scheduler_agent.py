"""LLM-owned SchedulerAgent for TG task dispatch decisions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import (
    AgentContext,
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphScope,
    WritePermission,
)
from src.core.scheduling.candidate_task_service import CandidateTaskService, RuntimeConstraintService
from src.core.scheduling.llm_scheduler_advisor import LLMSchedulerAdvisor
from src.core.scheduling.llm_scheduler_models import ScheduleDecision


class SchedulingContext(BaseModel):
    """Compatibility envelope for scheduler prompt hints."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_assignments: int = Field(default=1, ge=1, le=10)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    available_workers: list[dict[str, Any]] = Field(default_factory=list)
    session_summary: dict[str, Any] = Field(default_factory=dict)
    lock_summary: dict[str, Any] = Field(default_factory=dict)
    budget_summary: dict[str, Any] = Field(default_factory=dict)
    policy_flags: dict[str, Any] = Field(default_factory=dict)


class SchedulerAgent(BaseAgent):
    """Choose the next existing TG task by delegating final judgment to an LLM."""

    def __init__(
        self,
        name: str = "scheduler_agent",
        *,
        advisor: LLMSchedulerAdvisor | None = None,
        candidate_service: CandidateTaskService | None = None,
        runtime_service: RuntimeConstraintService | None = None,
    ) -> None:
        self._advisor = advisor
        self._candidate_service = candidate_service or CandidateTaskService()
        self._runtime_service = runtime_service or RuntimeConstraintService()
        self.last_decision: ScheduleDecision | None = None
        super().__init__(
            name=name,
            kind=AgentKind.SCHEDULER,
            write_permission=WritePermission(
                scopes=[],
                allow_structural_write=False,
                allow_state_write=False,
                allow_event_emit=False,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        super().validate_input(agent_input)

    def run_decision(
        self,
        *,
        operation_id: str,
        graph_context: dict[str, Any],
        candidate_tasks: list[dict[str, Any]],
        runtime_summary: dict[str, Any],
        policy_context: dict[str, Any],
        tool_catalog: dict[str, Any],
        recent_outcomes: list[dict[str, Any]],
    ) -> ScheduleDecision:
        """Return one LLM-owned ScheduleDecision with no deterministic fallback."""

        if self._advisor is None:
            decision = ScheduleDecision(
                decision="blocked",
                task_id=None,
                worker_id=None,
                rationale="scheduler_llm_unavailable",
                metadata={"accepted": False, "reason": "scheduler_llm_unavailable"},
            )
            self.last_decision = decision
            return decision
        decision = self._advisor.choose_next_task(
            operation_id=operation_id,
            graph_context=graph_context,
            candidate_tasks=candidate_tasks,
            runtime_summary=runtime_summary,
            policy_context=policy_context,
            tool_catalog=tool_catalog,
            recent_outcomes=recent_outcomes,
        )
        self.last_decision = decision
        return decision

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        payload = dict(agent_input.raw_payload)
        operation_id = agent_input.context.operation_id
        candidate_tasks = [
            dict(item) for item in payload.get("candidate_tasks", []) if isinstance(item, dict)
        ]
        if not candidate_tasks and isinstance(payload.get("tg_graph"), dict):
            from src.core.models.runtime import RuntimeState
            from src.core.models.tg import TaskGraph

            runtime_state = None
            if isinstance(payload.get("runtime_state"), dict):
                runtime_state = RuntimeState.model_validate(payload["runtime_state"])
            candidate_tasks = self._candidate_service.collect(
                TaskGraph.from_dict(payload["tg_graph"]),
                runtime_state,
            )
        graph_context = dict(payload.get("graph_context") or {})
        if not graph_context:
            graph_context = {
                "operation_id": operation_id,
                "tg_graph": payload.get("tg_graph", {}),
                "context_source": "compatibility_scheduler_payload",
            }
        decision = self.run_decision(
            operation_id=operation_id,
            graph_context=graph_context,
            candidate_tasks=candidate_tasks,
            runtime_summary=dict(payload.get("runtime_summary") or {}),
            policy_context=dict(payload.get("policy_context") or {}),
            tool_catalog=dict(payload.get("tool_catalog") or {}),
            recent_outcomes=[
                dict(item) for item in payload.get("recent_outcomes", []) if isinstance(item, dict)
            ],
        )
        accepted = decision.decision == "dispatch" and bool(decision.metadata.get("accepted", True))
        return AgentOutput(
            decisions=[
                {
                    "id": f"schedule-{operation_id}-{decision.task_id or decision.decision}",
                    "accepted": accepted,
                    "action": decision.decision,
                    "task_id": decision.task_id,
                    "worker_id": decision.worker_id,
                    "rationale": decision.rationale,
                    "confidence": decision.confidence,
                    "payload": {
                        "task_type": decision.scheduled_task.stage_type
                        if decision.scheduled_task is not None
                        else None,
                    },
                    "schedule_decision": decision.model_dump(mode="json"),
                }
            ],
            logs=[f"SchedulerAgent decision={decision.decision}: {decision.rationale}"],
            errors=[],
        )

    def build_input(
        self,
        *,
        operation_id: str,
        graph_context: dict[str, Any],
        candidate_tasks: list[dict[str, Any]],
        runtime_summary: dict[str, Any],
        policy_context: dict[str, Any],
        tool_catalog: dict[str, Any],
        recent_outcomes: list[dict[str, Any]],
    ) -> AgentInput:
        return AgentInput(
            context=AgentContext(operation_id=operation_id),
            raw_payload={
                "graph_context": graph_context,
                "candidate_tasks": candidate_tasks,
                "runtime_summary": runtime_summary,
                "policy_context": policy_context,
                "tool_catalog": tool_catalog,
                "recent_outcomes": recent_outcomes,
            },
        )


__all__ = ["SchedulerAgent", "SchedulingContext"]
