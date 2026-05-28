"""Mission planner that proposes graph-driven stage tasks."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import GraphRef, stable_node_id
from src.core.stage.models import StageTask, StageType


STAGE_PLAN: tuple[tuple[StageType, str, list[str], int], ...] = (
    (
        StageType.RECON_STAGE,
        "Discover reachable hosts, services, fingerprints and entry points.",
        ["host/service/endpoint facts produced", "candidate vuln analysis tasks identified"],
        90,
    ),
    (
        StageType.VULN_ANALYSIS_STAGE,
        "Analyze discovered services and paths for viable vulnerability hypotheses.",
        ["vulnerability hypotheses produced", "exploit candidates ranked"],
        80,
    ),
    (
        StageType.EXPLOIT_STAGE,
        "Use validated exploit candidates to establish an access capability.",
        ["access capability or failure evidence produced"],
        70,
    ),
    (
        StageType.ACCESS_PIVOT_STAGE,
        "Validate access, credentials, sessions and pivot routes for downstream reachability.",
        ["session/credential/pivot context validated"],
        60,
    ),
    (
        StageType.GOAL_STAGE,
        "Verify the mission goal and produce final evidence.",
        ["goal satisfied evidence or final blocker produced"],
        50,
    ),
)


class MissionPlannerResult(BaseModel):
    """Planner output containing stage task proposals and dependencies."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str
    stage_tasks: list[StageTask] = Field(default_factory=list)
    dependencies: list[dict[str, str]] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionPlannerAdvisor(Protocol):
    """Optional LLM advisor hook for mission planning."""

    def propose_stage_tasks(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
    ) -> MissionPlannerResult | dict[str, Any]:
        """Return stage tasks for the current mission goal."""


class MissionPlannerAgent:
    """Translate a mission goal into coarse stage-level TG tasks."""

    def __init__(self, advisor: MissionPlannerAdvisor | None = None) -> None:
        self._advisor = advisor

    def run(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
    ) -> MissionPlannerResult:
        operation_id = str(graph_context.get("operation_id") or "operation")
        explicit = self._explicit_plan(operation_id=operation_id, graph_context=graph_context)
        if explicit is not None:
            return explicit
        if self._advisor is not None:
            raw = self._advisor.propose_stage_tasks(
                goal=goal,
                graph_context=graph_context,
                policy_context=dict(policy_context or {}),
            )
            advised = raw if isinstance(raw, MissionPlannerResult) else MissionPlannerResult.model_validate(raw)
            if advised.stage_tasks:
                return advised
            fallback = self._heuristic_plan(operation_id=operation_id, goal=goal, graph_context=graph_context)
            fallback.metadata = {
                **dict(fallback.metadata),
                "advisor_fallback": advised.model_dump(mode="json"),
            }
            return fallback
        return self._heuristic_plan(operation_id=operation_id, goal=goal, graph_context=graph_context)

    def _explicit_plan(self, *, operation_id: str, graph_context: dict[str, Any]) -> MissionPlannerResult | None:
        extra = graph_context.get("extra")
        if not isinstance(extra, dict):
            return None
        raw_tasks = extra.get("stage_tasks")
        if not isinstance(raw_tasks, list):
            return None
        stage_tasks: list[StageTask] = []
        for index, item in enumerate(raw_tasks, start=1):
            if isinstance(item, StageTask):
                stage_tasks.append(item)
            elif isinstance(item, dict):
                task = self._stage_task_from_payload(item, operation_id=operation_id, index=index)
                if task is not None:
                    stage_tasks.append(task)
        if not stage_tasks:
            return None
        dependencies = extra.get("stage_dependencies")
        return MissionPlannerResult(
            operation_id=operation_id,
            stage_tasks=stage_tasks,
            dependencies=[dict(item) for item in dependencies if isinstance(item, dict)] if isinstance(dependencies, list) else [],
            summary=f"mission planner accepted {len(stage_tasks)} explicit stage task proposal(s)",
            metadata={"planner": "explicit_stage_tasks"},
        )

    def _heuristic_plan(self, *, operation_id: str, goal: str, graph_context: dict[str, Any]) -> MissionPlannerResult:
        target_refs = self._target_refs(graph_context)
        stage_tasks: list[StageTask] = []
        dependencies: list[dict[str, str]] = []
        previous_id: str | None = None
        for stage_type, default_objective, success_criteria, priority in STAGE_PLAN:
            objective = goal if stage_type == StageType.GOAL_STAGE and goal else default_objective
            task_id = stable_node_id(
                "stage-task",
                {
                    "operation_id": operation_id,
                    "stage_type": stage_type.value,
                    "goal": goal,
                    "target_refs": [ref.key() for ref in target_refs],
                },
            )
            stage_tasks.append(
                StageTask(
                    task_id=task_id,
                    stage_type=stage_type,
                    objective=objective,
                    target_refs=target_refs,
                    required_context={"mission_goal": goal},
                    success_criteria=list(success_criteria),
                    max_steps=8,
                    risk_level="medium",
                    priority=priority,
                )
            )
            if previous_id is not None:
                dependencies.append({"source": previous_id, "target": task_id})
            previous_id = task_id
        return MissionPlannerResult(
            operation_id=operation_id,
            stage_tasks=stage_tasks,
            dependencies=dependencies,
            summary="heuristic mission planner generated five stage tasks",
            metadata={"planner": "heuristic_stage_chain"},
        )

    @staticmethod
    def _target_refs(graph_context: dict[str, Any]) -> list[GraphRef]:
        raw_refs = graph_context.get("target_refs")
        refs: list[GraphRef] = []
        if isinstance(raw_refs, list):
            for item in raw_refs:
                if isinstance(item, GraphRef):
                    refs.append(item)
                elif isinstance(item, dict):
                    try:
                        refs.append(GraphRef.model_validate(item))
                    except Exception:
                        continue
        return refs

    @staticmethod
    def _stage_task_from_payload(payload: dict[str, Any], *, operation_id: str, index: int) -> StageTask | None:
        raw_stage_type = payload.get("stage_type") or payload.get("task_type")
        if raw_stage_type is None:
            return None
        try:
            stage_type = StageType(str(raw_stage_type))
        except ValueError:
            return None
        task_id = str(
            payload.get("task_id")
            or stable_node_id(
                "stage-task",
                {
                    "operation_id": operation_id,
                    "index": index,
                    "stage_type": stage_type.value,
                    "objective": payload.get("objective") or payload.get("label"),
                },
            )
        )
        refs: list[GraphRef] = []
        for item in payload.get("target_refs", []) if isinstance(payload.get("target_refs"), list) else []:
            if isinstance(item, GraphRef):
                refs.append(item)
            elif isinstance(item, dict):
                try:
                    refs.append(GraphRef.model_validate(item))
                except Exception:
                    continue
        return StageTask(
            task_id=task_id,
            stage_type=stage_type,
            objective=str(payload.get("objective") or payload.get("label") or stage_type.value),
            target_refs=refs,
            required_context=dict(payload.get("required_context") or {}),
            success_criteria=[str(item) for item in payload.get("success_criteria", []) if item is not None],
            max_steps=int(payload.get("max_steps") or 8),
            risk_level=str(payload.get("risk_level") or "medium"),  # type: ignore[arg-type]
            priority=int(payload.get("priority") or 50),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = ["MissionPlannerAgent", "MissionPlannerAdvisor", "MissionPlannerResult", "STAGE_PLAN"]
