"""LLM-owned mission planner for graph-state-driven stage tasks."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ConfigDict, Field

from src.core.stage.models import GraphUpdateIntent, PlannerResult, StageTask


class MissionPlannerResult(PlannerResult):
    """Compatibility wrapper for the new PlannerResult contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    dependencies: list[dict[str, str]] = Field(default_factory=list)
    summary: str = ""

    @property
    def stage_tasks(self) -> list[StageTask]:
        return self.new_stage_tasks


class MissionPlannerAdvisor(Protocol):
    """LLM planner hook. The advisor owns task decomposition and next-task choice."""

    def propose_stage_tasks(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> MissionPlannerResult | PlannerResult | dict[str, Any]:
        """Return a full PlannerResult/MissionPlannerResult from the LLM."""


class MissionPlannerAgent:
    """Planner Agent facade.

    This class intentionally does not contain a deterministic stage sequence or
    keyword-to-stage planner. Its job is context assembly, LLM invocation through
    the advisor, schema validation, and returning a ResultApplier-owned intent.
    """

    def __init__(self, advisor: MissionPlannerAdvisor | None = None) -> None:
        self._advisor = advisor

    def run(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> MissionPlannerResult:
        operation_id = str(graph_context.get("operation_id") or "operation")
        if self._advisor is None:
            return MissionPlannerResult(
                operation_id=operation_id,
                reasoning_summary="PlannerAgent requires an LLM advisor; no hard-coded fallback is available.",
                new_stage_tasks=[],
                selected_next_task=None,
                replan_needed=True,
                stop_condition="planner_llm_unavailable",
                summary="LLM planner unavailable",
                graph_update_intents=[
                    GraphUpdateIntent(
                        target_graph="Runtime",
                        operation="append_evidence",
                        entity_type="planner_blocker",
                        entity_ref=operation_id,
                        payload={"reason": "planner_llm_unavailable", "goal": goal},
                        confidence=1.0,
                        source="planner",
                    )
                ],
                metadata={"planner": "llm_planner_required", "accepted": False},
            )

        raw = self._advisor.propose_stage_tasks(
            goal=goal,
            graph_context=graph_context,
            policy_context=dict(policy_context or {}),
            recent_stage_results=list(recent_stage_results or []),
        )
        result = raw if isinstance(raw, MissionPlannerResult) else MissionPlannerResult.model_validate(raw)
        if result.operation_id == "operation":
            result.operation_id = operation_id
        if not result.summary:
            result.summary = result.reasoning_summary or f"planner proposed {len(result.new_stage_tasks)} stage task(s)"
        return result


__all__ = ["MissionPlannerAgent", "MissionPlannerAdvisor", "MissionPlannerResult"]
