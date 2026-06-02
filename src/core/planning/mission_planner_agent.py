"""LLM-owned mission planner for graph-state-driven decisions."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from src.core.planning.models import PlannerDecision


class MissionPlannerAdvisor(Protocol):
    """LLM planner hook. The advisor owns next-decision choice."""

    def propose_next_decision(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> PlannerDecision | dict[str, Any]:
        """Return a PlannerDecision from the LLM."""


class MissionPlannerAgent:
    """Planner Agent facade.

    This class intentionally does not contain a deterministic stage sequence or
    keyword-to-stage planner. Its job is context assembly, LLM invocation through
    the advisor, schema validation, and returning the PlannerAgent-owned decision.
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
    ) -> PlannerDecision:
        operation_id = str(graph_context.get("operation_id") or "operation")
        cycle_index = int(graph_context.get("cycle_index") or graph_context.get("runtime", {}).get("cycle_index") or 0)
        if self._advisor is None:
            return PlannerDecision(
                operation_id=operation_id,
                cycle_index=cycle_index,
                decision="replan",
                selected_agent=None,
                selected_stage=None,
                objective=goal,
                target_refs=[],
                required_context={},
                success_criteria=[],
                risk_level="medium",
                max_steps=1,
                reasoning_summary="PlannerAgent requires an LLM advisor; no hard-coded fallback is available.",
                handoff_acceptance=None,
                stop_condition="planner_llm_unavailable",
                confidence=0.0,
                metadata={"planner": "llm_planner_required", "accepted": False},
            )

        raw = self._advisor.propose_next_decision(
            goal=goal,
            graph_context=graph_context,
            policy_context=dict(policy_context or {}),
            recent_stage_results=list(recent_stage_results or []),
        )
        result = raw if isinstance(raw, PlannerDecision) else _decision_from_payload(
            raw,
            operation_id=operation_id,
            cycle_index=cycle_index,
            goal=goal,
        )
        if result.operation_id == "operation":
            result.operation_id = operation_id
        return result


def _decision_from_payload(
    payload: Any,
    *,
    operation_id: str,
    cycle_index: int,
    goal: str,
) -> PlannerDecision:
    if isinstance(payload, PlannerDecision):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("planner advisor must return PlannerDecision or dict")
    try:
        return PlannerDecision.model_validate(payload)
    except ValidationError:
        converted = _legacy_payload_to_decision_payload(payload, operation_id=operation_id, cycle_index=cycle_index, goal=goal)
        return PlannerDecision.model_validate(converted)


def _legacy_payload_to_decision_payload(
    payload: dict[str, Any],
    *,
    operation_id: str,
    cycle_index: int,
    goal: str,
) -> dict[str, Any]:
    selected_task = payload.get("selected_next_task")
    selected_fields = _legacy_selected_task_fields(selected_task)
    if selected_fields.get("selected_agent") and selected_fields.get("selected_stage"):
        decision = "dispatch_agent"
    elif payload.get("stop_condition"):
        decision = "stop_failed" if payload.get("replan_needed") else "stop_success"
    else:
        decision = "replan"

    return {
        "operation_id": payload.get("operation_id") or operation_id,
        "cycle_index": payload.get("cycle_index") or cycle_index,
        "decision": decision,
        "selected_agent": selected_fields.get("selected_agent") if decision == "dispatch_agent" else None,
        "selected_stage": selected_fields.get("selected_stage") if decision == "dispatch_agent" else None,
        "objective": selected_fields.get("objective") or payload.get("objective") or goal,
        "target_refs": selected_fields.get("target_refs") or payload.get("target_refs") or [],
        "required_context": selected_fields.get("required_context") or payload.get("required_context") or {},
        "success_criteria": selected_fields.get("success_criteria") or payload.get("success_criteria") or [],
        "risk_level": selected_fields.get("risk_level") or payload.get("risk_level") or "medium",
        "max_steps": selected_fields.get("max_steps") or payload.get("max_steps") or 3,
        "reasoning_summary": payload.get("reasoning_summary") or payload.get("summary") or "",
        "handoff_acceptance": payload.get("handoff_acceptance"),
        "stop_condition": payload.get("stop_condition"),
        "confidence": payload.get("confidence") if payload.get("confidence") is not None else 0.5,
        "metadata": payload.get("metadata") or {"legacy_planner_payload": True},
    }


def _legacy_selected_task_fields(selected_task: Any) -> dict[str, Any]:
    if not isinstance(selected_task, dict):
        return {}
    selected_stage = selected_task.get("selected_stage") or selected_task.get("stage_type") or selected_task.get("task_type")
    stage_agent = {
        "recon": "recon_agent",
        "vuln_analysis": "vuln_analysis_agent",
        "exploit": "exploit_validation_agent",
        "access_pivot": "access_pivot_agent",
        "goal": "goal_agent",
        "RECON_STAGE": "recon_agent",
        "VULN_ANALYSIS_STAGE": "vuln_analysis_agent",
        "EXPLOIT_STAGE": "exploit_validation_agent",
        "ACCESS_PIVOT_STAGE": "access_pivot_agent",
        "GOAL_STAGE": "goal_agent",
    }.get(str(selected_stage))
    canonical_stage = {
        "recon": "RECON_STAGE",
        "vuln_analysis": "VULN_ANALYSIS_STAGE",
        "exploit": "EXPLOIT_STAGE",
        "access_pivot": "ACCESS_PIVOT_STAGE",
        "goal": "GOAL_STAGE",
    }.get(str(selected_stage), selected_stage)
    return {
        "selected_agent": selected_task.get("selected_agent") or stage_agent,
        "selected_stage": canonical_stage,
        "objective": selected_task.get("objective") or selected_task.get("label"),
        "target_refs": selected_task.get("target_refs") or selected_task.get("input_refs") or [],
        "required_context": selected_task.get("required_context") or {},
        "success_criteria": selected_task.get("success_criteria") or [],
        "risk_level": selected_task.get("risk_level") or "medium",
        "max_steps": selected_task.get("max_steps") or 3,
    }


__all__ = ["MissionPlannerAgent", "MissionPlannerAdvisor"]
