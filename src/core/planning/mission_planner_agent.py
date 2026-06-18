"""LLM-owned mission planner for graph-state-driven decisions."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError

from src.core.planning.graph_tools import PlannerGraphTools
from src.core.planning.models import PlannerDecision, PlannerOutcome
from src.core.stage.models import RoundDirective, normalize_stage_name


class MissionPlannerAdvisor(Protocol):
    """LLM planner hook. The advisor owns next-outcome choice."""

    def propose_next_decision(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> PlannerOutcome | PlannerDecision | dict[str, Any]:
        """Return a PlannerOutcome, or a legacy PlannerDecision for compatibility."""


class MissionPlannerAgent:
    """Planner Agent facade.

    This class intentionally does not contain a deterministic stage sequence or
    keyword-to-stage planner. Its job is context assembly, LLM invocation through
    the advisor, schema validation, and returning the PlannerAgent-owned outcome.
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
        """Compatibility wrapper returning the legacy PlannerDecision contract."""

        return self._legacy_decision(
            self.decide(
                goal=goal,
                graph_context=graph_context,
                policy_context=policy_context,
                recent_stage_results=recent_stage_results,
                graph_tools=None,
            ),
            goal=goal,
        )

    def decide(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any] | None = None,
        recent_stage_results: list[dict[str, Any]] | None = None,
        graph_tools: PlannerGraphTools | None = None,
    ) -> PlannerOutcome:
        """Return a P3 planner outcome: execute one directive or stop/pause."""

        operation_id = str(graph_context.get("operation_id") or "operation")
        cycle_index = int(graph_context.get("cycle_index") or graph_context.get("runtime", {}).get("cycle_index") or 0)
        if self._advisor is None:
            return PlannerOutcome(
                operation_id=operation_id,
                cycle_index=cycle_index,
                action="replan",
                reason="PlannerAgent requires an LLM advisor; no hard-coded fallback is available.",
                stop_condition="planner_llm_unavailable",
                confidence=0.0,
                metadata={"planner": "llm_planner_required", "accepted": False},
            )

        planner_context = dict(graph_context)
        if graph_tools is not None:
            planner_context["min_summary"] = graph_tools.build_min_summary()
            planner_context["graph_tools"] = {
                "read": ["kg_query", "kg_get_node", "kg_neighbors", "ag_get_timeline", "ag_get_step", "get_round_log"],
                "write": ["record_finding", "record_attack_step", "link_evidence"],
            }
        raw = self._advisor.propose_next_decision(
            goal=goal,
            graph_context=planner_context,
            policy_context=dict(policy_context or {}),
            recent_stage_results=list(recent_stage_results or []),
        )
        outcome = _outcome_from_payload(raw, operation_id=operation_id, cycle_index=cycle_index, goal=goal)
        if graph_tools is not None and isinstance(raw, dict):
            tool_results = graph_tools.apply_tool_calls(raw.get("planner_tool_calls") or raw.get("tool_calls"))
            if tool_results:
                outcome.metadata = {**dict(outcome.metadata), "planner_graph_tool_results": tool_results}
        if outcome.operation_id == "operation":
            outcome.operation_id = operation_id
        return outcome

    @staticmethod
    def _legacy_decision(outcome: PlannerOutcome, *, goal: str) -> PlannerDecision:
        return decision_from_outcome(outcome, goal=goal)


def _outcome_from_payload(
    payload: Any,
    *,
    operation_id: str,
    cycle_index: int,
    goal: str,
) -> PlannerOutcome:
    if isinstance(payload, PlannerOutcome):
        return payload
    if isinstance(payload, PlannerDecision):
        return outcome_from_decision(payload)
    if not isinstance(payload, dict):
        raise TypeError("planner advisor must return PlannerOutcome, PlannerDecision or dict")
    try:
        return PlannerOutcome.model_validate(payload)
    except ValidationError:
        if "directive" in payload and payload.get("action") == "execute":
            raise
        decision = _decision_from_payload(payload, operation_id=operation_id, cycle_index=cycle_index, goal=goal)
        return outcome_from_decision(decision)


def outcome_from_decision(decision: PlannerDecision) -> PlannerOutcome:
    if decision.decision == "dispatch_agent":
        stage = normalize_stage_name(decision.selected_stage)
        capability = _capability_for_stage(stage)
        directive = RoundDirective(
            operation_id=decision.operation_id,
            cycle_index=decision.cycle_index,
            capability=capability,
            objective=decision.objective,
            target_refs=list(decision.target_refs),
            allowed_tools=_allowed_tools(decision.allowed_tool_names),
            tool_hints=_tool_hints(decision),
            max_tools=decision.max_steps,
            success_hint="; ".join(decision.success_criteria) or None,
            required_context=dict(decision.required_context),
            risk_level=decision.risk_level,
            legacy_agent_name=decision.selected_agent,
            legacy_stage_type=stage,
        )
        return PlannerOutcome(
            operation_id=decision.operation_id,
            cycle_index=decision.cycle_index,
            action="execute",
            directive=directive,
            reason=decision.reasoning_summary,
            confidence=decision.confidence,
            metadata={**dict(decision.metadata), "legacy_planner_decision": decision.model_dump(mode="json")},
        )
    action = "replan" if decision.decision == "replan" else decision.decision
    return PlannerOutcome(
        operation_id=decision.operation_id,
        cycle_index=decision.cycle_index,
        action=action,  # type: ignore[arg-type]
        reason=decision.reasoning_summary or decision.objective,
        stop_condition=decision.stop_condition,
        confidence=decision.confidence,
        metadata={**dict(decision.metadata), "legacy_planner_decision": decision.model_dump(mode="json")},
    )


def decision_from_outcome(outcome: PlannerOutcome, *, goal: str) -> PlannerDecision:
    if outcome.action == "execute" and outcome.directive is not None:
        directive = outcome.directive
        return PlannerDecision(
            operation_id=outcome.operation_id,
            cycle_index=outcome.cycle_index,
            decision="dispatch_agent",
            selected_agent=directive.legacy_agent_name or _agent_for_stage(directive.legacy_stage_type),
            selected_stage=directive.legacy_stage_type or _stage_for_capability(directive.capability),
            objective=directive.objective,
            target_refs=list(directive.target_refs),
            required_context=dict(directive.required_context),
            success_criteria=[directive.success_hint] if directive.success_hint else [],
            risk_level=directive.risk_level,  # type: ignore[arg-type]
            max_steps=directive.max_tools,
            allowed_tool_names=list(directive.allowed_tools),
            reasoning_summary=outcome.reason,
            stop_condition=outcome.stop_condition,
            confidence=outcome.confidence,
            metadata=dict(outcome.metadata),
        )
    decision = "replan" if outcome.action == "replan" else outcome.action
    return PlannerDecision(
        operation_id=outcome.operation_id,
        cycle_index=outcome.cycle_index,
        decision=decision,  # type: ignore[arg-type]
        selected_agent=None,
        selected_stage=None,
        objective=goal,
        risk_level="medium",
        max_steps=1,
        reasoning_summary=outcome.reason,
        stop_condition=outcome.stop_condition,
        confidence=outcome.confidence,
        metadata=dict(outcome.metadata),
    )


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


_STAGE_TO_CAPABILITY = {
    "RECON_STAGE": "recon",
    "VULN_ANALYSIS_STAGE": "analysis",
    "EXPLOIT_STAGE": "exploit",
    "ACCESS_PIVOT_STAGE": "pivot",
    "GOAL_STAGE": "goal",
}

_CAPABILITY_TO_STAGE = {
    "recon": "RECON_STAGE",
    "analysis": "VULN_ANALYSIS_STAGE",
    "exploit": "EXPLOIT_STAGE",
    "pivot": "ACCESS_PIVOT_STAGE",
    "lateral": "ACCESS_PIVOT_STAGE",
    "goal": "GOAL_STAGE",
    "evidence": "GOAL_STAGE",
}

_STAGE_TO_AGENT = {
    "RECON_STAGE": "recon_agent",
    "VULN_ANALYSIS_STAGE": "vuln_analysis_agent",
    "EXPLOIT_STAGE": "exploit_validation_agent",
    "ACCESS_PIVOT_STAGE": "access_pivot_agent",
    "GOAL_STAGE": "goal_agent",
}


def _capability_for_stage(stage: str) -> str:
    return _STAGE_TO_CAPABILITY.get(stage, "evidence")


def _stage_for_capability(capability: str | None) -> str:
    return _CAPABILITY_TO_STAGE.get(str(capability or "evidence"), "GOAL_STAGE")


def _agent_for_stage(stage: str | None) -> str:
    return _STAGE_TO_AGENT.get(str(stage or "GOAL_STAGE"), "goal_agent")


def _allowed_tools(raw: list[str] | str | None) -> list[str]:
    if isinstance(raw, str):
        if raw.strip() == "*":
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _tool_hints(decision: PlannerDecision) -> list[dict[str, Any]]:
    raw = decision.required_context.get("tool_hints") or decision.metadata.get("tool_hints") or []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _legacy_selected_task_fields(selected_task: Any) -> dict[str, Any]:
    if not isinstance(selected_task, dict):
        return {}
    selected_stage = selected_task.get("selected_stage") or selected_task.get("stage_type") or selected_task.get("task_type")
    return {
        "selected_agent": selected_task.get("selected_agent"),
        "selected_stage": selected_stage,
        "objective": selected_task.get("objective") or selected_task.get("label"),
        "target_refs": selected_task.get("target_refs") or selected_task.get("input_refs") or [],
        "required_context": selected_task.get("required_context") or {},
        "success_criteria": selected_task.get("success_criteria") or [],
        "risk_level": selected_task.get("risk_level") or "medium",
        "max_steps": selected_task.get("max_steps") or 3,
    }


__all__ = ["MissionPlannerAgent", "MissionPlannerAdvisor", "decision_from_outcome", "outcome_from_decision"]
