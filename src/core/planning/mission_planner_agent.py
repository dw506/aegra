"""LLM-owned mission planner for graph-state-driven decisions."""

from __future__ import annotations

from typing import Any, Protocol

from src.core.planning.graph_tools import PlannerGraphTools
from src.core.planning.models import PlannerOutcome


class MissionPlannerAdvisor(Protocol):
    """LLM planner hook. The advisor owns next-outcome choice."""

    def propose_next_decision(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_stage_results: list[dict[str, Any]] | None = None,
    ) -> PlannerOutcome | dict[str, Any]:
        """Return the next PlannerOutcome (or a dict validated into one)."""


class MissionPlannerAgent:
    """Planner Agent facade.

    This class intentionally does not contain a deterministic stage sequence or
    keyword-to-stage planner. Its job is context assembly, LLM invocation through
    the advisor, schema validation, and returning the PlannerAgent-owned outcome.
    """

    def __init__(self, advisor: MissionPlannerAdvisor | None = None) -> None:
        self._advisor = advisor

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
        outcome = _outcome_from_payload(raw, operation_id=operation_id, cycle_index=cycle_index)
        if graph_tools is not None and isinstance(raw, dict):
            tool_results = graph_tools.apply_tool_calls(raw.get("planner_tool_calls") or raw.get("tool_calls"))
            if tool_results:
                outcome.metadata = {**dict(outcome.metadata), "planner_graph_tool_results": tool_results}
        if outcome.operation_id == "operation":
            outcome.operation_id = operation_id
        return outcome


def _outcome_from_payload(payload: Any, *, operation_id: str, cycle_index: int) -> PlannerOutcome:
    if isinstance(payload, PlannerOutcome):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("planner advisor must return PlannerOutcome or dict")
    payload = dict(payload)
    payload.setdefault("operation_id", operation_id)
    payload.setdefault("cycle_index", cycle_index)
    return PlannerOutcome.model_validate(payload)


__all__ = ["MissionPlannerAgent", "MissionPlannerAdvisor"]
