"""Route PlannerDecision dispatches to concrete Stage Agents."""

from __future__ import annotations

from typing import Any

from src.core.planning.models import PlannerDecision
from src.core.stage.models import StageExecutionRequest, StageResult, normalize_stage_name
from src.core.stage.registry import StageAgentRegistry


class StageDispatcher:
    """Resolve the planner-selected agent and run the matching StageAgent."""

    def __init__(self, registry: StageAgentRegistry) -> None:
        self._registry = registry

    def dispatch(
        self,
        decision: PlannerDecision,
        *,
        kg_snapshot: dict[str, Any] | None = None,
        ag_process_history: dict[str, Any] | None = None,
        runtime_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
        mcp_tool_catalog: dict[str, Any] | None = None,
    ) -> StageResult:
        if decision.decision != "dispatch_agent":
            raise ValueError("StageDispatcher requires decision=dispatch_agent")
        if not decision.selected_agent:
            raise ValueError("selected_agent is required for dispatch_agent")
        if not decision.selected_stage:
            raise ValueError("selected_stage is required for dispatch_agent")

        try:
            agent = self._registry.resolve_agent(decision.selected_agent)
            self._registry.validate_assignment(
                agent_name=decision.selected_agent,
                stage_type=decision.selected_stage,
            )
        except ValueError as exc:
            return self._needs_replan_result(decision, str(exc))
        try:
            request = StageExecutionRequest(
                operation_id=decision.operation_id,
                cycle_index=decision.cycle_index,
                agent_name=decision.selected_agent,
                stage_type=decision.selected_stage,
                objective=decision.objective,
                target_refs=list(decision.target_refs),
                required_context=dict(decision.required_context),
                success_criteria=list(decision.success_criteria),
                risk_level=decision.risk_level,
                max_steps=decision.max_steps,
                kg_snapshot=dict(kg_snapshot or {}),
                ag_process_history=dict(ag_process_history or {}),
                runtime_context=dict(runtime_context or {}),
                policy_context=dict(policy_context or {}),
                mcp_tool_catalog=dict(mcp_tool_catalog or {}),
            )
        except ValueError as exc:
            return self._needs_replan_result(decision, str(exc))
        return agent.run(request)

    @staticmethod
    def _needs_replan_result(decision: PlannerDecision, reason: str) -> StageResult:
        try:
            stage_type = normalize_stage_name(decision.selected_stage or "RECON_STAGE")
        except ValueError:
            stage_type = "RECON_STAGE"
        return StageResult(
            operation_id=decision.operation_id,
            stage_task_id=f"stage-{decision.operation_id}-{decision.cycle_index}-{decision.selected_agent or 'unresolved'}",
            stage_type=stage_type,
            agent_name=decision.selected_agent or "unresolved_agent",
            status="needs_replan",
            summary=reason,
            replan_recommendation=reason,
            runtime_hints={"cycle_index": decision.cycle_index},
        )


__all__ = ["StageDispatcher"]
