"""Route PlannerDecision dispatches to concrete Stage Agents."""

from __future__ import annotations

from typing import Any

from src.core.planning.models import PlannerDecision
from src.core.stage.models import StageExecutionRequest, StageName, StageResult, StageType, normalize_stage_name
from src.core.stage.registry import StageAgentRegistry


class StageDispatcher:
    """Resolve the planner-selected agent and run the matching StageAgent."""

    AGENT_STAGE_MAP = {
        "recon_agent": StageType.RECON_STAGE,
        "vuln_analysis_agent": StageType.VULN_ANALYSIS_STAGE,
        "exploit_validation_agent": StageType.EXPLOIT_STAGE,
        "access_pivot_agent": StageType.ACCESS_PIVOT_STAGE,
        "goal_agent": StageType.GOAL_STAGE,
    }

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

        expected_stage = self.AGENT_STAGE_MAP.get(decision.selected_agent)
        if expected_stage is None:
            raise ValueError(f"unsupported selected_agent: {decision.selected_agent}")

        selected_stage: StageName = normalize_stage_name(decision.selected_stage)
        if selected_stage != normalize_stage_name(expected_stage):
            raise ValueError(
                f"selected_agent {decision.selected_agent} requires selected_stage "
                f"{expected_stage.value}, got {selected_stage}"
            )

        agent = self._registry.resolve(selected_stage)
        request = StageExecutionRequest(
            operation_id=decision.operation_id,
            cycle_index=decision.cycle_index,
            agent_name=decision.selected_agent,
            stage_type=selected_stage,
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
        return agent.run(request)


__all__ = ["StageDispatcher"]
