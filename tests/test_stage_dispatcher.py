from __future__ import annotations

from src.core.planning.models import PlannerDecision
from src.core.stage.dispatcher import StageDispatcher
from src.core.stage.models import StageExecutionRequest, StageResult, StageType
from src.core.stage.registry import StageAgentRegistry


class RecordingStageAgent:
    def __init__(self, agent_name: str, stage_type: StageType) -> None:
        self.agent_name = agent_name
        self.stage_type = stage_type
        self.requests: list[StageExecutionRequest] = []

    def run(self, request: StageExecutionRequest) -> StageResult:
        self.requests.append(request)
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-{request.agent_name}",
            stage_type=request.stage_type,
            agent_name=self.agent_name,
            status="succeeded",
            summary=f"ran {self.agent_name}",
        )


def _decision(agent_name: str, stage: str) -> PlannerDecision:
    return PlannerDecision(
        operation_id="op-dispatch",
        cycle_index=3,
        decision="dispatch_agent",
        selected_agent=agent_name,  # type: ignore[arg-type]
        selected_stage=stage,  # type: ignore[arg-type]
        objective="execute selected stage",
        risk_level="low",
        max_steps=2,
        confidence=0.9,
    )


def test_stage_dispatcher_resolves_all_five_agents() -> None:
    agents = [
        RecordingStageAgent("recon_agent", StageType.RECON_STAGE),
        RecordingStageAgent("vuln_analysis_agent", StageType.VULN_ANALYSIS_STAGE),
        RecordingStageAgent("exploit_validation_agent", StageType.EXPLOIT_STAGE),
        RecordingStageAgent("access_pivot_agent", StageType.ACCESS_PIVOT_STAGE),
        RecordingStageAgent("goal_agent", StageType.GOAL_STAGE),
    ]
    dispatcher = StageDispatcher(StageAgentRegistry(agents))  # type: ignore[arg-type]

    for agent in agents:
        result = dispatcher.dispatch(
            _decision(agent.agent_name, agent.stage_type.value),
            kg_snapshot={"nodes": []},
            ag_process_history={"nodes": []},
            runtime_context={"operation_id": "op-dispatch"},
            policy_context={"authorized": True},
            mcp_tool_catalog={"available": False},
        )

        assert result.agent_name == agent.agent_name
        assert result.stage_type == agent.stage_type.value
        assert agent.requests[-1].agent_name == agent.agent_name
        assert agent.requests[-1].stage_type == agent.stage_type.value


def test_stage_dispatcher_replans_for_agent_stage_mismatch() -> None:
    dispatcher = StageDispatcher(
        StageAgentRegistry([RecordingStageAgent("recon_agent", StageType.RECON_STAGE)])  # type: ignore[arg-type]
    )

    result = dispatcher.dispatch(_decision("recon_agent", "GOAL_STAGE"))

    assert result.status == "needs_replan"
    assert result.agent_name == "recon_agent"
    assert result.stage_type == "GOAL_STAGE"
    assert "not GOAL_STAGE" in result.summary


def test_stage_dispatcher_returns_needs_replan_for_unregistered_agent() -> None:
    dispatcher = StageDispatcher(
        StageAgentRegistry([RecordingStageAgent("recon_agent", StageType.RECON_STAGE)])  # type: ignore[arg-type]
    )

    result = dispatcher.dispatch(_decision("custom_agent", "RECON_STAGE"))

    assert result.status == "needs_replan"
    assert result.agent_name == "custom_agent"
    assert "no StageAgent registered" in result.summary
