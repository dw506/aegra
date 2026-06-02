from __future__ import annotations

from src.core.planning.models import PlannerDecision
from src.core.stage.dispatcher import StageDispatcher
from src.core.stage.models import StageExecutionRequest, StageResult
from src.core.stage.registry import StageAgentRegistry


class RecordingReconAgent:
    agent_name = "recon_agent"
    stage_type = "RECON_STAGE"

    def __init__(self) -> None:
        self.requests: list[StageExecutionRequest] = []

    def run(self, request: StageExecutionRequest) -> StageResult:
        self.requests.append(request)
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-recon_agent",
            stage_type=request.stage_type,
            agent_name=self.agent_name,
            status="succeeded",
            summary="recon complete",
        )


def test_planner_decision_dispatches_selected_recon_agent() -> None:
    recon_agent = RecordingReconAgent()
    dispatcher = StageDispatcher(StageAgentRegistry([recon_agent]))  # type: ignore[list-item]
    decision = PlannerDecision(
        operation_id="op-dispatch",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="collect environment facts",
        risk_level="low",
        max_steps=2,
        confidence=0.9,
    )

    result = dispatcher.dispatch(decision)

    assert result.agent_name == "recon_agent"
    assert recon_agent.requests[0].agent_name == "recon_agent"
    assert recon_agent.requests[0].stage_type == "RECON_STAGE"
