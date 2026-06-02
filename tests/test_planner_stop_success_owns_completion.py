from __future__ import annotations

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.models.runtime import RuntimeStatus
from src.core.planning.models import PlannerDecision
from src.core.stage.models import StageExecutionRequest, StageResult
from src.core.stage.registry import StageAgentRegistry


class GoalSatisfiedAgent:
    agent_name = "goal_agent"
    stage_type = "GOAL_STAGE"

    def run(self, request: StageExecutionRequest) -> StageResult:
        return StageResult(
            operation_id=request.operation_id,
            stage_task_id=f"stage-{request.operation_id}-{request.cycle_index}-goal_agent",
            stage_type="GOAL_STAGE",
            agent_name=self.agent_name,
            status="succeeded",
            summary="goal evidence satisfied",
            findings=[{"kind": "GoalCheck", "goal_satisfied": True}],
            evidence_refs=["evidence::goal"],
            runtime_hints={
                "goal_satisfied": True,
                "goal_summary": "goal evidence satisfied",
                "goal_evidence_refs": ["evidence::goal"],
            },
        )


class SequencedPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, **_: object) -> PlannerDecision:
        self.calls += 1
        if self.calls == 1:
            return PlannerDecision(
                operation_id="operation",
                cycle_index=0,
                decision="dispatch_agent",
                selected_agent="goal_agent",
                selected_stage="GOAL_STAGE",
                objective="verify goal evidence",
                risk_level="low",
                max_steps=1,
                confidence=0.9,
            )
        return PlannerDecision(
            operation_id="operation",
            cycle_index=0,
            decision="stop_success",
            objective="complete after planner reviewed goal evidence",
            risk_level="low",
            max_steps=1,
            stop_condition="goal_satisfied",
            reasoning_summary="GoalAgent produced evidence-backed goal_satisfied=true and the required goal evidence is recorded.",
            confidence=0.9,
        )


def test_goal_agent_hint_does_not_complete_until_planner_stop_success(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path)
    orchestrator = AppOrchestrator(settings=settings, graph_memory_store=GraphMemoryStore(tmp_path / "graphs"))
    orchestrator.stage_registry = StageAgentRegistry([GoalSatisfiedAgent()])  # type: ignore[list-item]
    orchestrator.mission_planner = SequencedPlanner()  # type: ignore[assignment]
    orchestrator.create_operation("op-goal-stop")

    first = orchestrator.run_operation_cycle("op-goal-stop", graph_refs=[], planner_payload={"mission_goal": "prove goal"})

    assert first.runtime_state.operation_status == RuntimeStatus.READY
    assert first.runtime_state.execution.metadata["goal_satisfied"] is True
    assert first.stopped is False

    second = orchestrator.run_operation_cycle("op-goal-stop", graph_refs=[], planner_payload={"mission_goal": "prove goal"})

    assert second.runtime_state.operation_status == RuntimeStatus.COMPLETED
    assert second.stopped is True
    assert second.stop_reason == "goal_satisfied"
