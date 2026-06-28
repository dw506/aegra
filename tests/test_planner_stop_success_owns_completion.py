from __future__ import annotations

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.models.runtime import RuntimeStatus
from src.core.planning.models import PlannerOutcome
from src.core.execution.models import RoundDirective, ExecutionRequest, ExecutionResult
from src.core.execution.execution_agent import ExecutionAgent


class GoalSatisfiedAgent:
    agent_name = "goal_agent"

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            operation_id=request.operation_id,
            execution_id=f"execution-{request.operation_id}-{request.cycle_index}-goal_agent",
            capability="goal",
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


class GoalWithoutExplicitSatisfiedHintAgent:
    agent_name = "goal_agent"

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            operation_id=request.operation_id,
            execution_id=f"execution-{request.operation_id}-{request.cycle_index}-goal_agent",
            capability="goal",
            agent_name=self.agent_name,
            status="succeeded",
            summary="goal stage succeeded but did not prove completion",
            findings=[{"kind": "GoalCheck", "goal_satisfied": True}],
            evidence_refs=["evidence::goal"],
        )


class SequencedPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **_: object) -> PlannerOutcome:
        self.calls += 1
        if self.calls == 1:
            return PlannerOutcome(
                operation_id="operation",
                cycle_index=0,
                action="execute",
                directive=RoundDirective(
                    operation_id="operation",
                    cycle_index=0,
                    capability="goal",
                    objective="verify goal evidence",
                    max_tools=1,
                    risk_level="low",
                ),
                confidence=0.9,
            )
        return PlannerOutcome(
            operation_id="operation",
            cycle_index=0,
            action="stop_success",
            reason="GoalAgent produced evidence-backed goal_satisfied=true and the required goal evidence is recorded.",
            stop_condition="goal_satisfied",
            confidence=0.9,
        )


def test_goal_agent_hint_does_not_complete_until_planner_stop_success(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path)
    orchestrator = AppOrchestrator(settings=settings, graph_memory_store=GraphMemoryStore(tmp_path / "graphs"))
    orchestrator.execution_agent = ExecutionAgent(GoalSatisfiedAgent())  # type: ignore[arg-type]
    orchestrator.planner = SequencedPlanner()  # type: ignore[assignment]
    orchestrator.create_operation("op-goal-stop")

    first = orchestrator.run_operation_cycle("op-goal-stop", graph_refs=[], planner_payload={"mission_goal": "prove goal"})

    assert first.runtime_state.operation_status == RuntimeStatus.READY
    assert first.runtime_state.execution.metadata["goal_satisfied"] is True
    assert first.stopped is False

    second = orchestrator.run_operation_cycle("op-goal-stop", graph_refs=[], planner_payload={"mission_goal": "prove goal"})

    assert second.runtime_state.operation_status == RuntimeStatus.COMPLETED
    assert second.stopped is True
    assert second.stop_reason == "goal_satisfied"


def test_goal_stage_success_without_explicit_hint_does_not_mark_goal_satisfied(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path)
    orchestrator = AppOrchestrator(settings=settings, graph_memory_store=GraphMemoryStore(tmp_path / "graphs"))
    orchestrator.execution_agent = ExecutionAgent(GoalWithoutExplicitSatisfiedHintAgent())  # type: ignore[arg-type]
    orchestrator.planner = SequencedPlanner()  # type: ignore[assignment]
    orchestrator.create_operation("op-goal-no-hint")

    first = orchestrator.run_operation_cycle("op-goal-no-hint", graph_refs=[], planner_payload={"mission_goal": "prove goal"})

    assert first.runtime_state.operation_status == RuntimeStatus.READY
    assert first.runtime_state.execution.metadata.get("goal_satisfied") is not True
    assert first.stopped is False
