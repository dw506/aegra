from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.models import StageResult


def test_goal_agent_goal_satisfied_runtime_hint_is_recorded_without_dispatching_next_agent() -> None:
    state = RuntimeState(operation_id="op-goal", execution=OperationRuntime(operation_id="op-goal"))
    result = StageResult(
        operation_id="op-goal",
        stage_task_id="stage-op-goal-1-goal_agent",
        stage_type="GOAL_STAGE",
        agent_name="goal_agent",
        status="succeeded",
        summary="goal evidence satisfied",
        findings=[{"kind": "GoalCheck", "goal_satisfied": True}],
        evidence_refs=["evidence::goal"],
        runtime_hints={"goal_satisfied": True, "goal_summary": "goal evidence satisfied", "goal_evidence_refs": ["evidence::goal"]},
    )

    PhaseTwoResultApplier().apply_stage_result(result, state, KnowledgeGraph(), AttackGraph())

    assert state.execution.metadata["goal_state"]["goal_satisfied"] is True
    assert state.execution.metadata["goal_state"]["source_task_id"] == "stage-op-goal-1-goal_agent"
    assert "last_planner_decision" not in state.execution.metadata
