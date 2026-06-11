from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.context.goal_context import build_goal_context
from src.core.stage.models import StageExecutionRequest, StageResult


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


def test_goal_context_uses_configured_requirements_without_runtime_routes() -> None:
    request = StageExecutionRequest(
        operation_id="op-goal",
        cycle_index=1,
        agent_name="goal_agent",
        stage_type="GOAL_STAGE",
        objective="verify configured goal",
        required_context={"goal_requirements": {"require_all": ["validated_access", "goal_check"]}},
        success_criteria=["all configured conditions have evidence"],
    )

    context = build_goal_context(
        request,
        graph_context={"recent_evidence": []},
        runtime_context={"pivot_routes": [{"route_id": "hidden-route"}]},
        policy_context={"authorized_targets": []},
        memory=[],
        available_tools={},
    )

    assert context["goal_focus"]["goal_requirements"] == {"require_all": ["validated_access", "goal_check"]}
    assert "runtime_pivot_routes" not in context["goal_focus"]
