from __future__ import annotations

from src.core.models.attack_process import AttackProcessEdgeType, AttackProcessNodeType
from src.core.planning.models import PlannerDecision
from src.core.runtime.attack_log_extractor import AttackLogExtractor
from src.core.stage.models import StageHandoffSuggestion, StageResult, StageType, ToolTrace


def test_attack_log_extractor_outputs_ag_process_chain() -> None:
    decision = PlannerDecision(
        operation_id="op-log-ag",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="collect facts",
        risk_level="low",
        max_steps=2,
        confidence=0.8,
    )
    stage_result = StageResult(
        operation_id="op-log-ag",
        stage_task_id="stage-op-log-ag-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="handoff to goal check",
        tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok")],
        handoff_suggestion=StageHandoffSuggestion(
            suggested_agent="goal_agent",
            suggested_stage="GOAL_STAGE",
            reason="facts collected",
            confidence=0.7,
        ),
    )

    extraction = AttackLogExtractor().extract(decision, stage_result)

    node_types = [node.node_type for node in extraction.ag_nodes]
    edge_types = {edge.edge_type for edge in extraction.ag_edges}
    assert node_types[:3] == [
        AttackProcessNodeType.ATTACK_CYCLE,
        AttackProcessNodeType.PLANNER_DECISION,
        AttackProcessNodeType.AGENT_EXECUTION,
    ]
    assert AttackProcessNodeType.TOOL_CALL in node_types
    assert AttackProcessNodeType.STAGE_RESULT in node_types
    assert AttackProcessNodeType.HANDOFF_SUGGESTION in node_types
    assert {
        AttackProcessEdgeType.PLANNED,
        AttackProcessEdgeType.DISPATCHED_TO,
        AttackProcessEdgeType.CALLED_TOOL,
        AttackProcessEdgeType.PRODUCED_RESULT,
        AttackProcessEdgeType.SUGGESTED_HANDOFF,
    }.issubset(edge_types)
