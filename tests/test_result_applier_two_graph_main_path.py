from __future__ import annotations

from pathlib import Path

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.planning.models import PlannerDecision
from src.core.runtime.attack_log_extractor import AttackLogExtractor
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.models import StageResult, ToolTrace


def test_result_applier_branches_write_kg_ag_runtime_without_tg() -> None:
    state = RuntimeState(operation_id="op-two-graph", execution=OperationRuntime(operation_id="op-two-graph"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    decision = PlannerDecision(
        operation_id="op-two-graph",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="Collect host facts",
        risk_level="low",
        max_steps=2,
        confidence=0.9,
    )
    planner_apply = applier.apply_planner_decision(decision, state, kg, ag)

    stage_result = StageResult(
        operation_id="op-two-graph",
        stage_task_id="stage-op-two-graph-1-recon_agent",
        stage_type="RECON_STAGE",
        agent_name="recon_agent",
        status="succeeded",
        summary="host discovered",
        discovered_entities=[
            {"id": "host-1", "type": "Host", "summary": "host 1", "confidence": 0.9}
        ],
        tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok")],
    )
    stage_apply = applier.apply_stage_result(stage_result, state, kg, ag)
    log_apply = applier.apply_log_extraction(AttackLogExtractor().extract(stage_result), state, ag)

    assert planner_apply.tg_graph is None
    assert stage_apply.tg_graph is None
    assert log_apply.tg_graph is None
    assert state.execution.metadata["last_planner_decision"]["decision"] == "dispatch_agent"
    assert kg.get_node("host-1").label == "host 1"
    assert any(node.node_type.value == "STAGE_RESULT" for node in ag.find_process_nodes())
    assert any(entry["event_type"] == "attack_log_extraction_applied" for entry in state.execution.metadata["audit_log"])


def test_result_applier_has_no_top_level_task_graph_import() -> None:
    source = Path("src/core/runtime/result_applier.py").read_text(encoding="utf-8").splitlines()[:90]

    assert not any("from src.core.models.tg import TaskGraph" in line for line in source)
    assert not any("merge_task_graphs" in line for line in source)
