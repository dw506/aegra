from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.agents.agent_protocol import GraphScope
from src.core.models.ag import AttackGraph, GraphRef
from src.core.models.attack_process import AttackProcessNodeType
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.planning.models import PlannerDecision
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.models import StageResult, StageType, ToolTrace


def test_result_applier_writes_planner_stage_tool_and_kg_facts_without_tg() -> None:
    state = RuntimeState(operation_id="op-apply", execution=OperationRuntime(operation_id="op-apply"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    decision = PlannerDecision(
        operation_id="op-apply",
        cycle_index=1,
        decision="dispatch_agent",
        selected_agent="recon_agent",
        selected_stage="RECON_STAGE",
        objective="map exposed service",
        risk_level="low",
        max_steps=2,
        confidence=0.9,
    )
    planner_apply = applier.apply_planner_decision(decision, state, kg, ag)

    stage_result = StageResult(
        operation_id="op-apply",
        stage_task_id="stage-op-apply-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="host and service observed",
        discovered_entities=[
            {"id": "host-1", "type": "Host", "summary": "10.0.0.5", "address": "10.0.0.5", "confidence": 0.9},
            {"id": "svc-1", "type": "Service", "summary": "http service", "port": 80, "protocol": "http"},
            {"id": "evidence-1", "type": "Evidence", "summary": "probe output"},
        ],
        tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok")],
    )
    stage_apply = applier.apply_stage_result(stage_result, state, kg, ag)

    assert {delta.graph for delta in planner_apply.visual_graph_deltas + stage_apply.visual_graph_deltas} <= {
        "kg",
        "ag",
        "runtime",
    }
    process_types = {node.node_type for node in ag.find_process_nodes()}
    assert AttackProcessNodeType.PLANNER_DECISION in process_types
    assert AttackProcessNodeType.STAGE_RESULT in process_types
    assert AttackProcessNodeType.TOOL_CALL in process_types
    assert kg.get_node("host-1") is not None
    assert kg.get_node("svc-1") is not None
    assert kg.get_node("evidence-1") is not None
    assert "task_graph" not in state.execution.metadata


def test_result_applier_maps_query_refs_to_ag_protocol_refs() -> None:
    ref = PhaseTwoResultApplier._to_protocol_ref(
        GraphRef(graph="query", ref_id="expected-output::svc", ref_type="ExpectedEvidence")
    )

    assert ref.graph == GraphScope.AG
    assert ref.ref_id == "expected-output::svc"
    assert ref.metadata["original_graph"] == "query"
