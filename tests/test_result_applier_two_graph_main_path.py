from __future__ import annotations

from pathlib import Path

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.planning.models import PlannerOutcome
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.models import RoundDirective, StageResult, ToolTrace


def test_result_applier_branches_write_kg_ag_runtime_without_tg() -> None:
    state = RuntimeState(operation_id="op-two-graph", execution=OperationRuntime(operation_id="op-two-graph"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    outcome = PlannerOutcome(
        operation_id="op-two-graph",
        cycle_index=1,
        action="execute",
        directive=RoundDirective(
            operation_id="op-two-graph",
            cycle_index=1,
            capability="recon",
            objective="Collect host facts",
            max_tools=2,
            risk_level="low",
        ),
        confidence=0.9,
    )
    planner_apply = applier.apply_planner_outcome(outcome, state, kg, ag)

    stage_result = StageResult(
        operation_id="op-two-graph",
        stage_task_id="stage-op-two-graph-1-recon_agent",
        capability="recon",
        agent_name="recon_agent",
        status="succeeded",
        summary="host discovered",
        discovered_entities=[
            {"id": "host-1", "type": "Host", "summary": "host 1", "confidence": 0.9}
        ],
        tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok")],
    )
    stage_apply = applier.apply_stage_result(stage_result, state, kg, ag)
    assert stage_apply.ag_graph is not None

    assert state.execution.metadata["last_planner_outcome"]["action"] == "execute"
    assert kg.get_node("host-1").label == "host 1"
    # v3 result-tier AG: exactly one ATTACK_STEP per round, no process nodes.
    process_nodes = ag.find_process_nodes()
    step_nodes = [node for node in process_nodes if node.node_type.value == "ATTACK_STEP"]
    assert len(step_nodes) == 1
    assert step_nodes[0].capability == "recon"
    assert {node.node_type.value for node in process_nodes} == {"ATTACK_STEP"}


def test_result_applier_has_no_top_level_task_graph_import() -> None:
    source = Path("src/core/runtime/result_applier.py").read_text(encoding="utf-8").splitlines()[:90]

    assert not any("from src.core.models.tg import TaskGraph" in line for line in source)
    assert not any("merge_task_graphs" in line for line in source)
