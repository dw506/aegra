from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.planning.graph_tools import PlannerGraphTools


def test_planner_graph_tools_query_and_record_finding() -> None:
    state = RuntimeState(operation_id="op-tools", execution=OperationRuntime(operation_id="op-tools"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    tools = PlannerGraphTools(
        operation_id="op-tools",
        cycle_index=1,
        kg=kg,
        ag=ag,
        runtime_state=state,
    )

    result = tools.record_finding(
        {
            "host_ref": "host::10.0.0.5",
            "title": "Service exposure needs validation",
            "severity": "medium",
            "summary": "Planner judged this service as relevant to the missing condition.",
            "evidence_refs": ["evidence::probe"],
        }
    )
    findings = tools.kg_query("Finding", {"severity": "medium"})
    attack_step = tools.record_attack_step(
        {
            "capability": "recon",
            "target_ref": "host::10.0.0.5",
            "status": "success",
            "summary": "Recon objective selected",
            "evidence_refs": ["evidence::probe"],
            "kg_node_refs": [findings[0]["id"]],
        }
    )

    assert result["applied_entity_ids"]
    assert findings[0]["properties"]["title"] == "Service exposure needs validation"
    assert attack_step["recorded"] is True
    assert state.execution.metadata["planner_attack_step_records"][0]["capability"] == "recon"
