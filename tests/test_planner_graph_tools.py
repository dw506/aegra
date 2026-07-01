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
    findings = [node.model_dump(mode="json") for node in kg.list_nodes(type="Finding")]
    attack_step = tools.record_attack_step(
        {
            "target_ref": "host::10.0.0.5",
            "status": "success",
            "summary": "Recon objective selected",
            "evidence_refs": ["evidence::probe"],
        }
    )

    assert result["applied_entity_ids"]
    assert findings[0]["properties"]["title"] == "Service exposure needs validation"
    assert attack_step["recorded"] is True
    assert state.execution.metadata["planner_attack_step_records"][0]["summary"] == "Recon objective selected"


def test_planner_graph_tools_read_surface() -> None:
    state = RuntimeState(operation_id="op-read", execution=OperationRuntime(operation_id="op-read"))
    state.execution.metadata["success_condition_progress"] = {
        "missing": ["exploit_success_recorded"],
        "satisfied": ["target_imported"],
        "eligible_for_stop": False,
    }
    kg = KnowledgeGraph()
    ag = AttackGraph()
    tools = PlannerGraphTools(operation_id="op-read", cycle_index=1, kg=kg, ag=ag, runtime_state=state)

    # Seed a node via the write tool, then read it back through the read surface.
    tools.record_finding(
        {
            "host_ref": "host::10.0.0.9",
            "title": "Readable finding",
            "severity": "low",
            "summary": "for read-tool test",
        }
    )

    # manifests
    manifest = PlannerGraphTools.tool_manifest()
    assert "read" in manifest and "query_kg_nodes" in manifest["read"]
    assert {entry["name"] for entry in PlannerGraphTools.read_tool_manifest()} == set(manifest["read"])

    # get_success_progress
    progress = tools.get_success_progress()
    assert progress["missing"] == ["exploit_success_recorded"]

    # query_kg_nodes + get_node
    nodes = tools.query_kg_nodes(node_type="Finding")
    assert len(nodes) == 1
    node_id = nodes[0]["id"]
    detail = tools.get_node(node_id)
    assert detail["node"]["id"] == node_id
    assert "edges_out" in detail and "edges_in" in detail
    assert tools.get_node("missing-id")["error"] == "node_not_found"

    # list_runtime + dispatcher + limit
    assert tools.list_runtime("sessions") == []
    assert tools.apply_read_call("query_kg_nodes", {"node_type": "Finding", "limit": 1})[0]["id"] == node_id
    assert tools.apply_read_call("nope", {})["error"] == "unknown_read_tool"
