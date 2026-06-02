from __future__ import annotations

from datetime import datetime, timezone

from src.core.models.ag import AttackGraph, GraphRef
from src.core.models.attack_process import (
    AgentExecutionNode,
    AttackProcessEdge,
    AttackProcessEdgeType,
    AttackProcessNodeType,
    PlannerDecisionNode,
    StageResultNode,
    ToolCallNode,
)


def test_attack_graph_adds_and_restores_process_nodes() -> None:
    graph = AttackGraph()
    planner = PlannerDecisionNode(
        id="process-planner-1",
        label="Planner selected recon",
        operation_id="op-1",
        cycle_index=1,
        agent_name="PlannerAgent",
        status="selected",
        refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    execution = AgentExecutionNode(
        id="process-agent-1",
        label="Recon agent execution",
        operation_id="op-1",
        cycle_index=1,
        agent_name="ReconAgent",
        status="running",
        created_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    tool_call = ToolCallNode(
        id="process-tool-1",
        label="Nmap scan",
        operation_id="op-1",
        cycle_index=1,
        agent_name="ReconAgent",
        status="completed",
        properties={"tool": "nmap"},
        created_at=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    )
    result = StageResultNode(
        id="process-result-1",
        label="Recon result",
        operation_id="op-1",
        cycle_index=1,
        agent_name="ReconAgent",
        status="completed",
        summary="Open service found.",
        created_at=datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc),
    )

    graph.add_process_node(planner)
    graph.add_process_node(execution)
    graph.add_process_node(tool_call)
    graph.add_process_node(result)
    graph.add_process_edge(
        AttackProcessEdge(
            id="process-edge-1",
            edge_type=AttackProcessEdgeType.DISPATCHED_TO,
            source=planner.id,
            target=execution.id,
            label="dispatched to",
        )
    )
    graph.add_process_edge(
        id="process-edge-2",
        edge_type=AttackProcessEdgeType.CALLED_TOOL,
        source=execution.id,
        target=tool_call.id,
        label="called tool",
    )
    graph.add_process_edge(
        id="process-edge-3",
        edge_type=AttackProcessEdgeType.PRODUCED_RESULT,
        source=tool_call.id,
        target=result.id,
        label="produced result",
    )

    serialized = graph.to_dict()
    restored = AttackGraph.from_dict(serialized)

    assert {node["kind"] for node in serialized["nodes"]} == {"process"}
    assert isinstance(restored.get_node(planner.id), PlannerDecisionNode)
    assert isinstance(restored.get_node(execution.id), AgentExecutionNode)
    assert isinstance(restored.get_node(tool_call.id), ToolCallNode)
    assert isinstance(restored.get_node(result.id), StageResultNode)
    assert restored.get_edge("process-edge-2").edge_type == AttackProcessEdgeType.CALLED_TOOL
    assert restored.neighbors(execution.id, AttackProcessEdgeType.CALLED_TOOL, direction="out")[0].id == tool_call.id
    assert restored.by_subject_ref(GraphRef(graph="kg", ref_id="host-1", ref_type="Host"))[0].id == planner.id


def test_attack_graph_finds_and_sorts_process_nodes() -> None:
    graph = AttackGraph()
    graph.add_process_node(
        id="process-planner-1",
        node_type=AttackProcessNodeType.PLANNER_DECISION,
        label="Planner selected recon",
        operation_id="op-1",
        cycle_index=1,
        agent_name="PlannerAgent",
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    graph.add_process_node(
        id="process-agent-1",
        node_type=AttackProcessNodeType.AGENT_EXECUTION,
        label="Recon agent execution",
        operation_id="op-1",
        cycle_index=1,
        agent_name="ReconAgent",
        created_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    graph.add_process_node(
        id="process-agent-2",
        node_type=AttackProcessNodeType.AGENT_EXECUTION,
        label="Exploit validation agent execution",
        operation_id="op-2",
        cycle_index=2,
        agent_name="ExploitValidationAgent",
        created_at=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    )

    assert [node.id for node in graph.find_process_nodes(operation_id="op-1")] == [
        "process-planner-1",
        "process-agent-1",
    ]
    assert [node.id for node in graph.find_process_nodes(cycle_index=1, agent_name="ReconAgent")] == [
        "process-agent-1",
    ]
    assert [node.id for node in graph.find_process_nodes(node_type=AttackProcessNodeType.AGENT_EXECUTION)] == [
        "process-agent-1",
        "process-agent-2",
    ]
    assert [node.id for node in graph.recent_process_nodes(limit=2)] == [
        "process-agent-2",
        "process-agent-1",
    ]
