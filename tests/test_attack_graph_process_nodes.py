from __future__ import annotations

from datetime import datetime, timezone

from src.core.models.ag import AttackGraph, GraphRef
from src.core.models.attack_process import (
    AttackProcessEdge,
    AttackProcessEdgeType,
    AttackProcessNodeType,
    AttackStepNode,
    GoalOutcomeNode,
)


def test_attack_graph_adds_and_restores_result_timeline_nodes() -> None:
    graph = AttackGraph()
    recon = AttackStepNode(
        id="attack-step-1",
        label="Recon completed",
        operation_id="op-1",
        cycle_index=1,
        agent_name="execution_agent",
        status="succeeded",
        capability="recon",
        refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    exploit = AttackStepNode(
        id="attack-step-2",
        label="Exploit completed",
        operation_id="op-1",
        cycle_index=2,
        agent_name="execution_agent",
        status="succeeded",
        capability="exploit",
        created_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    outcome = GoalOutcomeNode(
        id="goal-outcome-1",
        label="Goal completed",
        operation_id="op-1",
        cycle_index=3,
        status="success",
        properties={"achieved_level": "full"},
        created_at=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    )

    graph.add_process_node(recon)
    graph.add_process_node(exploit)
    graph.add_process_node(outcome)
    graph.add_process_edge(
        AttackProcessEdge(
            id="process-edge-1",
            edge_type=AttackProcessEdgeType.NEXT,
            source=recon.id,
            target=exploit.id,
            label="next",
        )
    )
    graph.add_process_edge(
        id="process-edge-2",
        edge_type=AttackProcessEdgeType.ADVANCED,
        source=exploit.id,
        target=outcome.id,
        label="advanced",
    )

    serialized = graph.to_dict()
    restored = AttackGraph.from_dict(serialized)

    assert {node["kind"] for node in serialized["nodes"]} == {"process"}
    assert isinstance(restored.get_node(recon.id), AttackStepNode)
    assert isinstance(restored.get_node(exploit.id), AttackStepNode)
    assert isinstance(restored.get_node(outcome.id), GoalOutcomeNode)
    assert restored.get_edge("process-edge-2").edge_type == AttackProcessEdgeType.ADVANCED
    assert restored.neighbors(recon.id, AttackProcessEdgeType.NEXT, direction="out")[0].id == exploit.id
    assert restored.by_subject_ref(GraphRef(graph="kg", ref_id="host-1", ref_type="Host"))[0].id == recon.id


def test_attack_graph_finds_and_sorts_process_nodes() -> None:
    graph = AttackGraph()
    graph.add_process_node(
        id="attack-step-1",
        node_type=AttackProcessNodeType.ATTACK_STEP,
        label="Recon completed",
        operation_id="op-1",
        cycle_index=1,
        agent_name="execution_agent",
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    graph.add_process_node(
        id="attack-step-2",
        node_type=AttackProcessNodeType.ATTACK_STEP,
        label="Exploit completed",
        operation_id="op-1",
        cycle_index=2,
        agent_name="execution_agent",
        created_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    graph.add_process_node(
        id="goal-outcome-1",
        node_type=AttackProcessNodeType.GOAL_OUTCOME,
        label="Goal outcome",
        operation_id="op-2",
        cycle_index=3,
        agent_name="execution_agent",
        created_at=datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc),
    )

    assert [node.id for node in graph.find_process_nodes(operation_id="op-1")] == [
        "attack-step-1",
        "attack-step-2",
    ]
    assert [node.id for node in graph.find_process_nodes(cycle_index=1, agent_name="execution_agent")] == [
        "attack-step-1",
    ]
    assert [node.id for node in graph.find_process_nodes(node_type=AttackProcessNodeType.ATTACK_STEP)] == [
        "attack-step-1",
        "attack-step-2",
    ]
    assert [node.id for node in graph.recent_process_nodes(limit=2)] == [
        "goal-outcome-1",
        "attack-step-2",
    ]
