from __future__ import annotations

from src.core.models.ag import (
    AGEdgeType,
    ActionNode,
    ActionNodeType,
    ActivationStatus,
    AttackGraph,
    BaseAGEdge,
    GraphRef,
    StateNode,
    StateNodeType,
    TruthStatus,
)


def test_state_and_action_node_creation() -> None:
    state = StateNode(
        id="state-1",
        node_type=StateNodeType.HOST_KNOWN,
        label="Host known",
        subject_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        truth_status=TruthStatus.ACTIVE,
        confidence=0.8,
        goal_relevance=0.5,
        created_from=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )
    action = ActionNode(
        id="action-1",
        action_type=ActionNodeType.ENUMERATE_HOST,
        label="Enumerate host",
        bound_args={"host_id": "host-1"},
        activation_status=ActivationStatus.ACTIVATABLE,
    )

    assert state.node_type == StateNodeType.HOST_KNOWN
    assert action.action_type == ActionNodeType.ENUMERATE_HOST


def test_attack_graph_adds_and_serializes_nodes() -> None:
    graph = AttackGraph()
    state = StateNode(
        id="state-1",
        node_type=StateNodeType.HOST_KNOWN,
        label="Host known",
        subject_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        truth_status=TruthStatus.ACTIVE,
        confidence=0.8,
        goal_relevance=0.5,
        created_from=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )
    action = ActionNode(
        id="action-1",
        action_type=ActionNodeType.ENUMERATE_HOST,
        label="Enumerate host",
        bound_args={"host_id": "host-1"},
        activation_status=ActivationStatus.ACTIVATABLE,
    )
    edge = BaseAGEdge(
        id="edge-1",
        edge_type=AGEdgeType.REQUIRES,
        source="state-1",
        target="action-1",
        label="requires",
    )

    graph.add_node(state)
    graph.add_node(action)
    graph.add_edge(edge)
    restored = AttackGraph.from_dict(graph.to_dict())

    assert restored.get_node("state-1").id == "state-1"
    assert restored.get_edge("edge-1").edge_type == AGEdgeType.REQUIRES
    assert restored.by_subject_ref(GraphRef(graph="kg", ref_id="host-1", ref_type="Host"))[0].id == "state-1"
