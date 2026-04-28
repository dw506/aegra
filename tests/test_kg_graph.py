from __future__ import annotations

import pytest

from kg.enums import EdgeType
from kg.exceptions import DuplicateEntityError, ValidationConstraintError
from kg.graph import KnowledgeGraph
from kg.models import (
    AppliesToHostEdge,
    CanReachEdge,
    DerivedFromEdge,
    Evidence,
    Finding,
    Goal,
    Host,
    HostsEdge,
    Identity,
    IdentityPresentOnEdge,
    Observation,
    ObservedOnEdge,
    PrivilegeState,
    Service,
    Session,
    SessionOnEdge,
    SupportedByEdge,
    TargetsEdge,
)


def build_sample_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_node(Host(id="host-1", label="Gateway", hostname="gateway.local", confidence=0.95))
    graph.add_node(Service(id="svc-1", label="SSH", service_name="ssh", port=22, protocol="tcp"))
    graph.add_node(Identity(id="id-1", label="Operator", username="operator"))
    graph.add_node(Session(id="sess-1", label="Operator Session", session_kind="console"))
    graph.add_node(
        PrivilegeState(
            id="priv-1",
            label="Local Admin",
            privilege_level="admin",
            confidence=0.9,
        )
    )
    graph.add_node(Observation(id="obs-1", label="Port Scan", observation_kind="scan"))
    graph.add_node(Evidence(id="ev-1", label="Scan Output", evidence_kind="text"))
    graph.add_node(Finding(id="finding-1", label="SSH Exposed", finding_kind="exposure"))
    graph.add_node(Goal(id="goal-1", label="Validate Exposure", category="validation"))

    graph.add_edge(HostsEdge(id="edge-hosts", label="hosts", source="host-1", target="svc-1"))
    graph.add_edge(
        IdentityPresentOnEdge(
            id="edge-identity-host",
            label="identity_present_on",
            source="id-1",
            target="host-1",
        )
    )
    graph.add_edge(
        SessionOnEdge(id="edge-session-host", label="session_on", source="sess-1", target="host-1")
    )
    graph.add_edge(
        AppliesToHostEdge(
            id="edge-priv-host",
            label="applies_to_host",
            source="priv-1",
            target="host-1",
        )
    )
    graph.add_edge(
        ObservedOnEdge(
            id="edge-obs-host",
            label="observed_on",
            source="obs-1",
            target="host-1",
        )
    )
    graph.add_edge(
        DerivedFromEdge(
            id="edge-ev-obs",
            label="derived_from",
            source="ev-1",
            target="obs-1",
        )
    )
    graph.add_edge(
        TargetsEdge(id="edge-goal-finding", label="targets", source="goal-1", target="finding-1")
    )
    graph.add_edge(
        CanReachEdge(id="edge-reach", label="can_reach", source="host-1", target="svc-1")
    )
    graph.link_supported_by("finding-1", "ev-1")
    return graph


def test_add_and_query_core_entities() -> None:
    graph = build_sample_graph()

    assert [node.id for node in graph.get_hosts()] == ["host-1"]
    assert [node.id for node in graph.get_services_for_host("host-1")] == ["svc-1"]
    assert [node.id for node in graph.get_identities_on_host("host-1")] == ["id-1"]
    assert [node.id for node in graph.get_sessions_on_host("host-1")] == ["sess-1"]
    assert [node.id for node in graph.get_privilege_states_for_host("host-1")] == ["priv-1"]
    assert [node.id for node in graph.get_goal_related_entities("goal-1")] == ["finding-1"]
    assert [node.id for node in graph.get_reachable_targets("host-1")] == ["svc-1"]
    high_confidence_ids = {node.id for node in graph.get_entities_by_confidence(0.9)}
    assert {"host-1", "priv-1"}.issubset(high_confidence_ids)


def test_evidence_traceability_helpers() -> None:
    graph = build_sample_graph()

    evidence = graph.get_supporting_evidence("finding-1")
    chain = graph.get_support_chain("finding-1", max_depth=3)
    observations = graph.get_observations_for_entity("finding-1")
    subjects = graph.get_entities_supported_by_evidence("ev-1")

    assert [node.id for node in evidence] == ["ev-1"]
    assert any([ref.entity_id for ref in path] == ["finding-1", "ev-1", "obs-1"] for path in chain)
    assert [node.id for node in observations] == ["obs-1"]
    assert [node.id for node in subjects] == ["finding-1"]


def test_edge_constraints_are_enforced() -> None:
    graph = KnowledgeGraph()
    graph.add_node(Host(id="host-1", label="Gateway"))
    graph.add_node(Host(id="host-2", label="Target"))
    graph.add_node(Evidence(id="ev-1", label="Packet Capture"))

    with pytest.raises(ValidationConstraintError):
        graph.add_edge(
            ObservedOnEdge(
                id="edge-invalid-observed",
                label="observed_on",
                source="host-1",
                target="host-2",
            )
        )

    with pytest.raises(ValidationConstraintError):
        graph.add_edge(
            SupportedByEdge(
                id="edge-invalid-supported",
                label="supported_by",
                source="host-1",
                target="host-2",
            )
        )

    with pytest.raises(ValidationConstraintError):
        graph.link_supported_by("host-1", "host-2")


def test_duplicate_ids_and_roundtrip_serialization() -> None:
    graph = KnowledgeGraph()
    graph.add_node(Host(id="host-1", label="Gateway"))

    with pytest.raises(DuplicateEntityError):
        graph.add_node(Host(id="host-1", label="Gateway Copy"))

    payload = build_sample_graph().to_dict()
    restored = KnowledgeGraph.from_dict(payload)

    assert restored.get_node("host-1").label == "Gateway"
    assert restored.get_edge("edge-hosts").type == EdgeType.HOSTS


def test_subgraph_contains_only_selected_nodes() -> None:
    graph = build_sample_graph()
    subgraph = graph.subgraph({"host-1", "svc-1"})

    assert [node.id for node in subgraph.list_nodes()] == ["host-1", "svc-1"]
    assert [edge.id for edge in subgraph.list_edges()] == ["edge-hosts", "edge-reach"]


def test_upsert_observation_and_evidence_update_existing_nodes() -> None:
    graph = KnowledgeGraph()
    graph.upsert_observation({"id": "obs-1", "label": "Scan", "confidence": 0.5})
    graph.upsert_observation({"id": "obs-1", "label": "Scan Updated", "confidence": 0.8})
    graph.upsert_evidence({"id": "ev-1", "label": "Artifact", "confidence": 0.4})
    graph.upsert_evidence({"id": "ev-1", "label": "Artifact Updated", "confidence": 0.7})

    assert graph.get_node("obs-1").label == "Scan Updated"
    assert graph.get_node("ev-1").label == "Artifact Updated"
