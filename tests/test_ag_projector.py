from __future__ import annotations

from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import ActionNodeType, ActivationStatus, ConstraintNodeType, GoalNode, StateNodeType, TruthStatus
from src.core.models.kg import (
    AuthenticatesAsEdge,
    CanReachEdge,
    Credential,
    DataAsset,
    Evidence,
    Goal,
    Host,
    HostsEdge,
    Identity,
    IdentityAvailableOnEdge,
    PivotsToEdge,
    PrivilegeState,
    PrivilegeSourceEdge,
    ReusesCredentialEdge,
    Service,
    Session,
    SessionOnEdge,
    TargetsEdge,
)
from src.core.models.kg_enums import EntityStatus


def build_sample_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(Host(id="host-2", label="Internal", confidence=0.82))
    kg.add_node(Service(id="svc-1", label="SSH", confidence=0.8))
    kg.add_node(Identity(id="id-1", label="Operator", confidence=0.75))
    kg.add_node(Credential(id="cred-1", label="Operator Cred", confidence=0.7))
    kg.add_node(Session(id="sess-1", label="Managed Session", confidence=0.85))
    kg.add_node(PrivilegeState(id="priv-1", label="Admin", status=EntityStatus.VALIDATED, confidence=0.9))
    kg.add_node(DataAsset(id="asset-1", label="Secrets Bundle", confidence=0.8))
    kg.add_node(Goal(id="goal-1", label="Validate Data Objective", category="data", confidence=0.95))

    kg.add_edge(HostsEdge(id="e-host-svc", label="hosts", source="host-1", target="svc-1"))
    kg.add_edge(CanReachEdge(id="e-reach", label="reach", source="host-1", target="svc-1", confidence=0.7))
    kg.add_edge(
        AuthenticatesAsEdge(
            id="e-auth",
            label="auth",
            source="cred-1",
            target="id-1",
            confidence=0.8,
        )
    )
    kg.add_edge(TargetsEdge(id="e-goal-asset", label="targets", source="goal-1", target="asset-1"))
    kg.add_edge(IdentityAvailableOnEdge(id="e-identity-host", label="identity_on", source="id-1", target="host-2", confidence=0.8))
    kg.add_edge(ReusesCredentialEdge(id="e-cred-reuse", label="cred_reuse", source="cred-1", target="host-2", confidence=0.78))
    kg.add_edge(SessionOnEdge(id="e-session-host", label="session_on", source="sess-1", target="host-2", confidence=0.9))
    kg.add_edge(PrivilegeSourceEdge(id="e-priv-source", label="priv_source", source="priv-1", target="sess-1", confidence=0.88))
    kg.add_edge(PivotsToEdge(id="e-pivot", label="pivot", source="host-1", target="host-2", confidence=0.84))
    return kg


def test_full_projection_creates_expected_state_and_goal_nodes() -> None:
    projector = AttackGraphProjector()
    ag = projector.project(build_sample_kg())

    assert ag.find_states(StateNodeType.HOST_KNOWN)
    assert ag.find_states(StateNodeType.HOST_VALIDATED)
    assert ag.find_states(StateNodeType.SERVICE_KNOWN)
    assert ag.find_states(StateNodeType.PATH_CANDIDATE)
    assert ag.find_states(StateNodeType.CREDENTIAL_USABLE)
    assert ag.find_states(StateNodeType.PIVOT_HOST_AVAILABLE)
    assert ag.find_states(StateNodeType.CREDENTIAL_REUSABLE_ON_HOST)
    assert ag.find_states(StateNodeType.SESSION_ACTIVE_ON_HOST)
    assert ag.find_states(StateNodeType.IDENTITY_AVAILABLE_ON_HOST)
    assert ag.find_states(StateNodeType.PRIVILEGE_SOURCE_KNOWN)
    assert ag.find_states(StateNodeType.LATERAL_SERVICE_EXPOSED)
    assert ag.find_states(StateNodeType.GOAL_RELEVANT_DATA_LOCATED)
    assert any(isinstance(node, GoalNode) for node in ag.get_goal_nodes())


def test_same_kg_input_produces_stable_action_and_state_ids() -> None:
    projector = AttackGraphProjector()
    kg = build_sample_kg()

    ag_one = projector.project(kg)
    ag_two = projector.project(kg)

    assert [node.id for node in ag_one.find_states()] == [node.id for node in ag_two.find_states()]
    assert [node.id for node in ag_one.find_actions()] == [node.id for node in ag_two.find_actions()]


def test_projection_metadata_is_bound_to_source_kg_version() -> None:
    projector = AttackGraphProjector()
    kg = build_sample_kg()

    ag = projector.project(kg)

    assert ag.source_kg_version == kg.version
    assert ag.to_dict()["metadata"]["source_kg_version"] == kg.version
    assert ag.to_dict()["metadata"]["source_change_count"] == kg.delta.change_count


def test_action_binding_is_deduplicated() -> None:
    projector = AttackGraphProjector()
    ag = projector.project(build_sample_kg())

    validate_service_actions = ag.find_actions(ActionNodeType.VALIDATE_SERVICE)
    assert len(validate_service_actions) == 1
    assert validate_service_actions[0].bound_args == {"host_id": "host-1", "service_id": "svc-1"}
    assert ag.find_actions(ActionNodeType.ESTABLISH_PIVOT_ROUTE)
    assert ag.find_actions(ActionNodeType.REUSE_CREDENTIAL_ON_HOST)
    assert ag.find_actions(ActionNodeType.EXPLOIT_LATERAL_SERVICE)


def test_service_with_direct_evidence_ids_projects_confirmed_active_state() -> None:
    kg = KnowledgeGraph()
    evidence_ids = [f"evidence-{index}" for index in range(5)]
    kg.add_node(Host(id="kg-host::a76ee01f7c67a6db", label="target", status=EntityStatus.VALIDATED, confidence=0.9))
    kg.add_node(
        Service(
            id="kg-host::a76ee01f7c67a6db:40961/tcp",
            label="40961/tcp",
            status=EntityStatus.OBSERVED,
            confidence=0.62,
            port=40961,
            protocol="tcp",
            evidence_ids=evidence_ids,
        )
    )
    for evidence_id in evidence_ids:
        kg.add_node(Evidence(id=evidence_id, label=evidence_id, confidence=0.8))
    kg.add_edge(
        HostsEdge(
            id="hosts::kg-host::a76ee01f7c67a6db::40961",
            label="hosts",
            source="kg-host::a76ee01f7c67a6db",
            target="kg-host::a76ee01f7c67a6db:40961/tcp",
            confidence=0.9,
        )
    )

    ag = AttackGraphProjector().project(kg)

    service_known = ag.find_states(StateNodeType.SERVICE_KNOWN)[0]
    service_confirmed = ag.find_states(StateNodeType.SERVICE_CONFIRMED)[0]
    validate_service = ag.find_actions(ActionNodeType.VALIDATE_SERVICE)[0]

    assert service_known.truth_status == TruthStatus.ACTIVE
    assert service_known.properties["evidence_count"] == 5
    assert service_confirmed.truth_status == TruthStatus.ACTIVE
    assert service_confirmed.properties["evidence_count"] == 5
    assert validate_service.activation_status == ActivationStatus.ACTIVATABLE


def test_scoped_http_target_projects_web_surface_from_open_unknown_service() -> None:
    kg = KnowledgeGraph()
    kg.add_node(
        Host(
            id="kg-host::target",
            label="127.0.0.1",
            status=EntityStatus.OBSERVED,
            confidence=0.9,
            properties={"url": "http://127.0.0.1:1482", "scheme": "http", "port": 1482},
        )
    )
    kg.add_node(
        Service(
            id="kg-host::target:1482/tcp",
            label="1482/tcp",
            status=EntityStatus.OBSERVED,
            confidence=1.0,
            port=1482,
            protocol="tcp",
            service_name="miteksys-lm",
            properties={"state": "open", "target_url": "http://127.0.0.1:1482", "validated": True},
        )
    )
    kg.add_edge(
        HostsEdge(
            id="hosts::target::1482",
            label="hosts",
            source="kg-host::target",
            target="kg-host::target:1482/tcp",
            confidence=1.0,
        )
    )

    ag = AttackGraphProjector().project(kg)

    web_states = ag.find_states(StateNodeType.WEB_ATTACK_SURFACE)
    web_actions = ag.find_actions(ActionNodeType.ENUMERATE_WEB_SURFACE)
    assert web_states
    assert web_states[0].properties["target_url"] == "http://127.0.0.1:1482"
    assert web_actions
    assert web_actions[0].bound_args["target_url"] == "http://127.0.0.1:1482"


def test_policy_constraints_block_matching_actions() -> None:
    projector = AttackGraphProjector()
    ag = projector.project(
        build_sample_kg(),
        policy_context={
            "constraints": [
                {
                    "constraint_type": ConstraintNodeType.APPROVAL_GATE.value,
                    "label": "Manual approval required",
                    "hard_or_soft": "hard",
                    "properties": {"approved": False},
                }
            ]
        },
    )

    session_actions = ag.find_actions(ActionNodeType.ESTABLISH_MANAGED_SESSION)
    assert session_actions
    assert session_actions[0].activation_status.value == "blocked"
