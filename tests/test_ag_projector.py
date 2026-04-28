from __future__ import annotations

from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import ActionNodeType, ConstraintNodeType, GoalNode, StateNodeType
from src.core.models.kg import (
    AuthenticatesAsEdge,
    CanReachEdge,
    Credential,
    DataAsset,
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
