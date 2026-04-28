from __future__ import annotations

from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import AttackGraphTaskBuilder, TaskCandidate, TaskGenerationRequest, TaskGraphBuilder
from src.core.models.ag import ActionNodeType, GraphRef
from src.core.models.kg import Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.models.tg import DecisionNode, DependencyType, TaskGroupType, TaskType


def build_sample_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(Service(id="svc-1", label="SSH", confidence=0.8))
    kg.add_node(Goal(id="goal-1", label="Validate Objective", category="data", confidence=0.9))
    kg.add_edge(HostsEdge(id="e-host-svc", label="hosts", source="host-1", target="svc-1"))
    kg.add_edge(TargetsEdge(id="e-goal-svc", label="targets", source="goal-1", target="svc-1"))
    return kg


def test_ag_to_tg_builds_stable_tasks_and_dependencies() -> None:
    ag = AttackGraphProjector().project(build_sample_kg())
    builder = AttackGraphTaskBuilder()
    enumerate_action = ag.find_actions(ActionNodeType.ENUMERATE_HOST)[0]
    session_action = ag.find_actions(ActionNodeType.ESTABLISH_MANAGED_SESSION)[0]

    request = TaskGenerationRequest(
        action_ids=[enumerate_action.id, session_action.id],
        include_blocked=True,
    )
    result_one = builder.build_candidates(ag, request)
    result_two = builder.build_candidates(ag, request)

    graph_one = result_one.task_graph
    graph_two = result_two.task_graph

    assert graph_one is not None
    assert graph_two is not None
    assert graph_one["nodes"] == graph_two["nodes"]
    assert graph_one["edges"] == graph_two["edges"]
    assert any(edge["dependency_type"] == DependencyType.DEPENDS_ON.value for edge in graph_one["edges"])
    assert result_one.source_ag_version == ag.version
    assert result_one.tg_version == graph_one["metadata"]["version"]
    assert result_one.frontier_version == graph_one["metadata"]["frontier_version"]


def test_builder_creates_evidence_tasks_and_default_filters_blocked() -> None:
    ag = AttackGraphProjector().project(
        build_sample_kg(),
        policy_context={
            "constraints": [
                {
                    "constraint_type": "APPROVAL_GATE",
                    "label": "Approval gate",
                    "properties": {"approved": False},
                }
            ]
        },
    )
    builder = AttackGraphTaskBuilder()
    blocked_action = ag.find_actions(ActionNodeType.ESTABLISH_MANAGED_SESSION)[0]

    filtered = builder.build_candidates(ag, TaskGenerationRequest(action_ids=[blocked_action.id]))
    included = builder.build_candidates(
        ag,
        TaskGenerationRequest(action_ids=[blocked_action.id], include_blocked=True),
    )

    assert filtered.candidates == []
    assert included.candidates[0].task_type == TaskType.IDENTITY_CONTEXT_CONFIRMATION
    assert included.task_graph is not None
    evidence_nodes = [
        node for node in included.task_graph["nodes"] if node["task_type"] == TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL.value
    ]
    assert evidence_nodes


def test_task_graph_builder_builds_from_candidates_with_stable_ids() -> None:
    builder = TaskGraphBuilder()
    candidate = TaskCandidate(
        source_action_id="action-1",
        task_type=TaskType.SERVICE_VALIDATION,
        input_bindings={"host_id": "host-1", "service_id": "svc-1"},
        target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
        precondition_refs=[],
        estimated_cost=0.2,
        estimated_risk=0.1,
        estimated_noise=0.1,
        goal_relevance=0.8,
        resource_keys={"service:svc-1"},
        parallelizable=True,
        approval_required=False,
    )

    req = TaskGenerationRequest(candidates=[candidate], group_type=TaskGroupType.STAGE, group_label="Stage 1")
    result_one = builder.build_from_candidates(req)
    result_two = builder.build_from_candidates(req)

    assert result_one.task_graph is not None
    assert result_one.task_graph["nodes"] == result_two.task_graph["nodes"]
    assert result_one.validation_errors == []


def test_task_graph_builder_links_alternatives_and_attaches_decision() -> None:
    builder = TaskGraphBuilder()
    left = TaskCandidate(
        source_action_id="action-1",
        task_type=TaskType.ASSET_CONFIRMATION,
        input_bindings={"host_id": "host-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        goal_relevance=0.8,
        resource_keys={"host:host-1"},
        parallelizable=False,
        approval_required=False,
    )
    right = TaskCandidate(
        source_action_id="action-2",
        task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION,
        input_bindings={"host_id": "host-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        goal_relevance=0.75,
        resource_keys={"host:host-1"},
        parallelizable=False,
        approval_required=False,
    )
    preview = builder.build_from_candidates(TaskGenerationRequest(candidates=[left, right], include_evidence_tasks=False))
    assert preview.task_graph is not None
    task_ids = [
        node["id"]
        for node in preview.task_graph["nodes"]
        if node.get("kind") == "task"
    ]
    decision = DecisionNode(
        id="decision-1",
        label="Choose branch",
        decision_type="planner_choice",
        option_task_ids=task_ids,
    )
    final = builder.build_from_candidates(
        TaskGenerationRequest(candidates=[left, right], include_evidence_tasks=False, decision_node=decision)
    )

    assert final.task_graph is not None
    edge_types = {edge["dependency_type"] for edge in final.task_graph["edges"]}
    assert DependencyType.ALTERNATIVE_TO.value in edge_types
    assert any(node.get("kind") == "decision" for node in final.task_graph["nodes"])


def test_task_graph_builder_accepts_nested_input_bindings() -> None:
    builder = TaskGraphBuilder()
    candidate = TaskCandidate(
        source_action_id="action-nested",
        task_type=TaskType.SERVICE_VALIDATION,
        input_bindings={"host": {"id": "host-1"}, "service": {"id": "svc-1", "port": 22}},
        target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
        goal_relevance=0.7,
        resource_keys={"service:svc-1"},
        parallelizable=True,
        approval_required=False,
    )

    result = builder.build_from_candidates(TaskGenerationRequest(candidates=[candidate], include_evidence_tasks=False))

    assert result.validation_errors == []


def test_task_graph_builder_serializes_same_host_and_allows_cross_host_parallelism() -> None:
    builder = TaskGraphBuilder()
    left = TaskCandidate(
        source_action_id="action-host-1-a",
        task_type=TaskType.ASSET_CONFIRMATION,
        input_bindings={"host_id": "host-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        parallelizable=True,
    )
    right = TaskCandidate(
        source_action_id="action-host-1-b",
        task_type=TaskType.SERVICE_VALIDATION,
        input_bindings={"host_id": "host-1", "service_id": "svc-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        parallelizable=True,
    )
    remote = TaskCandidate(
        source_action_id="action-host-2",
        task_type=TaskType.SERVICE_VALIDATION,
        input_bindings={"host_id": "host-2", "service_id": "svc-2"},
        target_refs=[GraphRef(graph="kg", ref_id="host-2", ref_type="Host")],
        parallelizable=True,
    )

    result = builder.build_from_candidates(
        TaskGenerationRequest(candidates=[left, right, remote], include_evidence_tasks=False)
    )

    assert result.task_graph is not None
    conflict_pairs = {
        (edge["source"], edge["target"])
        for edge in result.task_graph["edges"]
        if edge["dependency_type"] == DependencyType.CONFLICTS_WITH.value
    }
    same_host_tasks = [
        node["id"] for node in result.task_graph["nodes"] if node.get("source_action_id") in {"action-host-1-a", "action-host-1-b"}
    ]
    remote_task = next(node["id"] for node in result.task_graph["nodes"] if node.get("source_action_id") == "action-host-2")
    assert len(same_host_tasks) == 2
    assert (same_host_tasks[0], same_host_tasks[1]) in conflict_pairs or (same_host_tasks[1], same_host_tasks[0]) in conflict_pairs
    assert not any(remote_task in pair and any(task_id in pair for task_id in same_host_tasks) for pair in conflict_pairs)


def test_task_graph_builder_adds_stage_dependencies_and_locks_for_credentials_and_sessions() -> None:
    builder = TaskGraphBuilder()
    discovery = TaskCandidate(
        source_action_id="action-discovery",
        task_type=TaskType.SERVICE_VALIDATION,
        input_bindings={"host_id": "host-2", "service_id": "svc-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-2", ref_type="Host")],
        parallelizable=True,
    )
    lateral = TaskCandidate(
        source_action_id="action-lateral",
        task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION,
        input_bindings={"host_id": "host-2", "credential_id": "cred-1", "session_id": "sess-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-2", ref_type="Host")],
        parallelizable=True,
    )
    privilege = TaskCandidate(
        source_action_id="action-priv",
        task_type=TaskType.PRIVILEGE_CONFIGURATION_VALIDATION,
        input_bindings={"host_id": "host-2", "session_id": "sess-1"},
        target_refs=[GraphRef(graph="kg", ref_id="host-2", ref_type="Host")],
        parallelizable=True,
    )

    result = builder.build_from_candidates(
        TaskGenerationRequest(candidates=[discovery, lateral, privilege], include_evidence_tasks=False)
    )

    assert result.task_graph is not None
    nodes = {node["source_action_id"]: node for node in result.task_graph["nodes"] if node.get("kind") == "task"}
    assert "credential:cred-1" in nodes["action-lateral"]["resource_keys"]
    assert "session:sess-1" in nodes["action-lateral"]["resource_keys"]
    depends = [
        edge
        for edge in result.task_graph["edges"]
        if edge["dependency_type"] == DependencyType.DEPENDS_ON.value
    ]
    assert any(edge["source"] == nodes["action-discovery"]["id"] and edge["target"] == nodes["action-lateral"]["id"] for edge in depends)
    assert any(edge["source"] == nodes["action-lateral"]["id"] and edge["target"] == nodes["action-priv"]["id"] for edge in depends)
