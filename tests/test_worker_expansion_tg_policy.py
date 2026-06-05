from __future__ import annotations

from src.core.graph.tg_builder import AttackGraphTaskBuilder, TaskCandidate, TaskGenerationRequest, TaskGraphBuilder
from src.core.models.ag import ActionNodeType, GraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.models.tg import TaskNode, TaskStatus, TaskType
from src.core.runtime.policy_gate import PolicyGate, PolicyGateAction


def test_new_action_types_map_to_new_task_types() -> None:
    assert AttackGraphTaskBuilder._map_action_to_task_type(ActionNodeType.SCAN_PORTS) == TaskType.PORT_SCAN
    assert (
        AttackGraphTaskBuilder._map_action_to_task_type(ActionNodeType.FINGERPRINT_INTERNAL_SERVICE)
        == TaskType.INTERNAL_SERVICE_FINGERPRINT
    )
    assert AttackGraphTaskBuilder._map_action_to_task_type(ActionNodeType.DISCOVER_WEB_PATHS) == TaskType.WEB_DISCOVERY
    assert AttackGraphTaskBuilder._map_action_to_task_type(ActionNodeType.VALIDATE_CREDENTIAL) == TaskType.CREDENTIAL_VALIDATION
    assert (
        AttackGraphTaskBuilder._map_action_to_task_type(ActionNodeType.CHECK_CREDENTIAL_REUSE)
        == TaskType.CREDENTIAL_REUSE_VALIDATION
    )
    assert (
        AttackGraphTaskBuilder._map_action_to_task_type(ActionNodeType.VALIDATE_LATERAL_REACHABILITY)
        == TaskType.LATERAL_REACHABILITY_VALIDATION
    )


def test_resource_key_normalization_covers_service_route_and_cidr() -> None:
    candidate = AttackGraphTaskBuilder._normalize_candidate(
        TaskCandidate(
            source_action_id="action-1",
            task_type=TaskType.CREDENTIAL_REUSE_VALIDATION,
            input_bindings={
                "source_host_id": "host-1",
                "target_host_id": "host-2",
                "service_id": "svc-1",
                "credential_id": "cred-1",
                "session_id": "sess-1",
                "route_id": "route-1",
                "target_cidr": "10.10.0.0/24",
            },
            target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
        )
    )

    assert {
        "host:host-1",
        "host:host-2",
        "service:svc-1",
        "credential:cred-1",
        "session:sess-1",
        "route:route-1",
        "subnet:10.10.0.0/24",
    } <= candidate.resource_keys


def test_stage_dependencies_order_new_worker_tasks() -> None:
    request = TaskGenerationRequest(
        candidates=[
            TaskCandidate(source_action_id="scan", task_type=TaskType.PORT_SCAN, input_bindings={"host_id": "host-1"}),
            TaskCandidate(
                source_action_id="fingerprint",
                task_type=TaskType.INTERNAL_SERVICE_FINGERPRINT,
                input_bindings={"host_id": "host-1", "service_id": "svc-1", "route_id": "route-1"},
            ),
            TaskCandidate(
                source_action_id="web",
                task_type=TaskType.WEB_DISCOVERY,
                input_bindings={"host_id": "host-1", "service_id": "svc-1"},
            ),
            TaskCandidate(
                source_action_id="reuse",
                task_type=TaskType.CREDENTIAL_REUSE_VALIDATION,
                input_bindings={"host_id": "host-1", "credential_id": "cred-1"},
            ),
        ],
        include_evidence_tasks=False,
    )

    result = TaskGraphBuilder().build_from_candidates(request)
    graph = result.task_graph or {}
    edges = graph["edges"]
    nodes = {node["source_action_id"]: node["id"] for node in graph["nodes"] if node["kind"] == "task"}

    assert any(edge["source"] == nodes["scan"] and edge["target"] == nodes["fingerprint"] for edge in edges)
    assert any(edge["source"] == nodes["fingerprint"] and edge["target"] == nodes["web"] for edge in edges)
    assert any(edge["source"] == nodes["web"] and edge["target"] == nodes["reuse"] for edge in edges)


def test_policy_gate_requires_approval_for_sensitive_new_tasks() -> None:
    task = TaskNode(
        id="task-1",
        label="Credential reuse",
        task_type=TaskType.CREDENTIAL_REUSE_VALIDATION,
        status=TaskStatus.READY,
        input_bindings={"credential_id": "cred-1", "host_id": "host-2"},
    )
    state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))

    decision = PolicyGate().evaluate(task, runtime_state=state)

    assert decision.action == PolicyGateAction.ALLOW
    assert decision.metadata["policy_audit_only"] is True
    assert decision.metadata["original_action"] == PolicyGateAction.NEED_APPROVAL.value
