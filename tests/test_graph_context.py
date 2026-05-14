from __future__ import annotations

from src.core.agents.graph_context import GraphContextBuilder, GraphContextBuilderConfig
from src.core.models.ag import (
    ActivationStatus,
    ActionNode,
    ActionNodeType,
    AttackGraph,
    BaseAGEdge,
    AGEdgeType,
    GoalNode,
    GoalNodeType,
    GraphRef,
    StateNode,
    StateNodeType,
    TruthStatus,
)
from src.core.models.runtime import (
    OperationRuntime,
    OutcomeCacheEntry,
    RuntimeEventRef,
    RuntimeState,
    RuntimeStatus,
    TaskRuntime,
    TaskRuntimeStatus,
)
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType


def build_attack_graph() -> AttackGraph:
    graph = AttackGraph()
    host_ref = GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="127.0.0.1")
    service_ref = GraphRef(graph="kg", ref_id="svc-1", ref_type="Service", label="http:8080")
    state = StateNode(
        id="state-service",
        label="HTTP service known",
        node_type=StateNodeType.SERVICE_KNOWN,
        truth_status=TruthStatus.ACTIVE,
        confidence=0.9,
        goal_relevance=0.8,
        subject_refs=[service_ref],
        properties={
            "host": "127.0.0.1",
            "port": "8080",
            "protocol": "http",
            "service": "http",
            "version": "Werkzeug",
            "raw_output": "SHOULD_NOT_APPEAR",
        },
    )
    action = ActionNode(
        id="action-validate-service",
        label="Validate HTTP service",
        action_type=ActionNodeType.VALIDATE_SERVICE,
        activation_status=ActivationStatus.ACTIVATABLE,
        expected_value=0.7,
        risk=0.1,
        noise=0.1,
        source_refs=[host_ref, service_ref],
        required_capabilities={"http_probe"},
        resource_keys={"host:127.0.0.1"},
        properties={"note": "safe validation", "stdout": "SHOULD_NOT_APPEAR"},
    )
    goal = GoalNode(
        id="goal-objective",
        label="Find objective",
        goal_type=GoalNodeType.OBJECTIVE_SATISFIED,
        priority=90,
        business_value=0.9,
        scope_refs=[host_ref],
        success_criteria={"flag": "present"},
    )
    graph.add_node(state)
    graph.add_node(action)
    graph.add_node(goal)
    graph.add_edge(
        BaseAGEdge(
            id="edge-action-state",
            edge_type=AGEdgeType.REQUIRES,
            source=action.id,
            target=state.id,
            label="requires",
        )
    )
    return graph


def build_runtime_state() -> RuntimeState:
    state = RuntimeState(
        operation_id="op-graph-context",
        operation_status=RuntimeStatus.RUNNING,
        execution=OperationRuntime(
            operation_id="op-graph-context",
            status=RuntimeStatus.RUNNING,
            active_goal_id="goal-objective",
        ),
    )
    state.register_task(
        TaskRuntime(
            task_id="task-ok",
            tg_node_id="tg-task-ok",
            status=TaskRuntimeStatus.SUCCEEDED,
            attempt_count=1,
            last_outcome_ref="outcome://ok",
            metadata={"summary": "completed", "raw_output": "SHOULD_NOT_APPEAR"},
        )
    )
    state.register_task(
        TaskRuntime(
            task_id="task-failed",
            tg_node_id="tg-task-failed",
            status=TaskRuntimeStatus.FAILED,
            attempt_count=1,
            max_attempts=2,
            last_error="curl returned 403 with a very long body " + ("x" * 500),
            metadata={"reason": "forbidden"},
        )
    )
    state.push_event(
        RuntimeEventRef(
            event_id="event-1",
            event_type="tool_output",
            summary="nmap found tcp/8080 open",
            payload_ref="artifact://nmap-1",
            metadata={"stdout": "SHOULD_NOT_APPEAR", "tool": "nmap"},
        )
    )
    state.record_outcome(
        OutcomeCacheEntry(
            outcome_id="outcome-1",
            task_id="task-ok",
            outcome_type="service_observed",
            summary="HTTP service responded with status 200",
            payload_ref="artifact://curl-1",
            metadata={"response_body": "SHOULD_NOT_APPEAR", "status": 200},
        )
    )
    return state


def build_task_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="tg-task-ok",
            label="Validate HTTP",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.SUCCEEDED,
            target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
            source_action_id="action-validate-service",
        )
    )
    return graph


def test_graph_context_builder_extracts_compact_ag_and_runtime_slice() -> None:
    context = GraphContextBuilder().build(
        attack_graph=build_attack_graph(),
        runtime_state=build_runtime_state(),
        task_graph=build_task_graph(),
        policy_context={
            "authorized_hosts": ["127.0.0.1"],
            "cidr_whitelist": ["127.0.0.1/32"],
            "disabled_tools": ["sqlmap"],
            "command_allowlist": ["nmap"],
            "risk_policy": {"allow_safe_probe": True},
            "raw_output": "SHOULD_NOT_APPEAR",
        },
    )

    payload = context.model_dump(mode="json")

    assert payload["operation_id"] == "op-graph-context"
    assert payload["goals"][0]["ref"]["ref_id"] == "goal-objective"
    assert payload["known_services"][0]["port"] == 8080
    assert payload["known_services"][0]["service_name"] == "http"
    assert payload["frontier_actions"][0]["action_type"] == "VALIDATE_SERVICE"
    assert payload["tasks_by_status"]["succeeded"][0]["task_type"] == "SERVICE_VALIDATION"
    assert payload["tasks_by_status"]["failed"][0]["last_error"].endswith("[truncated]")
    assert payload["evidence"][0]["payload_ref"] == "artifact://nmap-1"
    assert payload["policy"]["authorized_hosts"] == ["127.0.0.1"]
    assert payload["context_stats"]["large_artifacts_included"] is False

    serialized = str(payload)
    assert "SHOULD_NOT_APPEAR" not in serialized
    assert "raw_output" not in serialized
    assert "stdout" not in serialized
    assert "response_body" not in serialized


def test_graph_context_builder_respects_item_limits() -> None:
    runtime_state = build_runtime_state()
    for index in range(5):
        runtime_state.record_outcome(
            OutcomeCacheEntry(
                outcome_id=f"outcome-extra-{index}",
                task_id="task-ok",
                outcome_type="extra",
                summary=f"extra summary {index}",
                payload_ref=f"artifact://extra-{index}",
            )
        )

    context = GraphContextBuilder(
        GraphContextBuilderConfig(max_evidence_items=2, max_tasks_per_status=1)
    ).build(
        attack_graph=build_attack_graph(),
        runtime_state=runtime_state,
        task_graph=build_task_graph(),
    )

    assert len(context.evidence) == 2
    assert len(context.tasks_by_status["failed"]) == 1
    assert len(context.tasks_by_status["succeeded"]) == 1
