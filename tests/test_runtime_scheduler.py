from __future__ import annotations

from datetime import timedelta

from src.core.models.ag import GraphRef
from src.core.models.runtime import (
    CredentialKind,
    CredentialRuntime,
    CredentialStatus,
    LockStatus,
    OperationRuntime,
    PivotRouteRuntime,
    PivotRouteStatus,
    ResourceLock,
    RuntimeState,
    SessionLeaseRuntime,
    SessionRuntime,
    SessionStatus,
    TaskRuntime,
    TaskRuntimeStatus,
    WorkerRuntime,
    WorkerStatus,
    utc_now,
)
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.runtime.scheduler import RuntimeScheduler


def build_runtime_state() -> RuntimeState:
    return RuntimeState(
        operation_id="op-1",
        execution=OperationRuntime(operation_id="op-1"),
    )


def build_task_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-1",
            label="Service validation",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.READY,
            source_action_id="action-1",
            input_bindings={"host_id": "host-1"},
            target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            resource_keys={"host:host-1"},
            estimated_noise=0.1,
            estimated_risk=0.1,
        )
    )
    graph.add_node(
        TaskNode(
            id="task-2",
            label="Goal validation",
            task_type=TaskType.GOAL_CONDITION_VALIDATION,
            status=TaskStatus.DRAFT,
            source_action_id="action-2",
            input_bindings={},
            target_refs=[GraphRef(graph="kg", ref_id="goal-1", ref_type="Goal")],
            estimated_noise=0.1,
            estimated_risk=0.1,
            gate_ids={"approval_gate"},
        )
    )
    return graph


def test_select_schedulable_tasks_only_returns_runtime_admissible_ready_tasks() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()

    selected = scheduler.select_schedulable_tasks(graph, runtime_state)

    assert selected == ["task-1"]


def test_assign_tasks_returns_assignment_decision_for_idle_worker() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()

    decisions = scheduler.assign_tasks(graph, runtime_state, ["task-1"])

    assert len(decisions) == 1
    assert decisions[0].accepted is True
    assert decisions[0].worker_id == "worker-1"


def test_tick_blocks_task_when_runtime_lock_conflicts() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()

    runtime_state.locks["host:host-1"] = ResourceLock(
        lock_key="host:host-1",
        owner_type="task",
        owner_id="other-task",
        status=LockStatus.ACTIVE,
        acquired_at=utc_now(),
    )

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids


def test_scheduler_enforces_per_host_concurrency_limit() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"max_concurrent_per_host": {"default": 1}}
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()
    runtime_state.register_task(
        TaskRuntime(
            task_id="other-task",
            tg_node_id="other-task",
            status=TaskRuntimeStatus.RUNNING,
            resource_keys={"host:host-1"},
            started_at=utc_now(),
        )
    )

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert result.decisions == []
    assert runtime_state.execution.metadata["audit_log"][-1]["reason"] == "per-host concurrency limit reached"


def test_scheduler_accepts_normalized_runtime_policy_metadata() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {
        "authorized_hosts": ["host-1"],
        "blocked_hosts": [],
        "cidr_whitelist": [],
        "default_task_timeout_sec": 900,
        "deny_egress": False,
        "loaded_at": utc_now().isoformat(),
        "loaded_from": "settings",
        "max_concurrent_per_host": {"default": 2},
        "policy_version": "v1",
        "rate_limit_per_subnet_per_min": {},
        "retry_backoff_base_sec": 0,
        "safety_stop": False,
        "sensitive_tags": [],
        "sensitive_task_types": [],
        "session_policies": {},
    }
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)

    result = scheduler.tick(build_task_graph(), runtime_state)

    assert result.selected_task_ids == ["task-1"]


def test_scheduler_enforces_subnet_rate_limit() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {
        "rate_limit_per_subnet_per_min": {"10.0.0.0/24": 1},
    }
    runtime_state.execution.metadata["subnet_rate_limit"] = {
        "10.0.0.0/24": {"window_started_at": utc_now().isoformat(), "count": 1}
    }
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()
    task = graph.get_node("task-1")
    task.input_bindings["subnet_cidr"] = "10.0.0.0/24"

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids


def test_scheduler_blocks_blacklisted_and_out_of_scope_hosts() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {
        "authorized_hosts": ["host-2"],
        "blocked_hosts": ["host-1"],
    }
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)

    result = scheduler.tick(build_task_graph(), runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids


def test_scheduler_blocks_egress_when_target_address_not_whitelisted() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {
        "cidr_whitelist": ["10.0.0.0/24"],
        "deny_egress": True,
    }
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()
    task = graph.get_node("task-1")
    task.input_bindings["target_address"] = "8.8.8.8"

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids


def test_scheduler_requires_sensitive_approval() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {
        "sensitive_task_types": [TaskType.SERVICE_VALIDATION.value],
    }
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)

    result = scheduler.tick(build_task_graph(), runtime_state)

    assert result.selected_task_ids == []
    runtime_state.budgets.approval_cache["task:task-1:approved"] = True

    result = scheduler.tick(build_task_graph(), runtime_state)

    assert result.selected_task_ids == ["task-1"]


def test_scheduler_honors_retry_backoff_window() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"retry_backoff_base_sec": 60}
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()
    graph.get_node("task-1").retry_policy.backoff_seconds = 60
    runtime_state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status=TaskRuntimeStatus.FAILED,
            attempt_count=1,
            max_attempts=2,
            finished_at=utc_now(),
        )
    )

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids


def test_scheduler_times_out_long_running_task_and_releases_owned_locks() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"default_task_timeout_sec": 30}
    runtime_state.workers["worker-1"] = WorkerRuntime(
        worker_id="worker-1",
        status=WorkerStatus.BUSY,
        current_task_id="task-1",
    )
    runtime_state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status=TaskRuntimeStatus.RUNNING,
            assigned_worker="worker-1",
            started_at=utc_now() - timedelta(seconds=40),
        )
    )
    runtime_state.locks["host:host-1"] = ResourceLock(
        lock_key="host:host-1",
        owner_type="task",
        owner_id="task-1",
        status=LockStatus.ACTIVE,
        acquired_at=utc_now() - timedelta(seconds=40),
    )

    scheduler.tick(build_task_graph(), runtime_state)

    assert runtime_state.execution.tasks["task-1"].status == TaskRuntimeStatus.TIMED_OUT
    assert runtime_state.locks["host:host-1"].status == LockStatus.RELEASED
    assert runtime_state.workers["worker-1"].status == WorkerStatus.IDLE


def test_scheduler_respects_session_exclusive_shared_policy() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    graph = build_task_graph()
    task = graph.get_node("task-1")
    task.resource_keys.add("session:sess-1")
    task.input_bindings["session_id"] = "sess-1"
    task.input_bindings["session_policy"] = "exclusive"
    runtime_state.locks["session-shared:sess-1"] = ResourceLock(
        lock_key="session-shared:sess-1",
        owner_type="task",
        owner_id="other-task",
        status=LockStatus.ACTIVE,
        acquired_at=utc_now(),
    )

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids


def test_scheduler_expires_session_family_before_admission() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    expired_session = SessionRuntime(
        session_id="sess-1",
        status=SessionStatus.ACTIVE,
        heartbeat_at=utc_now(),
    )
    expired_session.lease_expiry = expired_session.heartbeat_at - timedelta(seconds=1)
    runtime_state.sessions["sess-1"] = expired_session
    runtime_state.session_leases["lease-1"] = SessionLeaseRuntime(
        lease_id="lease-1",
        session_id="sess-1",
        owner_task_id="task-old",
    )
    runtime_state.credentials["cred-1"] = CredentialRuntime(
        credential_id="cred-1",
        principal="alice",
        kind=CredentialKind.TOKEN,
        status=CredentialStatus.VALID,
        source_session_id="sess-1",
    )
    runtime_state.pivot_routes["route-1"] = PivotRouteRuntime(
        route_id="route-1",
        destination_host="host-1",
        session_id="sess-1",
        status=PivotRouteStatus.ACTIVE,
    )

    scheduler.tick(build_task_graph(), runtime_state)

    assert runtime_state.sessions["sess-1"].status == SessionStatus.EXPIRED
    assert runtime_state.session_leases["lease-1"].metadata["release_reason"] == "lease_expired"
    assert runtime_state.credentials["cred-1"].status == CredentialStatus.EXPIRED
    assert runtime_state.pivot_routes["route-1"].status == PivotRouteStatus.FAILED


def test_scheduler_rejects_task_with_invalid_bound_credential() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    runtime_state.credentials["cred-1"] = CredentialRuntime(
        credential_id="cred-1",
        principal="alice",
        kind=CredentialKind.TOKEN,
        status=CredentialStatus.INVALID,
    )
    graph = build_task_graph()
    task = graph.get_node("task-1")
    task.input_bindings["credential_id"] = "cred-1"
    task.resource_keys.add("credential:cred-1")

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids
    assert runtime_state.execution.metadata["audit_log"][-1]["reason"] == "bound credential unavailable"


def test_scheduler_rejects_task_with_failed_bound_pivot_route() -> None:
    scheduler = RuntimeScheduler()
    runtime_state = build_runtime_state()
    runtime_state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    runtime_state.pivot_routes["route-1"] = PivotRouteRuntime(
        route_id="route-1",
        destination_host="host-1",
        source_host="host-0",
        session_id=None,
        status=PivotRouteStatus.FAILED,
    )
    graph = build_task_graph()
    task = graph.get_node("task-1")
    task.input_bindings["route_id"] = "route-1"

    result = scheduler.tick(graph, runtime_state)

    assert result.selected_task_ids == []
    assert "task-1" not in result.candidate_task_ids
    assert runtime_state.execution.metadata["audit_log"][-1]["reason"] == "pivot route unavailable"
