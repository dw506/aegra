from __future__ import annotations

from datetime import timedelta

import pytest

from src.core.models.runtime import (
    OperationRuntime,
    RuntimeState,
    RuntimeStatus,
    SessionStatus,
    TaskRuntime,
    TaskRuntimeStatus,
    utc_now,
)
from src.core.runtime.budgets import BudgetExceededError, RuntimeBudgetManager
from src.core.runtime.checkpoint_store import RuntimeCheckpointManager
from src.core.runtime.events import (
    ReplanRequestedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskQueuedEvent,
    TaskStartedEvent,
    WorkerAssignedEvent,
    WorkerReleasedEvent,
)
from src.core.runtime.locks import LockConflictError, RuntimeLockManager
from src.core.runtime.reducer import RuntimeStateReducer
from src.core.runtime.session_manager import RuntimeSessionManager
from src.core.runtime.store import InMemoryRuntimeStore


def build_runtime_state(operation_id: str = "op-1") -> RuntimeState:
    return RuntimeState(
        operation_id=operation_id,
        execution=OperationRuntime(operation_id=operation_id, status=RuntimeStatus.CREATED),
    )


def test_task_events_drive_successful_state_transition() -> None:
    state = build_runtime_state()
    reducer = RuntimeStateReducer()

    state = reducer.apply_event(
        state,
        TaskQueuedEvent(operation_id="op-1", task_id="task-1", tg_node_id="tg-1"),
    )
    assert state.execution.tasks["task-1"].status == TaskRuntimeStatus.QUEUED

    state = reducer.apply_event(
        state,
        TaskStartedEvent(
            operation_id="op-1",
            task_id="task-1",
            tg_node_id="tg-1",
            worker_id="worker-1",
        ),
    )
    assert state.execution.tasks["task-1"].status == TaskRuntimeStatus.RUNNING
    assert state.execution.tasks["task-1"].started_at is not None

    state = reducer.apply_event(
        state,
        TaskCompletedEvent(
            operation_id="op-1",
            task_id="task-1",
            tg_node_id="tg-1",
            outcome_id="outcome-1",
            outcome_ref="outcome://1",
        ),
    )
    assert state.execution.tasks["task-1"].status == TaskRuntimeStatus.SUCCEEDED
    assert state.execution.tasks["task-1"].finished_at is not None
    assert state.execution.tasks["task-1"].assigned_worker is None


def test_task_failed_increments_attempt_count() -> None:
    state = build_runtime_state()
    reducer = RuntimeStateReducer()
    state.execution.tasks["task-1"] = TaskRuntime(
        task_id="task-1",
        tg_node_id="tg-1",
        status=TaskRuntimeStatus.RUNNING,
        attempt_count=0,
        max_attempts=2,
    )

    state = reducer.apply_event(
        state,
        TaskFailedEvent(
            operation_id="op-1",
            task_id="task-1",
            tg_node_id="tg-1",
            error_message="boom",
            attempt_count=1,
        ),
    )

    task = state.execution.tasks["task-1"]
    assert task.status == TaskRuntimeStatus.FAILED
    assert task.attempt_count == 1
    assert task.last_error == "boom"


def test_worker_assignment_and_release_updates_both_sides() -> None:
    state = build_runtime_state()
    reducer = RuntimeStateReducer()

    state = reducer.apply_event(
        state,
        WorkerAssignedEvent(
            operation_id="op-1",
            worker_id="worker-1",
            task_id="task-1",
        ),
    )
    assert state.workers["worker-1"].current_task_id == "task-1"
    assert state.execution.tasks["task-1"].assigned_worker == "worker-1"

    state = reducer.apply_event(
        state,
        WorkerReleasedEvent(
            operation_id="op-1",
            worker_id="worker-1",
            task_id="task-1",
        ),
    )
    assert state.workers["worker-1"].current_task_id is None


def test_lock_manager_acquire_conflict_and_cleanup() -> None:
    state = build_runtime_state()
    manager = RuntimeLockManager()

    result = manager.acquire_lock(
        state=state,
        lock_key="host:1",
        owner_type="task",
        owner_id="task-1",
        ttl_seconds=1,
    )
    assert result.acquired is True
    assert manager.is_locked(state, "host:1") is True

    with pytest.raises(LockConflictError):
        manager.acquire_lock(
            state=state,
            lock_key="host:1",
            owner_type="task",
            owner_id="task-2",
            ttl_seconds=1,
        )

    state.locks["host:1"].expires_at = utc_now()
    expired = manager.cleanup_expired_locks(state)
    assert expired == 1
    assert manager.is_locked(state, "host:1") is False


def test_session_manager_open_heartbeat_expire_and_reuse() -> None:
    state = build_runtime_state()
    manager = RuntimeSessionManager()

    session = manager.open_session(
        state=state,
        session_id="sess-1",
        bound_identity="alice",
        bound_target="host-1",
        lease_seconds=60,
        reusability="shared",
    )
    original_heartbeat = session.heartbeat_at
    assert manager.is_session_usable(state, "sess-1") is True

    manager.heartbeat_session(state, "sess-1")
    assert state.sessions["sess-1"].heartbeat_at >= original_heartbeat

    reusable = manager.list_reusable_sessions(state, bound_target="host-1", bound_identity="alice")
    assert [item.session_id for item in reusable] == ["sess-1"]

    manager.expire_session(state, "sess-1", reason="timeout")
    assert state.sessions["sess-1"].status == SessionStatus.EXPIRED
    assert manager.is_session_usable(state, "sess-1") is False


def test_budget_manager_accumulates_and_detects_limit_exceeded() -> None:
    state = build_runtime_state()
    manager = RuntimeBudgetManager()
    state.budgets.token_budget_max = 10
    state.budgets.risk_budget_max = 1.0
    state.budgets.noise_budget_max = 1.0

    manager.consume_tokens(state, 4)
    manager.consume_risk(state, 0.2)
    manager.consume_noise(state, 0.3)

    assert state.budgets.token_budget_used == 4
    assert state.budgets.risk_budget_used == pytest.approx(0.2)
    assert state.budgets.noise_budget_used == pytest.approx(0.3)
    assert manager.would_exceed_budget(state, tokens=7) is True

    with pytest.raises(BudgetExceededError):
        manager.consume_tokens(state, 7)


def test_checkpoint_manager_create_and_get_checkpoint() -> None:
    state = build_runtime_state()
    manager = RuntimeCheckpointManager()

    created = manager.create_checkpoint(
        state=state,
        checkpoint_id="cp-1",
        created_after_tasks=["task-1"],
        kg_version="kg-v1",
        ag_version="ag-v1",
        tg_version="tg-v1",
        summary="stable",
    )
    loaded = manager.get_checkpoint(state, "cp-1")

    assert created.checkpoint_id == "cp-1"
    assert loaded.summary == "stable"
    assert manager.latest_stable_checkpoint(state).checkpoint_id == "cp-1"


def test_checkpoint_manager_writes_replan_marker_into_recovery_metadata() -> None:
    state = build_runtime_state()
    manager = RuntimeCheckpointManager()

    marker = manager.add_replan_marker(
        state,
        {"task_id": "task-9", "reason": "branch failed"},
    )

    recovery = state.execution.metadata["recovery"]
    assert marker["task_id"] == "task-9"
    assert recovery["replan_markers"][0]["reason"] == "branch failed"


def test_store_append_event_does_not_update_state_but_apply_event_does() -> None:
    store = InMemoryRuntimeStore()
    store.create_operation("op-1")
    event = TaskQueuedEvent(operation_id="op-1", task_id="task-1", tg_node_id="tg-1")

    store.append_event("op-1", event)
    state_after_append = store.get_state("op-1")
    assert state_after_append is not None
    assert "task-1" not in state_after_append.execution.tasks
    assert len(store.list_events("op-1")) == 1

    updated = store.apply_event("op-1", event)
    assert updated.execution.tasks["task-1"].status == TaskRuntimeStatus.QUEUED
    assert updated.event_cursor == 2


def test_store_snapshot_does_not_mutate_original_state() -> None:
    store = InMemoryRuntimeStore()
    store.create_operation("op-1")
    store.apply_event(
        "op-1",
        ReplanRequestedEvent(
            operation_id="op-1",
            request_id="replan-1",
            reason="runtime failure",
            task_ids=["task-1"],
        ),
    )

    snap = store.snapshot("op-1")
    snap.replan_requests.append(
        snap.replan_requests[0].model_copy(update={"request_id": "replan-2"})
    )

    original = store.get_state("op-1")
    assert original is not None
    assert [item.request_id for item in original.replan_requests] == ["replan-1"]
