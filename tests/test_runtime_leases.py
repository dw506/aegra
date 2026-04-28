from __future__ import annotations

from datetime import timedelta

import pytest

from src.core.models.runtime import OperationRuntime, RuntimeState, SessionRuntime, SessionStatus, TaskRuntime
from src.core.runtime.lease_manager import RuntimeLeaseManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_create_and_extend_lease_updates_expiry() -> None:
    state = build_state()
    state.add_session(SessionRuntime(session_id="sess-1", status=SessionStatus.ACTIVE))
    manager = RuntimeLeaseManager()
    lease = manager.create_lease(state, "lease-1", "sess-1", "task-1", lease_seconds=30)
    original_expiry = lease.lease_expiry

    updated = manager.extend_lease(state, "lease-1", extra_seconds=60)

    assert updated.lease_expiry > original_expiry


def test_bind_lease_updates_task_and_session_metadata() -> None:
    state = build_state()
    state.add_session(SessionRuntime(session_id="sess-1", status=SessionStatus.ACTIVE))
    state.register_task(TaskRuntime(task_id="task-1", tg_node_id="tg-1"))
    manager = RuntimeLeaseManager()
    manager.create_lease(state, "lease-1", "sess-1", "task-0", lease_seconds=30)

    lease = manager.bind_lease_to_task_or_session(state, "lease-1", task_id="task-1", session_id="sess-1")

    assert lease.owner_task_id == "task-1"
    assert state.execution.tasks["task-1"].metadata["session_lease_id"] == "lease-1"
    assert "lease-1" in state.sessions["sess-1"].metadata["lease_ids"]


def test_cleanup_expired_leases_marks_elapsed_leases() -> None:
    state = build_state()
    state.add_session(SessionRuntime(session_id="sess-1", status=SessionStatus.ACTIVE))
    manager = RuntimeLeaseManager()
    manager.create_lease(state, "lease-1", "sess-1", "task-1", lease_seconds=30)
    state.session_leases["lease-1"].lease_expiry = state.session_leases["lease-1"].acquired_at - timedelta(seconds=1)

    expired = manager.cleanup_expired_leases(state)

    assert expired == 1
    assert state.session_leases["lease-1"].metadata["expiry_reason"] == "lease_expired"


def test_extend_released_lease_raises() -> None:
    state = build_state()
    state.add_session(SessionRuntime(session_id="sess-1", status=SessionStatus.ACTIVE))
    manager = RuntimeLeaseManager()
    manager.create_lease(state, "lease-1", "sess-1", "task-1", lease_seconds=30)
    manager.release_lease(state, "lease-1", reason="done")

    with pytest.raises(ValueError):
        manager.extend_lease(state, "lease-1", extra_seconds=60)
