from __future__ import annotations

from datetime import timedelta

import pytest

from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.runtime.session_manager import RuntimeSessionManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_extend_lease_updates_expiry() -> None:
    state = build_state()
    manager = RuntimeSessionManager()
    session = manager.open_session(state, "sess-1", "alice", "host-1", lease_seconds=30, reusability="shared")
    original_expiry = session.lease_expiry

    updated = manager.extend_lease(state, "sess-1", extra_seconds=60)

    assert updated.lease_expiry > original_expiry


def test_bind_task_to_session_updates_metadata_and_task_runtime() -> None:
    state = build_state()
    state.execution.tasks["task-1"] = TaskRuntime(task_id="task-1", tg_node_id="tg-1")
    manager = RuntimeSessionManager()
    manager.open_session(state, "sess-1", "alice", "host-1", lease_seconds=30, reusability="shared")

    session = manager.bind_task_to_session(state, "task-1", "sess-1")

    assert "task-1" in session.metadata["bound_task_ids"]
    assert state.execution.tasks["task-1"].metadata["session_id"] == "sess-1"


def test_cleanup_expired_sessions_marks_elapsed_sessions_expired() -> None:
    state = build_state()
    manager = RuntimeSessionManager()
    manager.open_session(state, "sess-1", "alice", "host-1", lease_seconds=30, reusability="shared")
    state.sessions["sess-1"].lease_expiry = state.sessions["sess-1"].heartbeat_at - timedelta(seconds=1)

    expired = manager.cleanup_expired_sessions(state)

    assert expired == 1
    assert manager.is_session_usable(state, "sess-1") is False


def test_extend_lease_on_expired_session_raises() -> None:
    state = build_state()
    manager = RuntimeSessionManager()
    manager.open_session(state, "sess-1", "alice", "host-1", lease_seconds=1, reusability="exclusive")
    manager.expire_session(state, "sess-1")

    with pytest.raises(ValueError):
        manager.extend_lease(state, "sess-1", extra_seconds=60)

