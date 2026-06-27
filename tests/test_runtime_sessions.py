from __future__ import annotations

from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.runtime.session_manager import RuntimeSessionManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_bind_task_to_session_updates_session_metadata() -> None:
    state = build_state()
    manager = RuntimeSessionManager()
    manager.open_session(state, "sess-1", "alice", "host-1", lease_seconds=30, reusability="shared")

    session = manager.bind_task_to_session(state, "task-1", "sess-1")

    assert "task-1" in session.metadata["bound_task_ids"]
