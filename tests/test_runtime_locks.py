from __future__ import annotations

import pytest

from src.core.models.runtime import OperationRuntime, RuntimeState, utc_now
from src.core.runtime.locks import LockConflictError, RuntimeLockManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_acquire_many_rolls_back_on_conflict() -> None:
    state = build_state()
    manager = RuntimeLockManager()
    manager.acquire_lock(state, "host:1", "task", "task-a", ttl_seconds=30)

    with pytest.raises(LockConflictError):
        manager.acquire_many(
            state,
            ["host:1", "service:1"],
            owner_type="task",
            owner_id="task-b",
            ttl_seconds=30,
        )

    assert state.locks["host:1"].owner_id == "task-a"
    assert "service:1" not in state.locks or state.locks["service:1"].status != "active"


def test_release_all_for_owner_releases_only_owned_locks() -> None:
    state = build_state()
    manager = RuntimeLockManager()
    manager.acquire_lock(state, "host:1", "task", "task-a", ttl_seconds=30)
    manager.acquire_lock(state, "host:2", "task", "task-a", ttl_seconds=30)
    manager.acquire_lock(state, "host:3", "task", "task-b", ttl_seconds=30)

    released = manager.release_all_for_owner(state, "task-a")

    assert released == 2
    assert manager.is_locked(state, "host:1") is False
    assert manager.is_locked(state, "host:3") is True


def test_get_lock_returns_expired_lock_after_cleanup() -> None:
    state = build_state()
    manager = RuntimeLockManager()
    manager.acquire_lock(state, "host:1", "task", "task-a", ttl_seconds=30)
    state.locks["host:1"].expires_at = utc_now()

    manager.cleanup_expired_locks(state)
    lock = manager.get_lock(state, "host:1")

    assert lock is not None
    assert lock.status.value == "expired"


def test_expand_policy_lock_keys_adds_credential_mutex_and_session_mode() -> None:
    manager = RuntimeLockManager()

    expanded = manager.expand_policy_lock_keys(
        {"credential:cred-1", "session:sess-1"},
        session_policies={"sess-1": "shared"},
    )

    assert "credential:cred-1" in expanded
    assert "mutex:credential:cred-1" in expanded
    assert "session:sess-1" in expanded
    assert "session-shared:sess-1" in expanded


def test_session_policy_conflict_blocks_exclusive_against_active_shared_lock() -> None:
    state = build_state()
    manager = RuntimeLockManager()
    manager.acquire_lock(state, "session-shared:sess-1", "task", "task-a", ttl_seconds=30)

    assert manager.is_session_policy_conflict(
        state,
        session_id="sess-1",
        policy="exclusive",
        owner_id="task-b",
    ) is True
    assert manager.is_session_policy_conflict(
        state,
        session_id="sess-1",
        policy="shared",
        owner_id="task-b",
    ) is False
