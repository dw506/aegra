from __future__ import annotations

import pytest

from src.core.models.runtime import CredentialStatus, OperationRuntime, RuntimeState
from src.core.runtime.credential_manager import RuntimeCredentialManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_upsert_credential_and_bind_target() -> None:
    state = build_state()
    manager = RuntimeCredentialManager()

    credential = manager.upsert_credential(
        state,
        "cred-1",
        "alice",
        kind="token",
        secret_ref="secret://cred-1",
    )
    updated = manager.bind_target(state, "cred-1", "host-1")

    assert credential.kind.value == "token"
    assert updated.bound_targets == {"host-1"}


def test_record_validation_marks_valid_and_updates_timestamp() -> None:
    state = build_state()
    manager = RuntimeCredentialManager()
    manager.upsert_credential(state, "cred-1", "alice")

    credential = manager.record_validation(
        state,
        "cred-1",
        status="valid",
        target_id="host-1",
        metadata={"validator": "ssh-login"},
    )

    assert credential.status == CredentialStatus.VALID
    assert credential.last_validated_at is not None
    assert "host-1" in credential.bound_targets
    assert credential.metadata["validator"] == "ssh-login"


def test_mark_invalid_increments_failure_count() -> None:
    state = build_state()
    manager = RuntimeCredentialManager()
    manager.upsert_credential(state, "cred-1", "alice")

    credential = manager.mark_invalid(state, "cred-1", reason="auth_failed")

    assert credential.status == CredentialStatus.INVALID
    assert credential.failure_count == 1
    assert credential.metadata["invalid_reason"] == "auth_failed"


def test_get_missing_credential_raises() -> None:
    state = build_state()
    manager = RuntimeCredentialManager()

    with pytest.raises(ValueError):
        manager.get_credential(state, "missing")
