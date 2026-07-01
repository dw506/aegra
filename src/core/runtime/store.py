"""Runtime State storage abstraction with in-memory and file persistence.

This module stores RuntimeState snapshots for unit tests and local execution
flows. Historical append-only event sourcing has been removed; recovery now
normalizes the latest persisted state snapshot.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from threading import RLock
from typing import Any

from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.runtime.observability import (
    build_audit_report,
    build_recovery_snapshot,
    prepare_state_for_resume,
)


class RuntimeStore(ABC):
    """Abstract storage contract for runtime state snapshots.

    Future Redis implementation notes:
    - Store `RuntimeState` snapshots by `operation_id`.
    - `snapshot` should return a detached copy so callers cannot mutate store internals.
    """

    @abstractmethod
    def get_state(self, operation_id: str) -> RuntimeState | None:
        """Return the current runtime state snapshot for one operation."""

    @abstractmethod
    def save_state(self, state: RuntimeState) -> None:
        """Persist a full runtime state snapshot."""

    @abstractmethod
    def create_operation(
        self,
        operation_id: str,
        initial_state: RuntimeState | None = None,
    ) -> RuntimeState:
        """Create and return a new operation runtime state."""

    @abstractmethod
    def delete_operation(self, operation_id: str) -> None:
        """Delete one operation state."""

    @abstractmethod
    def snapshot(self, operation_id: str) -> RuntimeState:
        """Return a deep-copied runtime state snapshot for one operation."""

    @abstractmethod
    def list_operation_ids(self) -> list[str]:
        """Return the known operation identifiers held by the store."""

    @abstractmethod
    def recover_operation(self, operation_id: str, *, reason: str = "store_recovery") -> RuntimeState:
        """Normalize and persist one operation so it can be resumed safely."""

    @abstractmethod
    def export_state_snapshot(self, operation_id: str) -> dict[str, Any]:
        """Return one exportable runtime state snapshot for the given operation."""

    @abstractmethod
    def export_recovery_snapshot(self, operation_id: str) -> dict[str, Any]:
        """Return one exportable recovery-oriented snapshot for the given operation."""

    @abstractmethod
    def export_audit_report(self, operation_id: str) -> dict[str, Any]:
        """Return one exportable audit report for the given operation."""


class InMemoryRuntimeStore(RuntimeStore):
    """Thread-friendly in-memory store for Runtime State snapshots."""

    def __init__(self) -> None:
        self._states: dict[str, RuntimeState] = {}
        self._lock = RLock()

    def get_state(self, operation_id: str) -> RuntimeState | None:
        """Return a detached runtime state snapshot or `None` if missing."""

        with self._lock:
            state = self._states.get(operation_id)
            if state is None:
                return None
            return state.model_copy(deep=True)

    def save_state(self, state: RuntimeState) -> None:
        """Persist a detached copy of the provided runtime state."""

        with self._lock:
            self._states[state.operation_id] = state.model_copy(deep=True)

    def create_operation(
        self,
        operation_id: str,
        initial_state: RuntimeState | None = None,
    ) -> RuntimeState:
        """Create one operation with an empty or provided initial state."""

        with self._lock:
            if operation_id in self._states:
                raise ValueError(f"operation '{operation_id}' already exists")

            state = initial_state.model_copy(deep=True) if initial_state is not None else RuntimeState(
                operation_id=operation_id,
                execution=OperationRuntime(operation_id=operation_id),
            )
            if state.operation_id != operation_id:
                raise ValueError("initial_state.operation_id must match operation_id")

            self._states[operation_id] = state
            return state.model_copy(deep=True)

    def delete_operation(self, operation_id: str) -> None:
        """Delete one operation state."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            del self._states[operation_id]

    def snapshot(self, operation_id: str) -> RuntimeState:
        """Return a deep copy of the current runtime state for one operation."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            return self._states[operation_id].model_copy(deep=True)

    def list_operation_ids(self) -> list[str]:
        """Return operation identifiers in deterministic order."""

        with self._lock:
            return sorted(self._states)

    def recover_operation(self, operation_id: str, *, reason: str = "store_recovery") -> RuntimeState:
        """Normalize in-flight state so it may be resumed after a crash or restart."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            state = self._states[operation_id].model_copy(deep=True)
            prepare_state_for_resume(state, reason=reason)
            self._states[operation_id] = state.model_copy(deep=True)
            return state.model_copy(deep=True)

    def export_audit_report(self, operation_id: str) -> dict[str, Any]:
        """Build an exportable audit report from the current in-memory snapshot."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            return build_audit_report(self._states[operation_id])

    def export_state_snapshot(self, operation_id: str) -> dict[str, Any]:
        """Return one detached JSON-friendly runtime state snapshot."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            return self._states[operation_id].model_dump(mode="json")

    def export_recovery_snapshot(self, operation_id: str) -> dict[str, Any]:
        """Return one detached recovery snapshot built from the current state."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            return build_recovery_snapshot(self._states[operation_id])

    def _ensure_operation_exists(self, operation_id: str) -> None:
        """Raise when the requested operation does not exist in the store."""

        if operation_id not in self._states:
            raise ValueError(f"operation '{operation_id}' does not exist")


class FileRuntimeStore(RuntimeStore):
    """JSON-file-backed runtime store for local control-plane persistence."""

    STATE_FILENAME = "state.json"
    AUDIT_FILENAME = "audit.json"
    RECOVERY_FILENAME = "recovery.json"
    OPERATION_LOG_FILENAME = "operation-log.jsonl"

    def __init__(
        self,
        root_dir: str | Path,
    ) -> None:
        self._root_dir = Path(root_dir).resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def get_state(self, operation_id: str) -> RuntimeState | None:
        """Return a detached runtime state snapshot or `None` if missing."""

        with self._lock:
            state_path = self._state_path(operation_id)
            if not state_path.exists():
                return None
            return RuntimeState.model_validate(self._read_json(state_path))

    def save_state(self, state: RuntimeState) -> None:
        """Persist a detached copy of the provided runtime state."""

        with self._lock:
            op_dir = self._operation_dir(state.operation_id)
            op_dir.mkdir(parents=True, exist_ok=True)
            self._persist_state_artifacts(state)

    def create_operation(
        self,
        operation_id: str,
        initial_state: RuntimeState | None = None,
    ) -> RuntimeState:
        """Create one operation with an empty or provided initial state."""

        with self._lock:
            op_dir = self._operation_dir(operation_id)
            if self._state_path(operation_id).exists():
                raise ValueError(f"operation '{operation_id}' already exists")
            op_dir.mkdir(parents=True, exist_ok=True)

            state = initial_state.model_copy(deep=True) if initial_state is not None else RuntimeState(
                operation_id=operation_id,
                execution=OperationRuntime(operation_id=operation_id),
            )
            if state.operation_id != operation_id:
                raise ValueError("initial_state.operation_id must match operation_id")

            self._persist_state_artifacts(state)
            return state.model_copy(deep=True)

    def delete_operation(self, operation_id: str) -> None:
        """Delete one operation state."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            op_dir = self._operation_dir(operation_id)
            if op_dir.resolve().parent != self._root_dir:
                raise ValueError("refusing to delete an operation outside the configured store root")
            rmtree(op_dir)

    def snapshot(self, operation_id: str) -> RuntimeState:
        """Return a deep copy of the current runtime state for one operation."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            return self._load_state(operation_id).model_copy(deep=True)

    def list_operation_ids(self) -> list[str]:
        """Return operation identifiers in deterministic order."""

        with self._lock:
            operation_ids = []
            for child in self._root_dir.iterdir():
                if child.is_dir() and child.joinpath(self.STATE_FILENAME).exists():
                    operation_ids.append(child.name)
            return sorted(operation_ids)

    def recover_operation(self, operation_id: str, *, reason: str = "store_recovery") -> RuntimeState:
        """Normalize persisted state so it may be resumed safely."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            state = self._load_state(operation_id)
            prepare_state_for_resume(state, reason=reason)
            self._persist_state_artifacts(state)
            return state.model_copy(deep=True)

    def export_audit_report(self, operation_id: str) -> dict[str, Any]:
        """Return the audit report from disk when available, else build it from state."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            audit_path = self._audit_path(operation_id)
            if audit_path.exists():
                return dict(self._read_json(audit_path))
            return build_audit_report(self._load_state(operation_id))

    def export_state_snapshot(self, operation_id: str) -> dict[str, Any]:
        """Return the persisted state snapshot from disk."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            return self._load_state(operation_id).model_dump(mode="json")

    def export_recovery_snapshot(self, operation_id: str) -> dict[str, Any]:
        """Return the persisted recovery snapshot when available, else rebuild it."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            recovery_path = self._recovery_path(operation_id)
            if recovery_path.exists():
                return dict(self._read_json(recovery_path))
            return build_recovery_snapshot(self._load_state(operation_id))

    def _ensure_operation_exists(self, operation_id: str) -> None:
        """Raise when the requested operation does not exist in the store."""

        if not self._state_path(operation_id).exists():
            raise ValueError(f"operation '{operation_id}' does not exist")

    def _operation_dir(self, operation_id: str) -> Path:
        return self._root_dir / operation_id

    def _state_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.STATE_FILENAME

    def _audit_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.AUDIT_FILENAME

    def _recovery_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.RECOVERY_FILENAME

    def _operation_log_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.OPERATION_LOG_FILENAME

    def _load_state(self, operation_id: str) -> RuntimeState:
        return RuntimeState.model_validate(self._read_json(self._state_path(operation_id)))

    def _persist_state_artifacts(self, state: RuntimeState) -> None:
        self._write_json(self._state_path(state.operation_id), state.model_dump(mode="json"))
        self._write_json(self._audit_path(state.operation_id), build_audit_report(state))
        self._write_json(self._recovery_path(state.operation_id), build_recovery_snapshot(state))
        self._write_jsonl(self._operation_log_path(state.operation_id), state.execution.metadata.get("operation_log", []))

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _write_jsonl(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = payload if isinstance(payload, list) else []
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=True, sort_keys=True) for item in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        temp_path.replace(path)


__all__ = ["FileRuntimeStore", "InMemoryRuntimeStore", "RuntimeStore"]
