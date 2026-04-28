"""Runtime State storage abstraction with in-memory and file persistence.

This module stores RuntimeState snapshots and append-only runtime event logs.
It is designed for unit tests and local execution flows. A future Redis-backed
implementation should follow the same RuntimeStore contract, but no Redis
dependency is introduced here.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from threading import RLock
from typing import Any

from src.core.models.runtime import OperationRuntime, RuntimeEventRef, RuntimeState
from src.core.runtime.events import RuntimeEvent, coerce_runtime_event, event_to_ref
from src.core.runtime.observability import (
    build_audit_report,
    build_event_log_replay_annotations,
    build_recovery_snapshot,
    prepare_state_for_resume,
)
from src.core.runtime.reducer import RuntimeStateReducer


def _event_refs_from_log(events: list[tuple[int, RuntimeEvent]]) -> list[RuntimeEventRef]:
    return [
        event_to_ref(coerce_runtime_event(event), cursor=cursor)
        for cursor, event in events
    ]


class RuntimeStore(ABC):
    """Abstract storage contract for runtime state and event logs.

    Future Redis implementation notes:
    - Store `RuntimeState` snapshots by `operation_id`.
    - Store append-only event logs by `operation_id`, preserving cursor order.
    - `append_event` should allocate a monotonically increasing cursor.
    - `apply_event` should append the event, run the reducer and persist the new state.
    - `snapshot` should return a detached copy so callers cannot mutate store internals.
    """

    @abstractmethod
    def get_state(self, operation_id: str) -> RuntimeState | None:
        """Return the current runtime state snapshot for one operation."""

    @abstractmethod
    def save_state(self, state: RuntimeState) -> None:
        """Persist a full runtime state snapshot."""

    @abstractmethod
    def append_event(self, operation_id: str, event: RuntimeEvent) -> None:
        """Append one event to the event log without updating state."""

    @abstractmethod
    def list_events(self, operation_id: str, after_cursor: int | None = None) -> list[RuntimeEvent]:
        """Return events for one operation, optionally after a cursor."""

    @abstractmethod
    def apply_event(self, operation_id: str, event: RuntimeEvent) -> RuntimeState:
        """Append one event, reduce it into state and return the updated snapshot."""

    @abstractmethod
    def create_operation(
        self,
        operation_id: str,
        initial_state: RuntimeState | None = None,
    ) -> RuntimeState:
        """Create and return a new operation runtime state."""

    @abstractmethod
    def delete_operation(self, operation_id: str) -> None:
        """Delete one operation state and its event log."""

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
    """Thread-friendly in-memory store for Runtime State and event logs."""

    def __init__(self, reducer: RuntimeStateReducer | None = None) -> None:
        self._states: dict[str, RuntimeState] = {}
        self._event_logs: dict[str, list[tuple[int, RuntimeEvent]]] = {}
        self._cursors: dict[str, int] = {}
        self._reducer = reducer or RuntimeStateReducer()
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
            self._event_logs.setdefault(state.operation_id, [])
            self._cursors.setdefault(state.operation_id, state.event_cursor)

    def append_event(self, operation_id: str, event: RuntimeEvent) -> None:
        """Append one typed event to the log without mutating RuntimeState."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            typed_event = coerce_runtime_event(event)
            cursor = self._next_cursor(operation_id)
            self._event_logs[operation_id].append((cursor, typed_event))

    def list_events(self, operation_id: str, after_cursor: int | None = None) -> list[RuntimeEvent]:
        """List typed runtime events for one operation in cursor order."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            minimum = after_cursor or 0
            return [
                event
                for cursor, event in self._event_logs[operation_id]
                if cursor > minimum
            ]

    def apply_event(self, operation_id: str, event: RuntimeEvent) -> RuntimeState:
        """Append an event, reduce it into state and return the updated snapshot."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            typed_event = coerce_runtime_event(event)
            cursor = self._next_cursor(operation_id)
            self._event_logs[operation_id].append((cursor, typed_event))

            current_state = self._states[operation_id]
            next_state = self._reducer.apply_event(current_state, typed_event)
            next_state.event_cursor = cursor
            next_state.push_event(event_to_ref(typed_event, cursor=cursor))
            self._states[operation_id] = next_state
            return next_state.model_copy(deep=True)

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
            self._event_logs[operation_id] = []
            self._cursors[operation_id] = state.event_cursor
            return state.model_copy(deep=True)

    def delete_operation(self, operation_id: str) -> None:
        """Delete one operation state and its event log."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            del self._states[operation_id]
            del self._event_logs[operation_id]
            del self._cursors[operation_id]

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
            prepare_state_for_resume(
                state,
                reason=reason,
                event_refs=_event_refs_from_log(self._event_logs[operation_id]),
            )
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

    def _next_cursor(self, operation_id: str) -> int:
        """Allocate the next monotonically increasing event cursor."""

        cursor = self._cursors.get(operation_id, 0) + 1
        self._cursors[operation_id] = cursor
        return cursor


class FileRuntimeStore(RuntimeStore):
    """JSON-file-backed runtime store for local control-plane persistence."""

    STATE_FILENAME = "state.json"
    EVENTS_FILENAME = "events.json"
    AUDIT_FILENAME = "audit.json"
    RECOVERY_FILENAME = "recovery.json"
    OPERATION_LOG_FILENAME = "operation-log.jsonl"

    def __init__(
        self,
        root_dir: str | Path,
        reducer: RuntimeStateReducer | None = None,
    ) -> None:
        self._root_dir = Path(root_dir).resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._reducer = reducer or RuntimeStateReducer()
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
            events_path = self._events_path(state.operation_id)
            if not events_path.exists():
                self._write_json(events_path, [])

    def append_event(self, operation_id: str, event: RuntimeEvent) -> None:
        """Append one typed event to the log without mutating RuntimeState."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            typed_event = coerce_runtime_event(event)
            current_state = self._load_state(operation_id)
            event_log = self._load_event_log(operation_id)
            cursor = self._next_cursor(event_log, current_state.event_cursor)
            event_log.append((cursor, typed_event))
            self._write_event_log(operation_id, event_log)

    def list_events(self, operation_id: str, after_cursor: int | None = None) -> list[RuntimeEvent]:
        """List typed runtime events for one operation in cursor order."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            minimum = after_cursor or 0
            return [
                event
                for cursor, event in self._load_event_log(operation_id)
                if cursor > minimum
            ]

    def apply_event(self, operation_id: str, event: RuntimeEvent) -> RuntimeState:
        """Append an event, reduce it into state and return the updated snapshot."""

        with self._lock:
            self._ensure_operation_exists(operation_id)
            typed_event = coerce_runtime_event(event)
            current_state = self._load_state(operation_id)
            event_log = self._load_event_log(operation_id)
            cursor = self._next_cursor(event_log, current_state.event_cursor)
            event_log.append((cursor, typed_event))

            next_state = self._reducer.apply_event(current_state, typed_event)
            next_state.event_cursor = cursor
            next_state.push_event(event_to_ref(typed_event, cursor=cursor))

            self._write_event_log(operation_id, event_log)
            self._persist_state_artifacts(next_state)
            return next_state.model_copy(deep=True)

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
            self._write_json(self._events_path(operation_id), [])
            return state.model_copy(deep=True)

    def delete_operation(self, operation_id: str) -> None:
        """Delete one operation state and its event log."""

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
            event_log = self._load_event_log(operation_id)
            prepare_state_for_resume(
                state,
                reason=reason,
                event_refs=_event_refs_from_log(event_log),
            )
            # 中文注释：
            # recover 之后需要把 replay planning 一并写回 events.json，
            # 这样 file store 下可以直接看到当前 replay 窗口和候选事件。
            self._write_event_log(operation_id, event_log, state=state)
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

    def _events_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.EVENTS_FILENAME

    def _audit_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.AUDIT_FILENAME

    def _recovery_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.RECOVERY_FILENAME

    def _operation_log_path(self, operation_id: str) -> Path:
        return self._operation_dir(operation_id) / self.OPERATION_LOG_FILENAME

    def _load_state(self, operation_id: str) -> RuntimeState:
        return RuntimeState.model_validate(self._read_json(self._state_path(operation_id)))

    def _load_event_log(self, operation_id: str) -> list[tuple[int, RuntimeEvent]]:
        events_path = self._events_path(operation_id)
        if not events_path.exists():
            return []
        payload = self._read_json(events_path)
        return [(int(item["cursor"]), coerce_runtime_event(item["event"])) for item in payload]

    def _write_event_log(
        self,
        operation_id: str,
        events: list[tuple[int, RuntimeEvent]],
        *,
        state: RuntimeState | None = None,
    ) -> None:
        event_refs = _event_refs_from_log(events)
        replay_annotations = build_event_log_replay_annotations(state, event_refs) if state is not None else {}
        payload = [
            {
                "cursor": cursor,
                "event": coerce_runtime_event(event).model_dump(mode="json"),
                "event_ref": event_to_ref(coerce_runtime_event(event), cursor=cursor).model_dump(mode="json"),
                "replay": replay_annotations.get(event_to_ref(coerce_runtime_event(event), cursor=cursor).event_id),
            }
            for cursor, event in events
        ]
        self._write_json(self._events_path(operation_id), payload)

    def _persist_state_artifacts(self, state: RuntimeState) -> None:
        self._write_json(self._state_path(state.operation_id), state.model_dump(mode="json"))
        self._write_json(self._audit_path(state.operation_id), build_audit_report(state))
        self._write_json(self._recovery_path(state.operation_id), build_recovery_snapshot(state))
        self._write_jsonl(self._operation_log_path(state.operation_id), state.execution.metadata.get("operation_log", []))

    @staticmethod
    def _next_cursor(event_log: list[tuple[int, RuntimeEvent]], current_cursor: int) -> int:
        if event_log:
            return event_log[-1][0] + 1
        return current_cursor + 1

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
