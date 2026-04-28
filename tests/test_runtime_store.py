from __future__ import annotations

import json
from datetime import timedelta

from src.core.models.runtime import (
    LockStatus,
    OperationRuntime,
    ResourceLock,
    RuntimeEventRef,
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
from src.core.runtime.events import TaskQueuedEvent, event_to_ref
from src.core.runtime.observability import append_audit_log, append_operation_log
from src.core.runtime.store import FileRuntimeStore, InMemoryRuntimeStore


def build_state(operation_id: str = "op-1") -> RuntimeState:
    return RuntimeState(
        operation_id=operation_id,
        execution=OperationRuntime(operation_id=operation_id),
    )


def test_create_operation_with_initial_state() -> None:
    store = InMemoryRuntimeStore()
    state = build_state()

    created = store.create_operation("op-1", initial_state=state)

    assert created.operation_id == "op-1"
    assert store.get_state("op-1") is not None


def test_list_events_respects_after_cursor() -> None:
    store = InMemoryRuntimeStore()
    store.create_operation("op-1")

    first = TaskQueuedEvent(operation_id="op-1", task_id="task-1", tg_node_id="tg-1")
    second = TaskQueuedEvent(operation_id="op-1", task_id="task-2", tg_node_id="tg-2")
    store.append_event("op-1", first)
    store.append_event("op-1", second)

    events = store.list_events("op-1", after_cursor=1)

    assert [event.task_id for event in events] == ["task-2"]


def test_apply_event_persists_reduced_state() -> None:
    store = InMemoryRuntimeStore()
    store.create_operation("op-1")

    updated = store.apply_event(
        "op-1",
        TaskQueuedEvent(operation_id="op-1", task_id="task-1", tg_node_id="tg-1"),
    )

    assert updated.execution.tasks["task-1"].status == TaskRuntimeStatus.QUEUED
    assert store.get_state("op-1").execution.tasks["task-1"].status == TaskRuntimeStatus.QUEUED


def test_file_store_persists_state_and_events(tmp_path) -> None:
    store = FileRuntimeStore(tmp_path / "runtime-store")
    store.create_operation("op-1", initial_state=build_state())

    updated = store.apply_event(
        "op-1",
        TaskQueuedEvent(operation_id="op-1", task_id="task-1", tg_node_id="tg-1"),
    )
    reloaded = FileRuntimeStore(tmp_path / "runtime-store")

    assert updated.execution.tasks["task-1"].status == TaskRuntimeStatus.QUEUED
    assert reloaded.get_state("op-1").execution.tasks["task-1"].status == TaskRuntimeStatus.QUEUED
    assert [event.task_id for event in reloaded.list_events("op-1")] == ["task-1"]
    assert reloaded.list_operation_ids() == ["op-1"]


def test_store_recovery_and_audit_export(tmp_path) -> None:
    store = FileRuntimeStore(tmp_path / "runtime-store")
    state = build_state()
    state.execution.metadata["operation_log"] = [{"seq": 1, "event_type": "cycle_started"}]
    state.execution.metadata["audit_log"] = [{"audit_id": "audit-1", "event_type": "fact_write"}]
    state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status=TaskRuntimeStatus.RUNNING,
            assigned_worker="worker-1",
        )
    )
    state.workers["worker-1"] = WorkerRuntime(
        worker_id="worker-1",
        status=WorkerStatus.BUSY,
        current_task_id="task-1",
    )
    store.create_operation("op-1", initial_state=state)

    recovered = store.recover_operation("op-1", reason="unit_test_resume")
    audit_report = store.export_audit_report("op-1")
    audit_path = tmp_path / "runtime-store" / "op-1" / "audit.json"
    recovery_path = tmp_path / "runtime-store" / "op-1" / "recovery.json"
    operation_log_path = tmp_path / "runtime-store" / "op-1" / "operation-log.jsonl"

    assert recovered.execution.tasks["task-1"].status == TaskRuntimeStatus.PENDING
    assert recovered.execution.tasks["task-1"].metadata["resume_reason"] == "unit_test_resume"
    assert recovered.workers["worker-1"].status == WorkerStatus.IDLE
    assert audit_report["audit_log"][0]["event_type"] == "fact_write"
    assert json.loads(audit_path.read_text(encoding="utf-8"))["operation_log"][0]["event_type"] == "cycle_started"
    assert json.loads(recovery_path.read_text(encoding="utf-8"))["inflight_task_ids"] == []
    assert "cycle_started" in operation_log_path.read_text(encoding="utf-8")


def test_store_recovery_normalizes_sessions_locks_leases_and_pending_events(tmp_path) -> None:
    store = FileRuntimeStore(tmp_path / "runtime-store")
    now = utc_now()
    state = build_state("op-recovery")
    state.execution.metadata["recovery"] = {"unclean_shutdown": True}
    state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status=TaskRuntimeStatus.RUNNING,
            assigned_worker="worker-1",
            started_at=now,
            deadline=now + timedelta(minutes=5),
        )
    )
    state.workers["worker-1"] = WorkerRuntime(
        worker_id="worker-1",
        status=WorkerStatus.BUSY,
        current_task_id="task-1",
        current_load=1,
    )
    state.sessions["session-1"] = SessionRuntime(
        session_id="session-1",
        status=SessionStatus.ACTIVE,
        lease_expiry=now + timedelta(minutes=5),
        metadata={"bound_task_ids": ["task-1"], "lease_ids": ["lease-1"]},
    )
    state.session_leases["lease-1"] = SessionLeaseRuntime(
        lease_id="lease-1",
        session_id="session-1",
        owner_task_id="task-1",
        owner_worker_id="worker-1",
        lease_expiry=now + timedelta(minutes=5),
    )
    state.locks["host:host-1"] = ResourceLock(
        lock_key="host:host-1",
        owner_type="task",
        owner_id="task-1",
        status=LockStatus.ACTIVE,
    )
    state.pending_events.append(
        RuntimeEventRef(
            event_id="evt-1",
            event_type="session.opened",
            cursor=4,
            metadata={"source": "test"},
        )
    )
    store.create_operation("op-recovery", initial_state=state)

    recovered = store.recover_operation("op-recovery", reason="unit_test_resume")
    state_snapshot = store.export_state_snapshot("op-recovery")
    recovery_snapshot = store.export_recovery_snapshot("op-recovery")

    assert recovered.execution.tasks["task-1"].status == TaskRuntimeStatus.PENDING
    assert recovered.workers["worker-1"].status == WorkerStatus.IDLE
    assert recovered.workers["worker-1"].current_load == 0
    assert recovered.sessions["session-1"].status == SessionStatus.EXPIRED
    assert recovered.session_leases["lease-1"].metadata["release_reason"] == "unit_test_resume"
    assert recovered.locks["host:host-1"].status == LockStatus.RELEASED
    assert recovered.pending_events[0].metadata["recovery"]["replay_ready"] is True
    assert recovered.execution.metadata["recovery"]["released_lock_ids"] == ["host:host-1"]
    assert recovered.execution.metadata["recovery"]["expired_session_ids"] == ["session-1"]
    assert recovered.execution.metadata["recovery"]["recovered_event_count"] == 1
    assert state_snapshot["sessions"]["session-1"]["status"] == "expired"
    assert recovery_snapshot["recovery_metadata"]["released_lock_ids"] == ["host:host-1"]
    assert recovery_snapshot["recovery_metadata"]["replay_required"] is True


def test_inmemory_store_exports_recovery_snapshot_after_recovery() -> None:
    store = InMemoryRuntimeStore()
    now = utc_now()
    state = build_state("op-memory")
    state.execution.metadata["recovery"] = {"unclean_shutdown": True}
    state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status=TaskRuntimeStatus.CLAIMED,
            assigned_worker="worker-1",
            started_at=now,
        )
    )
    state.workers["worker-1"] = WorkerRuntime(
        worker_id="worker-1",
        status=WorkerStatus.BUSY,
        current_task_id="task-1",
        current_load=1,
    )
    store.create_operation("op-memory", initial_state=state)

    store.recover_operation("op-memory", reason="memory_resume")
    recovery_snapshot = store.export_recovery_snapshot("op-memory")

    assert recovery_snapshot["recovery_metadata"]["last_resume_reason"] == "memory_resume"
    assert recovery_snapshot["recovery_metadata"]["resumed_task_ids"] == ["task-1"]
    assert recovery_snapshot["inflight_task_ids"] == []


def test_file_store_exports_last_phase_checkpoint_in_recovery_snapshot(tmp_path) -> None:
    store = FileRuntimeStore(tmp_path / "runtime-store")
    state = build_state("op-phase")
    state.execution.metadata["phase_checkpoints"] = [
        {
            "cycle_index": 1,
            "phase": "apply_completed",
            "status": "completed",
            "at": utc_now().isoformat(),
            "selected_task_ids": ["task-1"],
            "applied_task_ids": ["task-1"],
            "runtime_event_count": 1,
        }
    ]
    state.execution.metadata["last_phase_checkpoint"] = dict(state.execution.metadata["phase_checkpoints"][0])
    state.execution.metadata["recovery"] = {
        "unclean_shutdown": True,
        "last_phase": "apply_completed",
        "last_phase_status": "completed",
        "last_phase_checkpoint": dict(state.execution.metadata["phase_checkpoints"][0]),
        "phase_checkpoint_count": 1,
    }
    store.create_operation("op-phase", initial_state=state)

    recovery_snapshot = store.export_recovery_snapshot("op-phase")

    assert recovery_snapshot["last_phase_checkpoint"]["phase"] == "apply_completed"
    assert recovery_snapshot["recovery_metadata"]["phase_checkpoint_count"] == 1


def test_store_recovery_persists_replay_plan_and_event_log_annotations(tmp_path) -> None:
    store = FileRuntimeStore(tmp_path / "runtime-store")
    state = build_state("op-replay")
    state.execution.metadata["recovery"] = {
        "unclean_shutdown": True,
        "last_replayed_cursor": 0,
    }
    store.create_operation("op-replay", initial_state=state)

    queued = TaskQueuedEvent(operation_id="op-replay", task_id="task-1", tg_node_id="task-1")
    store.append_event("op-replay", queued)
    updated = store.get_state("op-replay")
    assert updated is not None
    pending_ref = event_to_ref(queued, cursor=1)
    updated.push_event(pending_ref)
    store.save_state(updated)

    recovered = store.recover_operation("op-replay", reason="unit_test_replay")
    recovery_snapshot = store.export_recovery_snapshot("op-replay")
    events_path = tmp_path / "runtime-store" / "op-replay" / "events.json"
    persisted_events = json.loads(events_path.read_text(encoding="utf-8"))

    assert recovered.execution.metadata["recovery"]["replay_required"] is True
    assert recovered.execution.metadata["recovery"]["replay_status"] == "planned"
    assert recovered.execution.metadata["recovery"]["last_replayed_cursor"] == 0
    assert recovered.execution.metadata["recovery"]["replay_candidate_event_ids"] == [pending_ref.event_id]
    assert recovered.pending_events[0].metadata["replay"]["replay_status"] == "planned"
    assert recovery_snapshot["replay_plan"]["start_cursor"] == 1
    assert recovery_snapshot["replay_plan"]["replay_candidate_event_ids"] == [pending_ref.event_id]
    assert persisted_events[0]["event_ref"]["event_id"] == pending_ref.event_id
    assert persisted_events[0]["replay"]["replay_status"] == "candidate"
    assert persisted_events[0]["replay"]["start_cursor"] == 1


def test_audit_export_redacts_and_caps_logs(tmp_path) -> None:
    store = FileRuntimeStore(tmp_path / "runtime-store")
    state = build_state("op-audit")
    state.execution.metadata["control_plane"] = {
        "audit_max_entries": 2,
        "operation_log_max_entries": 2,
        "audit_redaction_enabled": True,
    }

    for index in range(3):
        append_operation_log(
            state,
            event_type="command_event",
            command=f"curl -H 'Authorization: Bearer op-secret-{index}' https://example.test/{index}",
            metadata={
                "token": f"token-{index}",
                "notes": "n" * 400,
            },
        )
        append_audit_log(
            state,
            {
                "event_type": "tool_invocation",
                "command": f"curl -H 'Authorization: Bearer audit-secret-{index}' password=pw-{index}",
                "metadata": {
                    "api_key": f"key-{index}",
                    "nested": {"secret": f"secret-{index}"},
                },
                "evidence_chain": {
                    "summary": "s" * 400,
                    "steps": [{"authorization": f"Bearer chain-{step}"} for step in range(25)],
                },
            },
        )

    store.create_operation("op-audit", initial_state=state)
    audit_report = store.export_audit_report("op-audit")
    audit_path = tmp_path / "runtime-store" / "op-audit" / "audit.json"
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    serialized_report = json.dumps(audit_report)

    assert len(audit_report["operation_log"]) == 2
    assert len(audit_report["audit_log"]) == 2
    assert "op-secret-0" not in serialized_report
    assert "audit-secret-2" not in serialized_report
    assert "token-2" not in serialized_report
    assert "[REDACTED]" in serialized_report
    assert "(truncated" in audit_report["audit_log"][-1]["evidence_chain"]["summary"]
    assert audit_report["audit_log"][-1]["evidence_chain"]["steps"][-1]["truncated_item_count"] == 5
    assert persisted["audit_log"] == audit_report["audit_log"]
