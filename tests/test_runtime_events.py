from __future__ import annotations

from src.core.runtime.events import (
    CheckpointCreatedEvent,
    ReplanRequestedEvent,
    TaskQueuedEvent,
    coerce_runtime_event,
    event_to_ref,
)


def test_coerce_runtime_event_builds_typed_event() -> None:
    event = coerce_runtime_event(
        {
            "event_type": "TaskQueued",
            "operation_id": "op-1",
            "task_id": "task-1",
            "tg_node_id": "tg-1",
            "payload": {"source": "scheduler"},
        }
    )

    assert isinstance(event, TaskQueuedEvent)
    assert event.task_id == "task-1"


def test_event_to_ref_uses_event_specific_payload_ref() -> None:
    event = CheckpointCreatedEvent(
        operation_id="op-1",
        checkpoint_id="cp-1",
        created_after_tasks=["task-1"],
    )

    ref = event_to_ref(event, cursor=7)

    assert ref.event_id == event.event_id
    assert ref.cursor == 7
    assert ref.payload_ref == "cp-1"


def test_event_to_ref_handles_replan_request() -> None:
    event = ReplanRequestedEvent(
        operation_id="op-1",
        request_id="rp-1",
        reason="branch failed",
        task_ids=["task-2"],
    )

    ref = event_to_ref(event, cursor=3)

    assert ref.payload_ref == "rp-1"
    assert "ReplanRequested" in ref.summary

