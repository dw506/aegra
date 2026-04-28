"""Runtime event models for event-driven Runtime State updates.

This module defines the structured events exchanged by the runtime layer.
The events are intentionally reducer-agnostic: they only describe runtime
changes and can be serialized, queued and replayed in tests.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.core.models.runtime import RuntimeEventRef, utc_now


class RuntimeEventType(str, Enum):
    """Supported event types for Runtime State transitions."""

    TASK_QUEUED = "TaskQueued"
    TASK_STARTED = "TaskStarted"
    TASK_COMPLETED = "TaskCompleted"
    TASK_FAILED = "TaskFailed"
    TASK_CANCELLED = "TaskCancelled"
    WORKER_ASSIGNED = "WorkerAssigned"
    WORKER_RELEASED = "WorkerReleased"
    LOCK_ACQUIRED = "LockAcquired"
    LOCK_RELEASED = "LockReleased"
    SESSION_OPENED = "SessionOpened"
    SESSION_EXPIRED = "SessionExpired"
    SESSION_HEARTBEAT = "SessionHeartbeat"
    BUDGET_CONSUMED = "BudgetConsumed"
    CHECKPOINT_CREATED = "CheckpointCreated"
    REPLAN_REQUESTED = "ReplanRequested"


class BaseRuntimeEvent(BaseModel):
    """Common metadata shared by all runtime events."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: RuntimeEventType
    operation_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskQueuedEvent(BaseRuntimeEvent):
    """Event emitted when a task enters the runtime queue."""

    event_type: Literal[RuntimeEventType.TASK_QUEUED] = RuntimeEventType.TASK_QUEUED
    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    queue_name: str | None = None
    deadline: datetime | None = None
    priority: int = Field(default=50, ge=0, le=100)


class TaskStartedEvent(BaseRuntimeEvent):
    """Event emitted when a queued task begins execution."""

    event_type: Literal[RuntimeEventType.TASK_STARTED] = RuntimeEventType.TASK_STARTED
    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    worker_id: str = Field(min_length=1)
    session_id: str | None = None
    attempt_count: int = Field(default=1, ge=0)


class TaskCompletedEvent(BaseRuntimeEvent):
    """Event emitted when a task completes successfully."""

    event_type: Literal[RuntimeEventType.TASK_COMPLETED] = RuntimeEventType.TASK_COMPLETED
    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    outcome_id: str | None = None
    outcome_ref: str | None = None
    summary: str | None = None


class TaskFailedEvent(BaseRuntimeEvent):
    """Event emitted when a task fails during execution."""

    event_type: Literal[RuntimeEventType.TASK_FAILED] = RuntimeEventType.TASK_FAILED
    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    retryable: bool = False
    attempt_count: int = Field(default=0, ge=0)


class TaskCancelledEvent(BaseRuntimeEvent):
    """Event emitted when a task is cancelled before completion."""

    event_type: Literal[RuntimeEventType.TASK_CANCELLED] = RuntimeEventType.TASK_CANCELLED
    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    reason: str | None = None


class WorkerAssignedEvent(BaseRuntimeEvent):
    """Event emitted when a worker is assigned to a task."""

    event_type: Literal[RuntimeEventType.WORKER_ASSIGNED] = RuntimeEventType.WORKER_ASSIGNED
    worker_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    queue_name: str | None = None
    current_load: int = Field(default=1, ge=0)


class WorkerReleasedEvent(BaseRuntimeEvent):
    """Event emitted when a worker is released from a task."""

    event_type: Literal[RuntimeEventType.WORKER_RELEASED] = RuntimeEventType.WORKER_RELEASED
    worker_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    reason: str | None = None
    current_load: int = Field(default=0, ge=0)


class LockAcquiredEvent(BaseRuntimeEvent):
    """Event emitted when runtime acquires a resource lock."""

    event_type: Literal[RuntimeEventType.LOCK_ACQUIRED] = RuntimeEventType.LOCK_ACQUIRED
    lock_key: str = Field(min_length=1)
    owner_type: Literal["operation", "task", "worker", "session"] = "task"
    owner_id: str = Field(min_length=1)
    expires_at: datetime | None = None


class LockReleasedEvent(BaseRuntimeEvent):
    """Event emitted when runtime releases a resource lock."""

    event_type: Literal[RuntimeEventType.LOCK_RELEASED] = RuntimeEventType.LOCK_RELEASED
    lock_key: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    reason: str | None = None


class SessionOpenedEvent(BaseRuntimeEvent):
    """Event emitted when runtime opens or registers a session."""

    event_type: Literal[RuntimeEventType.SESSION_OPENED] = RuntimeEventType.SESSION_OPENED
    session_id: str = Field(min_length=1)
    bound_identity: str | None = None
    bound_target: str | None = None
    lease_expiry: datetime | None = None
    reusability: Literal["single_use", "reusable", "sticky"] = "reusable"


class SessionExpiredEvent(BaseRuntimeEvent):
    """Event emitted when a session expires or becomes unusable."""

    event_type: Literal[RuntimeEventType.SESSION_EXPIRED] = RuntimeEventType.SESSION_EXPIRED
    session_id: str = Field(min_length=1)
    reason: str | None = None
    failure_count: int = Field(default=0, ge=0)


class SessionHeartbeatEvent(BaseRuntimeEvent):
    """Event emitted when a session heartbeat is refreshed."""

    event_type: Literal[RuntimeEventType.SESSION_HEARTBEAT] = RuntimeEventType.SESSION_HEARTBEAT
    session_id: str = Field(min_length=1)
    heartbeat_at: datetime = Field(default_factory=utc_now)
    lease_expiry: datetime | None = None


class BudgetConsumedEvent(BaseRuntimeEvent):
    """Event emitted when runtime consumes one or more budget dimensions."""

    event_type: Literal[RuntimeEventType.BUDGET_CONSUMED] = RuntimeEventType.BUDGET_CONSUMED
    time_budget_used_sec_delta: float = Field(default=0.0, ge=0.0)
    token_budget_used_delta: int = Field(default=0, ge=0)
    operation_budget_used_delta: int = Field(default=0, ge=0)
    noise_budget_used_delta: float = Field(default=0.0, ge=0.0)
    risk_budget_used_delta: float = Field(default=0.0, ge=0.0)
    approval_updates: dict[str, bool] = Field(default_factory=dict)
    policy_flag_updates: dict[str, bool | int | float | str] = Field(default_factory=dict)


class CheckpointCreatedEvent(BaseRuntimeEvent):
    """Event emitted when runtime creates a checkpoint marker."""

    event_type: Literal[RuntimeEventType.CHECKPOINT_CREATED] = RuntimeEventType.CHECKPOINT_CREATED
    checkpoint_id: str = Field(min_length=1)
    created_after_tasks: list[str] = Field(default_factory=list)
    kg_version: str | None = None
    ag_version: str | None = None
    tg_version: str | None = None
    summary: str = Field(default="")


class ReplanRequestedEvent(BaseRuntimeEvent):
    """Event emitted when runtime requests local or full replanning."""

    event_type: Literal[RuntimeEventType.REPLAN_REQUESTED] = RuntimeEventType.REPLAN_REQUESTED
    request_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    scope: Literal["local", "branch", "full"] = "local"


RuntimeEvent = Annotated[
    TaskQueuedEvent
    | TaskStartedEvent
    | TaskCompletedEvent
    | TaskFailedEvent
    | TaskCancelledEvent
    | WorkerAssignedEvent
    | WorkerReleasedEvent
    | LockAcquiredEvent
    | LockReleasedEvent
    | SessionOpenedEvent
    | SessionExpiredEvent
    | SessionHeartbeatEvent
    | BudgetConsumedEvent
    | CheckpointCreatedEvent
    | ReplanRequestedEvent,
    Field(discriminator="event_type"),
]

_RUNTIME_EVENT_ADAPTER = TypeAdapter(RuntimeEvent)


def coerce_runtime_event(data: BaseRuntimeEvent | dict[str, Any]) -> RuntimeEvent:
    """Coerce a raw dictionary or event model into a typed runtime event."""

    if isinstance(data, BaseRuntimeEvent):
        return _RUNTIME_EVENT_ADAPTER.validate_python(data.model_dump(mode="python"))
    return _RUNTIME_EVENT_ADAPTER.validate_python(data)


def event_to_ref(event: BaseRuntimeEvent | RuntimeEvent, cursor: int = 0) -> RuntimeEventRef:
    """Convert a concrete runtime event into a lightweight reference object."""

    typed_event = coerce_runtime_event(event)
    payload_ref = None

    if isinstance(
        typed_event,
        (
            TaskCompletedEvent,
            CheckpointCreatedEvent,
            ReplanRequestedEvent,
        ),
    ):
        if isinstance(typed_event, TaskCompletedEvent):
            payload_ref = typed_event.outcome_ref
        elif isinstance(typed_event, CheckpointCreatedEvent):
            payload_ref = typed_event.checkpoint_id
        elif isinstance(typed_event, ReplanRequestedEvent):
            payload_ref = typed_event.request_id

    summary_parts = [typed_event.event_type.value, typed_event.operation_id]
    if isinstance(typed_event, (TaskQueuedEvent, TaskStartedEvent, TaskCompletedEvent, TaskFailedEvent, TaskCancelledEvent)):
        summary_parts.append(typed_event.task_id)
    elif isinstance(typed_event, (WorkerAssignedEvent, WorkerReleasedEvent)):
        summary_parts.append(typed_event.worker_id)
    elif isinstance(typed_event, (SessionOpenedEvent, SessionExpiredEvent, SessionHeartbeatEvent)):
        summary_parts.append(typed_event.session_id)
    elif isinstance(typed_event, (LockAcquiredEvent, LockReleasedEvent)):
        summary_parts.append(typed_event.lock_key)
    elif isinstance(typed_event, CheckpointCreatedEvent):
        summary_parts.append(typed_event.checkpoint_id)
    elif isinstance(typed_event, ReplanRequestedEvent):
        summary_parts.append(typed_event.request_id)

    return RuntimeEventRef(
        event_id=typed_event.event_id,
        event_type=typed_event.event_type.value,
        created_at=typed_event.created_at,
        cursor=cursor,
        summary=":".join(summary_parts),
        payload_ref=payload_ref,
        metadata={"operation_id": typed_event.operation_id},
    )


__all__ = [
    "BaseRuntimeEvent",
    "BudgetConsumedEvent",
    "CheckpointCreatedEvent",
    "LockAcquiredEvent",
    "LockReleasedEvent",
    "ReplanRequestedEvent",
    "RuntimeEvent",
    "RuntimeEventType",
    "SessionExpiredEvent",
    "SessionHeartbeatEvent",
    "SessionOpenedEvent",
    "TaskCancelledEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskQueuedEvent",
    "TaskStartedEvent",
    "WorkerAssignedEvent",
    "WorkerReleasedEvent",
    "coerce_runtime_event",
    "event_to_ref",
]
