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

    SESSION_OPENED = "SessionOpened"
    REPLAN_REQUESTED = "ReplanRequested"


class BaseRuntimeEvent(BaseModel):
    """Common metadata shared by all runtime events."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: RuntimeEventType
    operation_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionOpenedEvent(BaseRuntimeEvent):
    """Event emitted when runtime opens or registers a session."""

    event_type: Literal[RuntimeEventType.SESSION_OPENED] = RuntimeEventType.SESSION_OPENED
    session_id: str = Field(min_length=1)
    bound_identity: str | None = None
    bound_target: str | None = None
    lease_expiry: datetime | None = None
    reusability: Literal["single_use", "reusable", "sticky"] = "reusable"


class ReplanRequestedEvent(BaseRuntimeEvent):
    """Event emitted when runtime requests local or full replanning."""

    event_type: Literal[RuntimeEventType.REPLAN_REQUESTED] = RuntimeEventType.REPLAN_REQUESTED
    request_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    scope: Literal["local", "branch", "full"] = "local"


RuntimeEvent = Annotated[
    SessionOpenedEvent | ReplanRequestedEvent,
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
    summary_parts = [typed_event.event_type.value, typed_event.operation_id]
    if isinstance(typed_event, SessionOpenedEvent):
        summary_parts.append(typed_event.session_id)
    elif isinstance(typed_event, ReplanRequestedEvent):
        payload_ref = typed_event.request_id
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
    "ReplanRequestedEvent",
    "RuntimeEvent",
    "RuntimeEventType",
    "SessionOpenedEvent",
    "coerce_runtime_event",
    "event_to_ref",
]
