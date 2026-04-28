"""KG delta event models exchanged between State Writer and Graph Projection."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import GraphRef


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def new_kg_event_id() -> str:
    """Return a compact unique identifier for KG delta events."""

    return f"kg-event-{uuid4().hex}"


class KGDeltaEventType(str, Enum):
    """Typed KG delta events emitted by the State Writer."""

    ENTITY_ADDED = "entity_added"
    ENTITY_UPDATED = "entity_updated"
    RELATION_ADDED = "relation_added"
    RELATION_UPDATED = "relation_updated"
    CONFIDENCE_CHANGED = "confidence_changed"
    STATE_INVALIDATED = "state_invalidated"


class KGDeltaEvent(BaseModel):
    """One normalized KG change event for downstream graph projection."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    event_id: str = Field(default_factory=new_kg_event_id, min_length=1)
    event_type: KGDeltaEventType
    source_agent: str = Field(min_length=1)
    target_ref: GraphRef
    patch: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KGEventBatch(BaseModel):
    """Batch envelope for passing KG delta events between agents."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    batch_id: str = Field(default_factory=lambda: f"kg-batch-{uuid4().hex}", min_length=1)
    events: list[KGDeltaEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_events(
        cls,
        events: Iterable[KGDeltaEvent | dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "KGEventBatch":
        """Build one batch from pre-built event models or serialized payloads."""

        normalized = [
            event if isinstance(event, KGDeltaEvent) else KGDeltaEvent.model_validate(event)
            for event in events
        ]
        return cls(
            events=normalized,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_single_event(
        cls,
        event: KGDeltaEvent | dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "KGEventBatch":
        """Build a one-event batch for simple producer flows."""

        return cls.from_events([event], metadata=metadata)


__all__ = [
    "KGDeltaEvent",
    "KGDeltaEventType",
    "KGEventBatch",
    "new_kg_event_id",
    "utc_now",
]
