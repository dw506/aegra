"""Reducer for applying runtime events to Runtime State.

The reducer is intentionally scoped to Runtime State updates only. It consumes
typed runtime events and returns a new RuntimeState snapshot without touching
KG or AG.
"""

from __future__ import annotations

from typing import Iterable

from src.core.models.runtime import (
    ReplanRequest,
    RuntimeState,
    SessionRuntime,
    SessionStatus,
)
from src.core.runtime.events import (
    BaseRuntimeEvent,
    ReplanRequestedEvent,
    RuntimeEvent,
    SessionOpenedEvent,
    coerce_runtime_event,
)


class RuntimeStateReducer:
    """Apply runtime events to RuntimeState snapshots.

    The reducer follows a mostly pure style: input state is not mutated.
    Instead, a deep copy is created and updated before being returned.
    """

    def apply_event(self, state: RuntimeState, event: BaseRuntimeEvent | RuntimeEvent) -> RuntimeState:
        """Apply one event and return a new RuntimeState snapshot."""

        typed_event = coerce_runtime_event(event)
        next_state = state.model_copy(deep=True)

        if typed_event.operation_id != next_state.operation_id:
            return next_state

        if isinstance(typed_event, SessionOpenedEvent):
            self._apply_session_opened(next_state, typed_event)
        elif isinstance(typed_event, ReplanRequestedEvent):
            self._apply_replan_requested(next_state, typed_event)

        next_state.execution.status = next_state.operation_status
        next_state.last_updated = typed_event.created_at
        return next_state

    def apply_events(
        self,
        state: RuntimeState,
        events: Iterable[BaseRuntimeEvent | RuntimeEvent],
    ) -> RuntimeState:
        """Apply multiple events in order and return the final RuntimeState snapshot."""

        next_state = state
        for event in events:
            next_state = self.apply_event(next_state, event)
        return next_state

    def _apply_session_opened(self, state: RuntimeState, event: SessionOpenedEvent) -> None:
        """Create or refresh a runtime session entry."""

        session = state.sessions.get(event.session_id)
        if session is None:
            state.sessions[event.session_id] = SessionRuntime(
                session_id=event.session_id,
                status=SessionStatus.ACTIVE,
                bound_identity=event.bound_identity,
                bound_target=event.bound_target,
                lease_expiry=event.lease_expiry,
                heartbeat_at=event.created_at,
                reusability=event.reusability,
                metadata=dict(event.payload),
            )
            return
        session.status = SessionStatus.ACTIVE
        session.bound_identity = event.bound_identity
        session.bound_target = event.bound_target
        session.lease_expiry = event.lease_expiry
        session.heartbeat_at = event.created_at
        session.reusability = event.reusability
        session.metadata.update(event.payload)

    def _apply_replan_requested(self, state: RuntimeState, event: ReplanRequestedEvent) -> None:
        """Append a runtime replan request."""

        state.replan_requests.append(
            ReplanRequest(
                request_id=event.request_id,
                created_at=event.created_at,
                reason=event.reason,
                task_ids=list(event.task_ids),
                scope=event.scope,
                metadata=dict(event.payload),
            )
        )


__all__ = ["RuntimeStateReducer"]
