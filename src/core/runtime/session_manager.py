"""Runtime session manager.

This module manages transient runtime sessions such as shell channels, tunnels
or agent-side execution handles. Sessions are runtime coordination objects
only; they are not persistent KG facts.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from enum import Enum

from src.core.models.runtime import RuntimeState, SessionRuntime, SessionStatus, utc_now

logger = logging.getLogger(__name__)


class SessionReusePolicy(str, Enum):
    """Reuse policy for one runtime session."""

    EXCLUSIVE = "exclusive"
    SHARED_READONLY = "shared_readonly"
    SHARED = "shared"

    @classmethod
    def coerce(cls, value: object, default: "SessionReusePolicy" | None = None) -> "SessionReusePolicy":
        """Resolve an arbitrary value to a known policy, falling back instead of raising.

        Tool callers (including the LLM) can supply free-form reuse_policy strings.
        An unknown value must never crash result write-back, so coerce to the most
        restrictive default (EXCLUSIVE) and record a warning for diagnosis.
        """

        fallback = default or cls.EXCLUSIVE
        if isinstance(value, cls):
            return value
        if value is None or value == "":
            return fallback
        try:
            return cls(str(value))
        except ValueError:
            logger.warning(
                "unknown session reuse_policy %r; falling back to %s", value, fallback.value
            )
            return fallback


class RuntimeSessionManager:
    """Manage runtime session lifecycle, lease refresh and reuse decisions."""

    def open_session(
        self,
        state: RuntimeState,
        session_id: str,
        bound_identity: str | None,
        bound_target: str | None,
        lease_seconds: int,
        reusability: str = "exclusive",
    ) -> SessionRuntime:
        """Create or refresh one runtime session with a new lease window."""

        now = utc_now()
        policy = SessionReusePolicy.coerce(reusability)
        lease_expiry = now + timedelta(seconds=lease_seconds)
        session = state.sessions.get(session_id)

        if session is None:
            session = SessionRuntime(
                session_id=session_id,
                status=SessionStatus.ACTIVE,
                bound_identity=bound_identity,
                bound_target=bound_target,
                lease_expiry=lease_expiry,
                heartbeat_at=now,
                reusability=self._model_reusability(policy),
                metadata={
                    "reuse_policy": policy.value,
                    "bound_execution_ids": [],
                },
            )
            state.sessions[session_id] = session
        else:
            session.status = SessionStatus.ACTIVE
            session.bound_identity = bound_identity
            session.bound_target = bound_target
            session.lease_expiry = lease_expiry
            session.heartbeat_at = now
            session.reusability = self._model_reusability(policy)
            session.metadata["reuse_policy"] = policy.value
            session.metadata.setdefault("bound_execution_ids", [])

        state.last_updated = now
        return session

    def get_session(self, state: RuntimeState, session_id: str) -> SessionRuntime:
        """Return one tracked runtime session."""

        try:
            return state.sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"session '{session_id}' does not exist") from exc

    def is_session_usable(self, state: RuntimeState, session_id: str) -> bool:
        """Return True when a session is active and lease-valid."""

        session = self.get_session(state, session_id)
        if session.status == SessionStatus.EXPIRED:
            return False
        return session.is_session_usable()

    def bind_execution_to_session(self, state: RuntimeState, execution_id: str, session_id: str) -> SessionRuntime:
        """Attach one source execution ID to the session metadata."""

        session = self.get_session(state, session_id)
        execution_ids = session.metadata.setdefault("bound_execution_ids", [])
        if execution_id not in execution_ids:
            execution_ids.append(execution_id)
        state.last_updated = utc_now()
        return session

    @staticmethod
    def _model_reusability(policy: SessionReusePolicy) -> str:
        """Map external reuse policy names onto the current SessionRuntime field."""

        mapping = {
            SessionReusePolicy.EXCLUSIVE: "single_use",
            SessionReusePolicy.SHARED_READONLY: "reusable",
            SessionReusePolicy.SHARED: "sticky",
        }
        return mapping[policy]


__all__ = ["RuntimeSessionManager", "SessionReusePolicy"]
