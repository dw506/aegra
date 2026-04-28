"""Runtime session manager.

This module manages transient runtime sessions such as shell channels, tunnels
or agent-side execution handles. Sessions are runtime coordination objects
only; they are not persistent KG facts.
"""

from __future__ import annotations

from datetime import timedelta
from enum import Enum

from src.core.models.runtime import RuntimeState, SessionRuntime, SessionStatus, utc_now


class SessionReusePolicy(str, Enum):
    """Reuse policy for one runtime session."""

    EXCLUSIVE = "exclusive"
    SHARED_READONLY = "shared_readonly"
    SHARED = "shared"


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
        policy = SessionReusePolicy(reusability)
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
                    "bound_task_ids": [],
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
            session.metadata.setdefault("bound_task_ids", [])

        state.last_updated = now
        return session

    def heartbeat_session(self, state: RuntimeState, session_id: str) -> SessionRuntime:
        """Refresh the heartbeat timestamp for one runtime session."""

        session = self.get_session(state, session_id)
        session.heartbeat_at = utc_now()
        state.last_updated = session.heartbeat_at
        return session

    def expire_session(
        self,
        state: RuntimeState,
        session_id: str,
        reason: str | None = None,
    ) -> SessionRuntime:
        """Mark one session as expired and record the optional reason."""

        session = self.get_session(state, session_id)
        now = utc_now()
        session.status = SessionStatus.EXPIRED
        session.lease_expiry = now
        session.heartbeat_at = now
        session.failure_count += 1
        if reason is not None:
            session.metadata["expiry_reason"] = reason
        state.last_updated = now
        return session

    def fail_session(
        self,
        state: RuntimeState,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> SessionRuntime:
        """Mark one session as failed and record the optional reason."""

        session = self.get_session(state, session_id)
        now = utc_now()
        session.status = SessionStatus.FAILED
        session.lease_expiry = now
        session.heartbeat_at = now
        session.failure_count += 1
        if reason is not None:
            session.metadata["failure_reason"] = reason
        state.last_updated = now
        return session

    def close_session(
        self,
        state: RuntimeState,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> SessionRuntime:
        """Close one session without treating it as a failure."""

        session = self.get_session(state, session_id)
        now = utc_now()
        session.status = SessionStatus.CLOSED
        session.lease_expiry = now
        session.heartbeat_at = now
        if reason is not None:
            session.metadata["close_reason"] = reason
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

    def extend_lease(self, state: RuntimeState, session_id: str, extra_seconds: int) -> SessionRuntime:
        """Extend the session lease by the requested number of seconds."""

        session = self.get_session(state, session_id)
        now = utc_now()
        if session.status == SessionStatus.EXPIRED:
            raise ValueError(f"session '{session_id}' is expired")
        base = session.lease_expiry if session.lease_expiry is not None and session.lease_expiry > now else now
        session.lease_expiry = base + timedelta(seconds=extra_seconds)
        session.heartbeat_at = now
        state.last_updated = now
        return session

    def bind_task_to_session(self, state: RuntimeState, task_id: str, session_id: str) -> SessionRuntime:
        """Attach one task ID to the session metadata and task runtime when present."""

        session = self.get_session(state, session_id)
        task_ids = session.metadata.setdefault("bound_task_ids", [])
        if task_id not in task_ids:
            task_ids.append(task_id)
        if task_id in state.execution.tasks:
            state.execution.tasks[task_id].metadata["session_id"] = session_id
        state.last_updated = utc_now()
        return session

    def unbind_task_from_session(
        self,
        state: RuntimeState,
        task_id: str,
        session_id: str,
    ) -> SessionRuntime:
        """Remove one task ID from the session metadata and task runtime when present."""

        session = self.get_session(state, session_id)
        task_ids = session.metadata.setdefault("bound_task_ids", [])
        session.metadata["bound_task_ids"] = [item for item in task_ids if item != task_id]
        if task_id in state.execution.tasks:
            task = state.execution.tasks[task_id]
            if task.metadata.get("session_id") == session_id:
                task.metadata.pop("session_id", None)
        state.last_updated = utc_now()
        return session

    def cleanup_expired_sessions(self, state: RuntimeState) -> int:
        """Mark active sessions as expired when their lease has elapsed."""

        now = utc_now()
        expired = 0
        for session in state.sessions.values():
            if session.status == SessionStatus.EXPIRED:
                continue
            if session.lease_expiry is None:
                continue
            if session.lease_expiry <= now:
                session.status = SessionStatus.EXPIRED
                session.failure_count += 1
                session.metadata.setdefault("expiry_reason", "lease_expired")
                expired += 1
        if expired:
            state.last_updated = now
        return expired

    def list_reusable_sessions(
        self,
        state: RuntimeState,
        bound_target: str | None = None,
        bound_identity: str | None = None,
    ) -> list[SessionRuntime]:
        """Return currently usable shared sessions, optionally filtered by bindings."""

        self.cleanup_expired_sessions(state)
        sessions: list[SessionRuntime] = []
        for session in state.sessions.values():
            if not session.is_session_usable():
                continue
            policy = SessionReusePolicy(session.metadata.get("reuse_policy", SessionReusePolicy.EXCLUSIVE.value))
            if policy == SessionReusePolicy.EXCLUSIVE:
                continue
            if bound_target is not None and session.bound_target != bound_target:
                continue
            if bound_identity is not None and session.bound_identity != bound_identity:
                continue
            sessions.append(session)
        return sorted(sessions, key=lambda item: item.session_id)

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
