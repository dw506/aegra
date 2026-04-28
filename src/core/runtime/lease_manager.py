"""Runtime session lease manager.

This module manages transient task-to-session lease records that coordinate
runtime ownership and reuse windows.
"""

from __future__ import annotations

from datetime import timedelta

from src.core.models.runtime import RuntimeState, SessionLeaseRuntime, utc_now


class RuntimeLeaseManager:
    """Manage runtime session lease lifecycle and task/session bindings."""

    def create_lease(
        self,
        state: RuntimeState,
        lease_id: str,
        session_id: str,
        owner_task_id: str,
        *,
        owner_worker_id: str | None = None,
        lease_seconds: int | None = None,
        reuse_policy: str = "exclusive",
        metadata: dict[str, object] | None = None,
    ) -> SessionLeaseRuntime:
        """Create or refresh one session lease record."""

        if session_id not in state.sessions:
            raise ValueError(f"session '{session_id}' does not exist")
        now = utc_now()
        lease_expiry = None if lease_seconds is None else now + timedelta(seconds=lease_seconds)
        lease = state.session_leases.get(lease_id)

        if lease is None:
            lease = SessionLeaseRuntime(
                lease_id=lease_id,
                session_id=session_id,
                owner_task_id=owner_task_id,
                owner_worker_id=owner_worker_id,
                acquired_at=now,
                lease_expiry=lease_expiry,
                reuse_policy=reuse_policy,
                metadata=dict(metadata or {}),
            )
            state.session_leases[lease_id] = lease
        else:
            lease.session_id = session_id
            lease.owner_task_id = owner_task_id
            lease.owner_worker_id = owner_worker_id
            lease.acquired_at = now
            lease.lease_expiry = lease_expiry
            lease.reuse_policy = reuse_policy
            if metadata:
                lease.metadata.update(dict(metadata))

        state.last_updated = now
        return lease

    def extend_lease(self, state: RuntimeState, lease_id: str, extra_seconds: int) -> SessionLeaseRuntime:
        """Extend the lease expiry by the requested number of seconds."""

        lease = self.get_lease(state, lease_id)
        now = utc_now()
        if self._is_released(lease):
            raise ValueError(f"lease '{lease_id}' is released")
        if not lease.is_active(now):
            raise ValueError(f"lease '{lease_id}' is expired")
        base = lease.lease_expiry if lease.lease_expiry is not None and lease.lease_expiry > now else now
        lease.lease_expiry = base + timedelta(seconds=extra_seconds)
        state.last_updated = now
        return lease

    def release_lease(
        self,
        state: RuntimeState,
        lease_id: str,
        *,
        reason: str | None = None,
    ) -> SessionLeaseRuntime:
        """Release one lease immediately and record the optional reason."""

        lease = self.get_lease(state, lease_id)
        now = utc_now()
        lease.lease_expiry = now
        lease.metadata["released_at"] = now.isoformat()
        if reason is not None:
            lease.metadata["release_reason"] = reason
        state.last_updated = now
        return lease

    def cleanup_expired_leases(self, state: RuntimeState) -> int:
        """Mark elapsed leases as expired in metadata and return the count."""

        now = utc_now()
        expired = 0
        for lease in state.session_leases.values():
            if self._is_released(lease):
                continue
            if lease.lease_expiry is None or lease.lease_expiry > now:
                continue
            lease.metadata.setdefault("expired_at", now.isoformat())
            lease.metadata.setdefault("expiry_reason", "lease_expired")
            expired += 1
        if expired:
            state.last_updated = now
        return expired

    def list_leases_for_task(
        self,
        state: RuntimeState,
        task_id: str,
        *,
        active_only: bool = False,
    ) -> list[SessionLeaseRuntime]:
        """Return tracked leases owned by the given task."""

        now = utc_now()
        leases = [lease for lease in state.session_leases.values() if lease.owner_task_id == task_id]
        if active_only:
            leases = [lease for lease in leases if not self._is_released(lease) and lease.is_active(now)]
        return sorted(leases, key=lambda item: item.lease_id)

    def list_leases_for_session(
        self,
        state: RuntimeState,
        session_id: str,
        *,
        active_only: bool = False,
    ) -> list[SessionLeaseRuntime]:
        """Return tracked leases pointing at the given session."""

        now = utc_now()
        leases = [lease for lease in state.session_leases.values() if lease.session_id == session_id]
        if active_only:
            leases = [lease for lease in leases if not self._is_released(lease) and lease.is_active(now)]
        return sorted(leases, key=lambda item: item.lease_id)

    # 中文注释：
    # task 结束时先释放 task 侧 lease，避免 session 仍然活着但旧 owner 没有退出。
    def release_leases_for_task(
        self,
        state: RuntimeState,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> int:
        """Release all active leases owned by the given task."""

        released = 0
        for lease in self.list_leases_for_task(state, task_id, active_only=True):
            self.release_lease(state, lease.lease_id, reason=reason)
            released += 1
        return released

    # 中文注释：
    # session 失效时要把挂在它下面的 lease 一并收掉，避免 scheduler 继续把它当作可复用占用。
    def release_leases_for_session(
        self,
        state: RuntimeState,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> int:
        """Release all active leases attached to the given session."""

        released = 0
        for lease in self.list_leases_for_session(state, session_id, active_only=True):
            self.release_lease(state, lease.lease_id, reason=reason)
            released += 1
        return released

    def bind_lease_to_task_or_session(
        self,
        state: RuntimeState,
        lease_id: str,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionLeaseRuntime:
        """Attach one lease to a task runtime and/or session metadata."""

        if task_id is None and session_id is None:
            raise ValueError("bind_lease_to_task_or_session requires task_id or session_id")
        lease = self.get_lease(state, lease_id)
        if task_id is not None:
            lease.owner_task_id = task_id
            if task_id in state.execution.tasks:
                state.execution.tasks[task_id].metadata["session_lease_id"] = lease_id
        if session_id is not None:
            if session_id not in state.sessions:
                raise ValueError(f"session '{session_id}' does not exist")
            lease.session_id = session_id
            lease_ids = state.sessions[session_id].metadata.setdefault("lease_ids", [])
            if lease_id not in lease_ids:
                lease_ids.append(lease_id)
        state.last_updated = utc_now()
        return lease

    def get_lease(self, state: RuntimeState, lease_id: str) -> SessionLeaseRuntime:
        """Return one tracked session lease."""

        try:
            return state.session_leases[lease_id]
        except KeyError as exc:
            raise ValueError(f"lease '{lease_id}' does not exist") from exc

    @staticmethod
    def _is_released(lease: SessionLeaseRuntime) -> bool:
        return "released_at" in lease.metadata


__all__ = ["RuntimeLeaseManager"]
