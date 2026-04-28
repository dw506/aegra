"""Runtime resource lock manager.

This module manages transient execution locks stored inside Runtime State.
Locks are runtime coordination primitives only. They are not world facts and
must not be persisted as KG, AG or TG structures.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.runtime import LockStatus, ResourceLock, RuntimeState, utc_now


class LockAcquireResult(BaseModel):
    """Structured result returned when attempting to acquire one or more locks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    acquired: bool
    lock_key: str = Field(min_length=1)
    owner_type: Literal["task", "worker", "session", "operation"]
    owner_id: str = Field(min_length=1)
    conflict_owner_type: Literal["task", "worker", "session", "operation"] | None = None
    conflict_owner_id: str | None = None
    expires_at: object | None = None
    reason: str | None = None


class LockConflictError(RuntimeError):
    """Raised when a lock is already held by a different runtime owner."""


class RuntimeLockManager:
    """Manage Runtime State locks for host, session or arbitrary resource keys."""

    # 中文注释：
    # 这里把“调度资源键”和“真正落到 Runtime 的锁键”分开处理。
    # 例如 credential 需要互斥锁，session 需要区分独占/共享策略。
    def expand_policy_lock_keys(
        self,
        resource_keys: set[str] | list[str] | tuple[str, ...],
        *,
        session_policies: dict[str, str] | None = None,
    ) -> set[str]:
        """Expand logical resource keys into runtime-enforced lock keys."""

        expanded: set[str] = set()
        session_policies = dict(session_policies or {})
        for key in sorted(set(resource_keys)):
            expanded.add(key)
            if key.startswith("credential:"):
                expanded.add(f"mutex:{key}")
                continue
            if key.startswith("session:"):
                session_id = key.split(":", 1)[1]
                policy = session_policies.get(session_id, "exclusive")
                if policy == "shared":
                    expanded.add(f"session-shared:{session_id}")
                elif policy == "shared_readonly":
                    expanded.add(f"session-shared-readonly:{session_id}")
                else:
                    expanded.add(f"session-exclusive:{session_id}")
        return expanded

    # 中文注释：
    # 共享会话只和独占会话冲突；独占会话与任何同 session 活动都冲突。
    def is_session_policy_conflict(
        self,
        state: RuntimeState,
        *,
        session_id: str,
        policy: str,
        owner_id: str | None = None,
    ) -> bool:
        """Return True when the requested session sharing mode conflicts now."""

        locks = self._lock_map(state)
        exclusive_key = f"session-exclusive:{session_id}"
        shared_keys = {f"session-shared:{session_id}", f"session-shared-readonly:{session_id}"}
        active = [
            lock
            for key, lock in locks.items()
            if key in {exclusive_key, *shared_keys}
            and lock.status == LockStatus.ACTIVE
            and not lock.is_lock_expired()
            and (owner_id is None or lock.owner_id != owner_id)
        ]
        if not active:
            return False
        if policy == "exclusive":
            return True
        return any(lock.lock_key == exclusive_key for lock in active)

    def acquire_lock(
        self,
        state: RuntimeState,
        lock_key: str,
        owner_type: Literal["task", "worker", "session", "operation"],
        owner_id: str,
        ttl_seconds: int | None = None,
    ) -> LockAcquireResult:
        """Acquire or refresh one runtime lock for the given owner."""

        self.cleanup_expired_locks(state)
        locks = self._lock_map(state)
        now = utc_now()
        expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None
        current = locks.get(lock_key)

        if current is None or current.status in {LockStatus.RELEASED, LockStatus.EXPIRED, LockStatus.STALE}:
            locks[lock_key] = ResourceLock(
                lock_key=lock_key,
                owner_type=owner_type,
                owner_id=owner_id,
                status=LockStatus.ACTIVE,
                acquired_at=now,
                expires_at=expires_at,
            )
            state.last_updated = now
            return LockAcquireResult(
                acquired=True,
                lock_key=lock_key,
                owner_type=owner_type,
                owner_id=owner_id,
                expires_at=expires_at,
            )

        if current.owner_id == owner_id and current.owner_type == owner_type:
            current.status = LockStatus.ACTIVE
            if ttl_seconds is not None:
                current.expires_at = expires_at
            state.last_updated = now
            return LockAcquireResult(
                acquired=True,
                lock_key=lock_key,
                owner_type=owner_type,
                owner_id=owner_id,
                expires_at=current.expires_at,
                reason="lock refreshed by current owner",
            )

        raise LockConflictError(
            f"lock '{lock_key}' is already held by {current.owner_type}:{current.owner_id}"
        )

    def acquire_many(
        self,
        state: RuntimeState,
        lock_keys: list[str] | set[str] | tuple[str, ...],
        owner_type: Literal["task", "worker", "session", "operation"],
        owner_id: str,
        ttl_seconds: int | None = None,
    ) -> list[LockAcquireResult]:
        """Acquire multiple locks atomically from the caller's perspective.

        If any lock conflicts, previously acquired locks in the same batch are
        released to avoid leaving a partial acquisition behind.
        """

        acquired_keys: list[str] = []
        results: list[LockAcquireResult] = []
        try:
            for lock_key in sorted(set(lock_keys)):
                result = self.acquire_lock(
                    state=state,
                    lock_key=lock_key,
                    owner_type=owner_type,
                    owner_id=owner_id,
                    ttl_seconds=ttl_seconds,
                )
                acquired_keys.append(lock_key)
                results.append(result)
            return results
        except LockConflictError:
            for lock_key in acquired_keys:
                self.release_lock(state, lock_key=lock_key, owner_id=owner_id)
            raise

    def release_lock(
        self,
        state: RuntimeState,
        lock_key: str,
        owner_id: str | None = None,
    ) -> bool:
        """Release one runtime lock when it exists and ownership matches."""

        locks = self._lock_map(state)
        current = locks.get(lock_key)
        if current is None:
            return False
        if owner_id is not None and current.owner_id != owner_id:
            return False
        current.status = LockStatus.RELEASED
        state.last_updated = utc_now()
        return True

    def release_all_for_owner(self, state: RuntimeState, owner_id: str) -> int:
        """Release all active locks currently owned by the given owner."""

        released = 0
        for lock in self._lock_map(state).values():
            if lock.owner_id != owner_id:
                continue
            if lock.status != LockStatus.ACTIVE:
                continue
            lock.status = LockStatus.RELEASED
            released += 1
        if released:
            state.last_updated = utc_now()
        return released

    def is_locked(self, state: RuntimeState, lock_key: str) -> bool:
        """Return True when the given lock key is actively held."""

        self.cleanup_expired_locks(state)
        lock = self.get_lock(state, lock_key)
        return lock is not None and lock.status == LockStatus.ACTIVE

    def get_lock(self, state: RuntimeState, lock_key: str) -> ResourceLock | None:
        """Return the current lock entry for a key when present and active."""

        self.cleanup_expired_locks(state)
        lock = self._lock_map(state).get(lock_key)
        if lock is None:
            return None
        return lock

    def cleanup_expired_locks(self, state: RuntimeState) -> int:
        """Mark active TTL-based locks as expired and return the count."""

        now = utc_now()
        expired = 0
        for lock in self._lock_map(state).values():
            if lock.status != LockStatus.ACTIVE:
                continue
            if lock.is_lock_expired(now):
                lock.status = LockStatus.EXPIRED
                expired += 1
        if expired:
            state.last_updated = now
        return expired

    @staticmethod
    def _lock_map(state: RuntimeState) -> dict[str, ResourceLock]:
        """Return the mutable lock mapping from RuntimeState.

        The current RuntimeState model uses `locks`. If a future version adds
        `resource_locks`, this method keeps the manager backward-compatible.
        """

        if hasattr(state, "resource_locks"):
            return getattr(state, "resource_locks")
        return state.locks


__all__ = ["LockAcquireResult", "LockConflictError", "RuntimeLockManager"]
