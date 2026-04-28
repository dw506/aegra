"""Runtime State query service.

This module exposes read-only runtime queries for scheduler, critic and
replanner flows. The queries operate only on Runtime State and do not inspect
KG, AG or TG structures directly.
"""

from __future__ import annotations

from src.core.models.runtime import (
    ResourceLock,
    RuntimeState,
    SessionRuntime,
    SessionStatus,
    TaskRuntime,
    TaskRuntimeStatus,
    WorkerRuntime,
    WorkerStatus,
)
from src.core.runtime.budgets import RuntimeBudgetManager


class RuntimeQueryService:
    """Read-only query helpers over Runtime State."""

    def find_active_tasks(self, state: RuntimeState) -> list[TaskRuntime]:
        """Return tasks currently claimed or running in runtime."""

        return sorted(
            (
                task
                for task in state.execution.tasks.values()
                if task.status in {TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.RUNNING}
            ),
            key=lambda item: item.task_id,
        )

    def find_queued_tasks(self, state: RuntimeState) -> list[TaskRuntime]:
        """Return tasks currently queued in runtime."""

        return sorted(
            (
                task
                for task in state.execution.tasks.values()
                if task.status == TaskRuntimeStatus.QUEUED
            ),
            key=lambda item: item.task_id,
        )

    def find_failed_tasks(self, state: RuntimeState) -> list[TaskRuntime]:
        """Return tasks currently marked as failed in runtime."""

        return sorted(
            (
                task
                for task in state.execution.tasks.values()
                if task.status in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.TIMED_OUT}
            ),
            key=lambda item: item.task_id,
        )

    def find_retryable_tasks(self, state: RuntimeState) -> list[TaskRuntime]:
        """Return failed tasks that still have retry budget."""

        return sorted(
            (
                task
                for task in state.execution.tasks.values()
                if task.is_task_retryable()
            ),
            key=lambda item: item.task_id,
        )

    def find_idle_workers(self, state: RuntimeState) -> list[WorkerRuntime]:
        """Return workers currently available for new assignments."""

        return sorted(
            (
                worker
                for worker in state.workers.values()
                if worker.status == WorkerStatus.IDLE and worker.current_task_id is None
            ),
            key=lambda item: item.worker_id,
        )

    def find_busy_workers(self, state: RuntimeState) -> list[WorkerRuntime]:
        """Return workers currently serving a runtime task."""

        return sorted(
            (
                worker
                for worker in state.workers.values()
                if worker.status == WorkerStatus.BUSY or worker.current_task_id is not None
            ),
            key=lambda item: item.worker_id,
        )

    def find_usable_sessions(
        self,
        state: RuntimeState,
        bound_target: str | None = None,
    ) -> list[SessionRuntime]:
        """Return runtime-usable sessions, optionally filtered by bound target."""

        sessions = [
            session
            for session in state.sessions.values()
            if session.status == SessionStatus.ACTIVE and session.is_session_usable()
        ]
        if bound_target is not None:
            sessions = [session for session in sessions if session.bound_target == bound_target]
        return sorted(sessions, key=lambda item: item.session_id)

    def find_locks_for_task(self, state: RuntimeState, task_id: str) -> list[ResourceLock]:
        """Return runtime locks currently owned by the given task."""

        lock_map = self._lock_map(state)
        return sorted(
            (
                lock
                for lock in lock_map.values()
                if lock.owner_type == "task"
                and lock.owner_id == task_id
                and not lock.is_lock_expired()
            ),
            key=lambda item: item.lock_key,
        )

    def is_task_blocked_by_runtime(
        self,
        state: RuntimeState,
        task_id: str,
        required_resource_keys: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> bool:
        """Return True when runtime conditions block task execution.

        This method is useful when TG considers a task ready but runtime-level
        resource locks or lease issues still prevent dispatch.
        """

        task = state.execution.tasks.get(task_id)
        if task is not None and task.status == TaskRuntimeStatus.BLOCKED:
            return True

        keys = set(required_resource_keys or [])
        if task is not None:
            keys.update(task.resource_keys)

        for resource_key in keys:
            lock = self._lock_map(state).get(resource_key)
            if lock is None or lock.is_lock_expired():
                continue
            if lock.owner_type == "task" and lock.owner_id == task_id:
                continue
            return True
        return False

    def remaining_budget_summary(self, state: RuntimeState) -> dict[str, int | float | None]:
        """Return remaining runtime budgets."""

        return RuntimeBudgetManager().remaining_budget_summary(state)

    def find_replan_requests(self, state: RuntimeState) -> list[dict]:
        """Return runtime replan requests in creation order."""

        return [
            request.model_dump(mode="json")
            for request in sorted(state.replan_requests, key=lambda item: (item.created_at, item.request_id))
        ]

    def latest_outcomes(self, state: RuntimeState, limit: int = 10) -> list[dict]:
        """Return the newest runtime outcomes up to the requested limit."""

        ordered = sorted(
            state.recent_outcomes,
            key=lambda item: (item.created_at, item.outcome_id),
            reverse=True,
        )
        return [item.model_dump(mode="json") for item in ordered[:limit]]

    def build_scheduler_view(self, state: RuntimeState) -> dict:
        """Build a compact scheduler-facing summary of Runtime State."""

        queued = self.find_queued_tasks(state)
        active = self.find_active_tasks(state)
        retryable = self.find_retryable_tasks(state)
        return {
            "operation_id": state.operation_id,
            "operation_status": state.operation_status.value,
            "queued_task_ids": [task.task_id for task in queued],
            "active_task_ids": [task.task_id for task in active],
            "retryable_task_ids": [task.task_id for task in retryable],
            "idle_worker_ids": [worker.worker_id for worker in self.find_idle_workers(state)],
            "busy_worker_ids": [worker.worker_id for worker in self.find_busy_workers(state)],
            "usable_session_ids": [session.session_id for session in self.find_usable_sessions(state)],
            "remaining_budgets": self.remaining_budget_summary(state),
            "event_cursor": state.event_cursor,
            "last_updated": state.last_updated.isoformat(),
        }

    def build_critic_view(self, state: RuntimeState) -> dict:
        """Build a compact critic-facing summary of Runtime State."""

        failed = self.find_failed_tasks(state)
        retryable = self.find_retryable_tasks(state)
        active_locks = [
            lock.model_dump(mode="json")
            for lock in sorted(self._lock_map(state).values(), key=lambda item: item.lock_key)
            if not lock.is_lock_expired()
        ]
        return {
            "operation_id": state.operation_id,
            "operation_status": state.operation_status.value,
            "failed_tasks": [task.model_dump(mode="json") for task in failed],
            "retryable_tasks": [task.model_dump(mode="json") for task in retryable],
            "replan_requests": self.find_replan_requests(state),
            "latest_outcomes": self.latest_outcomes(state, limit=10),
            "active_locks": active_locks,
            "remaining_budgets": self.remaining_budget_summary(state),
            "pending_event_count": len(state.pending_events),
        }

    @staticmethod
    def _lock_map(state: RuntimeState) -> dict[str, ResourceLock]:
        """Return the runtime lock mapping from state."""

        if hasattr(state, "resource_locks"):
            return getattr(state, "resource_locks")
        return state.locks


__all__ = ["RuntimeQueryService"]

