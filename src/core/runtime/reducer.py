"""Reducer for applying runtime events to Runtime State.

The reducer is intentionally scoped to Runtime State updates only. It consumes
typed runtime events and returns a new RuntimeState snapshot without touching
KG, AG or TG.
"""

from __future__ import annotations

from typing import Iterable

from src.core.models.runtime import (
    CheckpointRuntime,
    ReplanRequest,
    ResourceLock,
    RuntimeState,
    RuntimeStatus,
    SessionRuntime,
    SessionStatus,
    TaskRuntime,
    TaskRuntimeStatus,
    WorkerRuntime,
    WorkerStatus,
)
from src.core.runtime.events import (
    BaseRuntimeEvent,
    BudgetConsumedEvent,
    CheckpointCreatedEvent,
    LockAcquiredEvent,
    LockReleasedEvent,
    ReplanRequestedEvent,
    RuntimeEvent,
    SessionExpiredEvent,
    SessionHeartbeatEvent,
    SessionOpenedEvent,
    TaskCancelledEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskQueuedEvent,
    TaskStartedEvent,
    WorkerAssignedEvent,
    WorkerReleasedEvent,
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

        if isinstance(typed_event, TaskQueuedEvent):
            self._apply_task_queued(next_state, typed_event)
        elif isinstance(typed_event, TaskStartedEvent):
            self._apply_task_started(next_state, typed_event)
        elif isinstance(typed_event, TaskCompletedEvent):
            self._apply_task_completed(next_state, typed_event)
        elif isinstance(typed_event, TaskFailedEvent):
            self._apply_task_failed(next_state, typed_event)
        elif isinstance(typed_event, TaskCancelledEvent):
            self._apply_task_cancelled(next_state, typed_event)
        elif isinstance(typed_event, WorkerAssignedEvent):
            self._apply_worker_assigned(next_state, typed_event)
        elif isinstance(typed_event, WorkerReleasedEvent):
            self._apply_worker_released(next_state, typed_event)
        elif isinstance(typed_event, LockAcquiredEvent):
            self._apply_lock_acquired(next_state, typed_event)
        elif isinstance(typed_event, LockReleasedEvent):
            self._apply_lock_released(next_state, typed_event)
        elif isinstance(typed_event, SessionOpenedEvent):
            self._apply_session_opened(next_state, typed_event)
        elif isinstance(typed_event, SessionExpiredEvent):
            self._apply_session_expired(next_state, typed_event)
        elif isinstance(typed_event, SessionHeartbeatEvent):
            self._apply_session_heartbeat(next_state, typed_event)
        elif isinstance(typed_event, BudgetConsumedEvent):
            self._apply_budget_consumed(next_state, typed_event)
        elif isinstance(typed_event, CheckpointCreatedEvent):
            self._apply_checkpoint_created(next_state, typed_event)
        elif isinstance(typed_event, ReplanRequestedEvent):
            self._apply_replan_requested(next_state, typed_event)

        next_state.operation_status = self._derive_operation_status(next_state)
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

    def _apply_task_queued(self, state: RuntimeState, event: TaskQueuedEvent) -> None:
        """Create or update a queued task runtime entry."""

        task = self._ensure_task_runtime(state, task_id=event.task_id, tg_node_id=event.tg_node_id)
        task.status = TaskRuntimeStatus.QUEUED
        task.queued_at = event.created_at
        task.deadline = event.deadline
        task.metadata.update(event.payload)

    def _apply_task_started(self, state: RuntimeState, event: TaskStartedEvent) -> None:
        """Mark a task as running and capture its start timestamp."""

        task = self._ensure_task_runtime(state, task_id=event.task_id, tg_node_id=event.tg_node_id)
        task.status = TaskRuntimeStatus.RUNNING
        task.started_at = event.created_at
        task.assigned_worker = event.worker_id
        task.metadata.update(event.payload)
        if event.session_id is not None:
            task.metadata["session_id"] = event.session_id

    def _apply_task_completed(self, state: RuntimeState, event: TaskCompletedEvent) -> None:
        """Mark a task as succeeded and release any assigned worker."""

        task = self._ensure_task_runtime(state, task_id=event.task_id, tg_node_id=event.tg_node_id)
        released_worker_id = task.assigned_worker
        task.status = TaskRuntimeStatus.SUCCEEDED
        task.finished_at = event.created_at
        task.last_outcome_ref = event.outcome_ref or event.outcome_id
        task.metadata.update(event.payload)
        task.assigned_worker = None
        if released_worker_id is not None and released_worker_id in state.workers:
            worker = state.workers[released_worker_id]
            worker.current_task_id = None
            worker.current_load = 0
            if worker.status == WorkerStatus.BUSY:
                worker.status = WorkerStatus.IDLE

    def _apply_task_failed(self, state: RuntimeState, event: TaskFailedEvent) -> None:
        """Mark a task as failed and increment its attempt counter."""

        task = self._ensure_task_runtime(state, task_id=event.task_id, tg_node_id=event.tg_node_id)
        task.status = TaskRuntimeStatus.FAILED
        task.finished_at = event.created_at
        task.attempt_count = min(task.max_attempts, task.attempt_count + 1)
        task.last_error = event.error_message
        task.metadata.update(event.payload)

    def _apply_task_cancelled(self, state: RuntimeState, event: TaskCancelledEvent) -> None:
        """Mark a task as cancelled."""

        task = self._ensure_task_runtime(state, task_id=event.task_id, tg_node_id=event.tg_node_id)
        task.status = TaskRuntimeStatus.CANCELLED
        task.finished_at = event.created_at
        if event.reason is not None:
            task.last_error = event.reason
        task.metadata.update(event.payload)

    def _apply_worker_assigned(self, state: RuntimeState, event: WorkerAssignedEvent) -> None:
        """Bind a worker to a task and reflect the assignment on both sides."""

        worker = self._ensure_worker_runtime(state, event.worker_id)
        task = self._ensure_task_runtime(state, task_id=event.task_id, tg_node_id=event.task_id)
        worker.current_task_id = event.task_id
        worker.current_load = max(event.current_load, 1)
        worker.status = WorkerStatus.BUSY
        worker.metadata.update(event.payload)
        task.assigned_worker = event.worker_id

    def _apply_worker_released(self, state: RuntimeState, event: WorkerReleasedEvent) -> None:
        """Release a worker from its current task."""

        worker = self._ensure_worker_runtime(state, event.worker_id)
        worker.current_task_id = None
        worker.current_load = max(event.current_load, 0)
        if worker.status == WorkerStatus.BUSY:
            worker.status = WorkerStatus.IDLE
        worker.metadata.update(event.payload)

    def _apply_lock_acquired(self, state: RuntimeState, event: LockAcquiredEvent) -> None:
        """Acquire or refresh a runtime resource lock."""

        state.locks[event.lock_key] = ResourceLock(
            lock_key=event.lock_key,
            owner_type=event.owner_type,
            owner_id=event.owner_id,
            status="active",
            acquired_at=event.created_at,
            expires_at=event.expires_at,
            metadata=dict(event.payload),
        )

    def _apply_lock_released(self, state: RuntimeState, event: LockReleasedEvent) -> None:
        """Release a tracked runtime resource lock when present."""

        lock = state.locks.get(event.lock_key)
        if lock is None:
            return
        lock.status = "released"
        lock.metadata.update(event.payload)

    def _apply_session_opened(self, state: RuntimeState, event: SessionOpenedEvent) -> None:
        """Create or refresh a runtime session entry."""

        session = state.sessions.get(event.session_id)
        if session is None:
            session = SessionRuntime(
                session_id=event.session_id,
                status=SessionStatus.ACTIVE,
                bound_identity=event.bound_identity,
                bound_target=event.bound_target,
                lease_expiry=event.lease_expiry,
                heartbeat_at=event.created_at,
                reusability=event.reusability,
                metadata=dict(event.payload),
            )
            state.sessions[event.session_id] = session
            return
        session.status = SessionStatus.ACTIVE
        session.bound_identity = event.bound_identity
        session.bound_target = event.bound_target
        session.lease_expiry = event.lease_expiry
        session.heartbeat_at = event.created_at
        session.reusability = event.reusability
        session.metadata.update(event.payload)

    def _apply_session_expired(self, state: RuntimeState, event: SessionExpiredEvent) -> None:
        """Mark a tracked session as expired."""

        session = self._ensure_session_runtime(state, event.session_id)
        session.status = SessionStatus.EXPIRED
        session.failure_count = max(session.failure_count, event.failure_count)
        session.metadata.update(event.payload)
        if event.reason is not None:
            session.metadata["expiry_reason"] = event.reason

    def _apply_session_heartbeat(self, state: RuntimeState, event: SessionHeartbeatEvent) -> None:
        """Refresh session heartbeat and optional lease expiry."""

        session = self._ensure_session_runtime(state, event.session_id)
        session.heartbeat_at = event.heartbeat_at
        if event.lease_expiry is not None:
            session.lease_expiry = event.lease_expiry
        session.metadata.update(event.payload)

    def _apply_budget_consumed(self, state: RuntimeState, event: BudgetConsumedEvent) -> None:
        """Accumulate execution budget usage."""

        budgets = state.budgets
        budgets.time_budget_used_sec += event.time_budget_used_sec_delta
        budgets.token_budget_used += event.token_budget_used_delta
        budgets.operation_budget_used += event.operation_budget_used_delta
        budgets.noise_budget_used += event.noise_budget_used_delta
        budgets.risk_budget_used += event.risk_budget_used_delta
        budgets.approval_cache.update(event.approval_updates)
        budgets.policy_flags.update(event.policy_flag_updates)

    def _apply_checkpoint_created(self, state: RuntimeState, event: CheckpointCreatedEvent) -> None:
        """Append or replace a runtime checkpoint marker."""

        state.checkpoints[event.checkpoint_id] = CheckpointRuntime(
            checkpoint_id=event.checkpoint_id,
            created_at=event.created_at,
            created_after_tasks=list(event.created_after_tasks),
            kg_version=event.kg_version,
            ag_version=event.ag_version,
            tg_version=event.tg_version,
            summary=event.summary,
            metadata=dict(event.payload),
        )

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

    @staticmethod
    def _ensure_task_runtime(state: RuntimeState, task_id: str, tg_node_id: str) -> TaskRuntime:
        """Return an existing task runtime or create a minimal placeholder."""

        task = state.execution.tasks.get(task_id)
        if task is None:
            task = TaskRuntime(task_id=task_id, tg_node_id=tg_node_id)
            state.execution.tasks[task_id] = task
        return task

    @staticmethod
    def _ensure_worker_runtime(state: RuntimeState, worker_id: str) -> WorkerRuntime:
        """Return an existing worker runtime or create a minimal placeholder."""

        worker = state.workers.get(worker_id)
        if worker is None:
            worker = WorkerRuntime(worker_id=worker_id)
            state.workers[worker_id] = worker
        return worker

    @staticmethod
    def _ensure_session_runtime(state: RuntimeState, session_id: str) -> SessionRuntime:
        """Return an existing session runtime or create a minimal placeholder."""

        session = state.sessions.get(session_id)
        if session is None:
            session = SessionRuntime(session_id=session_id)
            state.sessions[session_id] = session
        return session

    @staticmethod
    def _derive_operation_status(state: RuntimeState) -> RuntimeStatus:
        """Infer a safe top-level runtime status from task activity."""

        tasks = list(state.execution.tasks.values())
        if not tasks:
            return state.operation_status
        if any(task.status == TaskRuntimeStatus.RUNNING for task in tasks):
            return RuntimeStatus.RUNNING
        if any(task.status in {TaskRuntimeStatus.QUEUED, TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.PENDING} for task in tasks):
            return RuntimeStatus.READY
        if all(task.status in {TaskRuntimeStatus.SUCCEEDED, TaskRuntimeStatus.SKIPPED} for task in tasks):
            return RuntimeStatus.COMPLETED
        if any(task.status == TaskRuntimeStatus.FAILED for task in tasks):
            return RuntimeStatus.FAILED
        if all(task.status == TaskRuntimeStatus.CANCELLED for task in tasks):
            return RuntimeStatus.CANCELLED
        return state.operation_status


__all__ = ["RuntimeStateReducer"]

