"""Runtime State core models.

This module defines the execution-time state used by the orchestration engine.
The models here intentionally describe only the current execution context for
one operation. They do not replace KG facts, AG planning structure or TG task
topology.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class RuntimeStatus(str, Enum):
    """Lifecycle status for the current operation runtime."""

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    QUIESCING = "quiescing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRuntimeStatus(str, Enum):
    """Execution-only status for one task instance during this run."""

    PENDING = "pending"
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class WorkerStatus(str, Enum):
    """Availability status for one runtime worker."""

    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    UNAVAILABLE = "unavailable"
    LOST = "lost"


class SessionStatus(str, Enum):
    """Lifecycle status for one runtime session handle."""

    OPENING = "opening"
    ACTIVE = "active"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"
    EXPIRED = "expired"


class CredentialStatus(str, Enum):
    """Lifecycle status for one runtime credential handle."""

    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"


class CredentialKind(str, Enum):
    """Credential categories tracked by runtime."""

    PASSWORD = "password"
    TOKEN = "token"
    SSH_KEY = "ssh_key"
    HASH = "hash"
    CERTIFICATE = "certificate"


class PivotRouteStatus(str, Enum):
    """Lifecycle status for one pivot or jump route."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    FAILED = "failed"
    CLOSED = "closed"


class LockStatus(str, Enum):
    """Lifecycle status for one runtime resource lock."""

    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    STALE = "stale"


class BaseRuntimeModel(BaseModel):
    """Shared validation and serialization settings for runtime models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeEventRef(BaseRuntimeModel):
    """Reference to a runtime event waiting to be processed or recently emitted."""

    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    cursor: int = Field(default=0, ge=0)
    summary: str | None = None
    payload_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayPlanStatus(str, Enum):
    """Lifecycle status for one runtime replay planning record."""

    NOT_REQUIRED = "not_required"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ReplayPlanRuntime(BaseRuntimeModel):
    """Lightweight replay planning snapshot kept alongside RuntimeState."""

    plan_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    replay_status: ReplayPlanStatus = ReplayPlanStatus.PLANNED
    replay_reason: str | None = None
    last_replayed_cursor: int = Field(default=0, ge=0)
    start_cursor: int = Field(default=0, ge=0)
    end_cursor: int = Field(default=0, ge=0)
    replay_candidate_event_ids: list[str] = Field(default_factory=list)
    pending_event_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_cursor_window(self) -> "ReplayPlanRuntime":
        """Ensure the replay cursor window remains monotonic."""

        if self.end_cursor < self.start_cursor:
            raise ValueError("end_cursor must be greater than or equal to start_cursor")
        if self.start_cursor < self.last_replayed_cursor:
            raise ValueError("start_cursor must be greater than or equal to last_replayed_cursor")
        return self


class OutcomeCacheEntry(BaseRuntimeModel):
    """Small cache entry describing a task outcome produced during this run."""

    outcome_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    outcome_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    payload_ref: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointRuntime(BaseRuntimeModel):
    """Runtime-visible checkpoint marker created during this operation."""

    checkpoint_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    created_after_tasks: list[str] = Field(default_factory=list)
    kg_version: str | None = None
    ag_version: str | None = None
    tg_version: str | None = None
    summary: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskRuntime(BaseRuntimeModel):
    """Execution-time state for one TG task node.

    This model stores only the task's live execution context. It does not
    duplicate TG edges or rebuild task structure.
    """

    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    status: TaskRuntimeStatus = TaskRuntimeStatus.PENDING
    assigned_worker: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1)
    queued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    deadline: datetime | None = None
    last_error: str | None = None
    last_outcome_ref: str | None = None
    resource_keys: set[str] = Field(default_factory=set)
    checkpoint_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "TaskRuntime":
        """Ensure task timestamps remain monotonic when they are present."""

        if self.started_at is not None and self.queued_at is not None and self.started_at < self.queued_at:
            raise ValueError("started_at must be greater than or equal to queued_at")
        if self.finished_at is not None and self.started_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at")
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count must be less than or equal to max_attempts")
        return self

    def is_task_retryable(self) -> bool:
        """Return True when the task may be retried in the current run."""

        return self.status in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.TIMED_OUT} and self.attempt_count < self.max_attempts


class WorkerRuntime(BaseRuntimeModel):
    """Execution-time state for one worker process or worker agent."""

    worker_id: str = Field(min_length=1)
    status: WorkerStatus = WorkerStatus.IDLE
    current_task_id: str | None = None
    current_load: int = Field(default=0, ge=0)
    capabilities: set[str] = Field(default_factory=set)
    heartbeat_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionRuntime(BaseRuntimeModel):
    """Execution-time state for one leased or reusable session handle."""

    session_id: str = Field(min_length=1)
    status: SessionStatus = SessionStatus.OPENING
    bound_identity: str | None = None
    bound_target: str | None = None
    lease_expiry: datetime | None = None
    heartbeat_at: datetime = Field(default_factory=utc_now)
    reusability: Literal["single_use", "reusable", "sticky"] = "reusable"
    failure_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_session_usable(self, now: datetime | None = None) -> bool:
        """Return True when the session is still usable for new work."""

        current_time = now or utc_now()
        if self.status != SessionStatus.ACTIVE:
            return False
        if self.lease_expiry is not None and self.lease_expiry <= current_time:
            return False
        return True


class CredentialRuntime(BaseRuntimeModel):
    """Execution-time state for one discovered or imported credential."""

    credential_id: str = Field(min_length=1)
    kind: CredentialKind = CredentialKind.PASSWORD
    principal: str = Field(min_length=1)
    secret_ref: str | None = None
    status: CredentialStatus = CredentialStatus.UNKNOWN
    bound_targets: set[str] = Field(default_factory=set)
    source_session_id: str | None = None
    last_validated_at: datetime | None = None
    failure_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_usable(self) -> bool:
        """Return True when the credential may be used for new tasks."""

        return self.status == CredentialStatus.VALID


class SessionLeaseRuntime(BaseRuntimeModel):
    """Lease record that assigns one session to a worker task."""

    lease_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    owner_task_id: str = Field(min_length=1)
    owner_worker_id: str | None = None
    acquired_at: datetime = Field(default_factory=utc_now)
    lease_expiry: datetime | None = None
    reuse_policy: Literal["exclusive", "shared_readonly", "shared"] = "exclusive"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_active(self, now: datetime | None = None) -> bool:
        """Return True when the lease is still valid."""

        current_time = now or utc_now()
        return self.lease_expiry is None or self.lease_expiry > current_time


class PivotRouteRuntime(BaseRuntimeModel):
    """Route state describing how one host may be reached through another."""

    route_id: str = Field(min_length=1)
    destination_host: str = Field(min_length=1)
    source_host: str | None = None
    via_host: str | None = None
    session_id: str | None = None
    status: PivotRouteStatus = PivotRouteStatus.CANDIDATE
    protocol: str | None = None
    last_verified_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_usable(self) -> bool:
        """Return True when the pivot route is active and usable."""

        return self.status == PivotRouteStatus.ACTIVE


class ResourceLock(BaseRuntimeModel):
    """Transient resource lock used to coordinate execution conflicts."""

    lock_key: str = Field(min_length=1)
    owner_type: Literal["operation", "task", "worker", "session"] = "task"
    owner_id: str = Field(min_length=1)
    status: LockStatus = LockStatus.ACTIVE
    acquired_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_expiry(self) -> "ResourceLock":
        """Ensure the lock expiry does not precede acquisition."""

        if self.expires_at is not None and self.expires_at < self.acquired_at:
            raise ValueError("expires_at must be greater than or equal to acquired_at")
        return self

    def is_lock_expired(self, now: datetime | None = None) -> bool:
        """Return True when the lock should no longer be treated as active."""

        current_time = now or utc_now()
        if self.status in {LockStatus.EXPIRED, LockStatus.STALE}:
            return True
        return self.expires_at is not None and self.expires_at <= current_time


class BudgetRuntime(BaseRuntimeModel):
    """Runtime counters and cached policy decisions for this operation."""

    time_budget_used_sec: float = Field(default=0.0, ge=0.0)
    time_budget_max_sec: float | None = Field(default=None, ge=0.0)
    token_budget_used: int = Field(default=0, ge=0)
    token_budget_max: int | None = Field(default=None, ge=0)
    operation_budget_used: int = Field(default=0, ge=0)
    operation_budget_max: int | None = Field(default=None, ge=0)
    noise_budget_used: float = Field(default=0.0, ge=0.0)
    noise_budget_max: float | None = Field(default=None, ge=0.0)
    risk_budget_used: float = Field(default=0.0, ge=0.0)
    risk_budget_max: float | None = Field(default=None, ge=0.0)
    approval_cache: dict[str, bool] = Field(default_factory=dict)
    policy_flags: dict[str, bool | int | float | str] = Field(default_factory=dict)


class ReplanRequest(BaseRuntimeModel):
    """Small runtime record indicating that local or full replanning is needed."""

    request_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    reason: str = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    scope: Literal["local", "branch", "full"] = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationRuntime(BaseRuntimeModel):
    """Top-level control state for the current operation run."""

    operation_id: str = Field(min_length=1)
    status: RuntimeStatus = RuntimeStatus.CREATED
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    active_goal_id: str | None = None
    active_phase: str | None = None
    summary: str = Field(default="")
    tasks: dict[str, TaskRuntime] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "OperationRuntime":
        """Ensure operation timestamps remain monotonic when they are present."""

        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at must be greater than or equal to created_at")
        if self.finished_at is not None and self.started_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at")
        return self

    def task_count(self, status: TaskRuntimeStatus | None = None) -> int:
        """Return the number of tracked tasks, optionally filtered by status."""

        if status is None:
            return len(self.tasks)
        return sum(1 for task in self.tasks.values() if task.status == status)


class RuntimeState(BaseRuntimeModel):
    """Aggregated runtime state snapshot for one operation.

    This object is the execution engine's local control-plane snapshot. It is
    designed for high-frequency updates, serialization and event-driven changes.
    """

    operation_id: str = Field(min_length=1)
    operation_status: RuntimeStatus = RuntimeStatus.CREATED
    execution: OperationRuntime
    workers: dict[str, WorkerRuntime] = Field(default_factory=dict)
    sessions: dict[str, SessionRuntime] = Field(default_factory=dict)
    credentials: dict[str, CredentialRuntime] = Field(default_factory=dict)
    session_leases: dict[str, SessionLeaseRuntime] = Field(default_factory=dict)
    pivot_routes: dict[str, PivotRouteRuntime] = Field(default_factory=dict)
    locks: dict[str, ResourceLock] = Field(default_factory=dict)
    budgets: BudgetRuntime = Field(default_factory=BudgetRuntime)
    checkpoints: dict[str, CheckpointRuntime] = Field(default_factory=dict)
    pending_events: list[RuntimeEventRef] = Field(default_factory=list)
    recent_outcomes: list[OutcomeCacheEntry] = Field(default_factory=list)
    replan_requests: list[ReplanRequest] = Field(default_factory=list)
    event_cursor: int = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_operation_identity(self) -> "RuntimeState":
        """Ensure the aggregate state and execution section use the same operation ID."""

        if self.execution.operation_id != self.operation_id:
            raise ValueError("execution.operation_id must match operation_id")
        return self

    def get_task_runtime(self, task_id: str) -> TaskRuntime:
        """Return the runtime state for one task ID."""

        return self.execution.tasks[task_id]

    def register_task(self, task_runtime: TaskRuntime) -> TaskRuntime:
        """Insert or replace one task runtime entry."""

        self.execution.tasks[task_runtime.task_id] = task_runtime
        self.last_updated = utc_now()
        return task_runtime

    def add_worker(self, worker: WorkerRuntime) -> WorkerRuntime:
        """Insert or replace one worker runtime entry."""

        self.workers[worker.worker_id] = worker
        self.last_updated = utc_now()
        return worker

    def add_session(self, session: SessionRuntime) -> SessionRuntime:
        """Insert or replace one session runtime entry."""

        self.sessions[session.session_id] = session
        self.last_updated = utc_now()
        return session

    def add_credential(self, credential: CredentialRuntime) -> CredentialRuntime:
        """Insert or replace one credential runtime entry."""

        self.credentials[credential.credential_id] = credential
        self.last_updated = utc_now()
        return credential

    def add_session_lease(self, lease: SessionLeaseRuntime) -> SessionLeaseRuntime:
        """Insert or replace one session lease runtime entry."""

        self.session_leases[lease.lease_id] = lease
        self.last_updated = utc_now()
        return lease

    def add_pivot_route(self, route: PivotRouteRuntime) -> PivotRouteRuntime:
        """Insert or replace one pivot route runtime entry."""

        self.pivot_routes[route.route_id] = route
        self.last_updated = utc_now()
        return route

    def add_lock(self, lock: ResourceLock) -> ResourceLock:
        """Insert or replace one resource lock entry."""

        self.locks[lock.lock_key] = lock
        self.last_updated = utc_now()
        return lock

    def add_checkpoint(self, checkpoint: CheckpointRuntime) -> CheckpointRuntime:
        """Insert or replace one checkpoint runtime entry."""

        self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        self.last_updated = utc_now()
        return checkpoint

    def push_event(self, event_ref: RuntimeEventRef) -> RuntimeEventRef:
        """Append one pending runtime event and advance the local cursor."""

        self.pending_events.append(event_ref)
        self.event_cursor = max(self.event_cursor, event_ref.cursor)
        self.last_updated = utc_now()
        return event_ref

    def record_outcome(self, outcome: OutcomeCacheEntry, keep_last: int = 50) -> OutcomeCacheEntry:
        """Append an outcome cache entry and keep only the most recent entries."""

        self.recent_outcomes.append(outcome)
        if keep_last > 0 and len(self.recent_outcomes) > keep_last:
            self.recent_outcomes[:] = self.recent_outcomes[-keep_last:]
        self.last_updated = utc_now()
        return outcome

    def request_replan(self, request: ReplanRequest) -> ReplanRequest:
        """Append one runtime replan request."""

        self.replan_requests.append(request)
        self.last_updated = utc_now()
        return request


__all__ = [
    "BaseRuntimeModel",
    "BudgetRuntime",
    "CheckpointRuntime",
    "CredentialKind",
    "CredentialRuntime",
    "CredentialStatus",
    "LockStatus",
    "OperationRuntime",
    "OutcomeCacheEntry",
    "PivotRouteRuntime",
    "PivotRouteStatus",
    "ReplayPlanRuntime",
    "ReplayPlanStatus",
    "ReplanRequest",
    "ResourceLock",
    "RuntimeEventRef",
    "RuntimeState",
    "RuntimeStatus",
    "SessionLeaseRuntime",
    "SessionRuntime",
    "SessionStatus",
    "TaskRuntime",
    "TaskRuntimeStatus",
    "WorkerRuntime",
    "WorkerStatus",
    "utc_now",
]
