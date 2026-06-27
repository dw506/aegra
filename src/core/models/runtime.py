#Aegra 单个 operation 在执行期间的运行时状态模型

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#UTC 时间生成函数
def utc_now() -> datetime:
   
    return datetime.now(timezone.utc)

#描述整个 operation 的生命周期
class RuntimeStatus(str, Enum):

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    QUIESCING = "quiescing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

#描述一个 session 的状态
class SessionStatus(str, Enum):

    OPENING = "opening"
    ACTIVE = "active"
    DRAINING = "draining"    #准备释放，不再接新任务
    CLOSED = "closed"
    FAILED = "failed"
    EXPIRED = "expired"        #租约过期

#描述凭据是否可用
class CredentialStatus(str, Enum):

    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"       #过期
    REVOKED = "revoked"       #被撤销

#描述凭据类型
class CredentialKind(str, Enum):

    PASSWORD = "password"
    TOKEN = "token"
    SSH_KEY = "ssh_key"
    HASH = "hash"
    CERTIFICATE = "certificate"

#描述一条 pivot / jump route 的状态
class PivotRouteStatus(str, Enum):
    """Lifecycle status for one pivot or jump route."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    FAILED = "failed"
    CLOSED = "closed"


class BaseRuntimeModel(BaseModel):
   #Runtime 模型的基类

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeEventRef(BaseRuntimeModel):
   #Runtime event 的轻量引用

    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    cursor: int = Field(default=0, ge=0)      # 事件游标，用来排序或增量处理
    summary: str | None = None
    payload_ref: str | None = None           #事件完整 payload 的引用地址
    metadata: dict[str, Any] = Field(default_factory=dict)    #扩展信息

#定义用于事件 replay 或恢复流程
class ReplayPlanStatus(str, Enum):

    NOT_REQUIRED = "not_required"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

#记录一次 replay 的窗口
class ReplayPlanRuntime(BaseRuntimeModel):

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

#最近任务结果的缓存条目
class OutcomeCacheEntry(BaseRuntimeModel):

    outcome_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    outcome_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    payload_ref: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

#运行时 session
class SessionRuntime(BaseRuntimeModel):

    session_id: str = Field(min_length=1)
    status: SessionStatus = SessionStatus.OPENING
    bound_identity: str | None = None
    bound_target: str | None = None
    lease_expiry: datetime | None = None          #租约过期时间
    heartbeat_at: datetime = Field(default_factory=utc_now)       #最近心跳时间
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

#运行时凭据
class CredentialRuntime(BaseRuntimeModel):

    credential_id: str = Field(min_length=1)
    kind: CredentialKind = CredentialKind.PASSWORD
    principal: str = Field(min_length=1)
    secret_ref: str | None = None                 #secret 的引用
    status: CredentialStatus = CredentialStatus.UNKNOWN            #控制凭据是否可用
    bound_targets: set[str] = Field(default_factory=set)       #表示这个凭据在哪些目标上验证过或绑定过
    source_session_id: str | None = None
    last_validated_at: datetime | None = None
    failure_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

#表示一条 pivot 访问路径
class PivotRouteRuntime(BaseRuntimeModel):

    route_id: str = Field(min_length=1)
    destination_host: str = Field(min_length=1)
    destination_zone: str | None = None
    destination_cidr: str | None = None
    source_host: str | None = None
    via_host: str | None = None
    session_id: str | None = None
    status: PivotRouteStatus = PivotRouteStatus.CANDIDATE
    protocol: str | None = None
    allowed_ports: set[int] = Field(default_factory=set)
    protocols: set[str] = Field(default_factory=set)
    hop_count: int = Field(default=1, ge=1)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_verified_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

#记录预算与策略缓存
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
    approval_cache: dict[str, bool] = Field(default_factory=dict)    #审批缓存 
    policy_flags: dict[str, bool | int | float | str] = Field(default_factory=dict)   #策略标记

#表示一次重新规划请求
class ReplanRequest(BaseRuntimeModel):
    """Small runtime record indicating that local or full replanning is needed."""

    request_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    reason: str = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    scope: Literal["local", "branch", "full"] = "local"
    metadata: dict[str, Any] = Field(default_factory=dict)

#当前 operation 的顶层控制状态
class OperationRuntime(BaseRuntimeModel):

    operation_id: str = Field(min_length=1)
    status: RuntimeStatus = RuntimeStatus.CREATED
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    active_goal_id: str | None = None
    active_phase: str | None = None
    summary: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "OperationRuntime":
        """Ensure operation timestamps remain monotonic when they are present."""

        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at must be greater than or equal to created_at")
        if self.finished_at is not None and self.started_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at")
        return self

#operation 的完整运行时快照
class RuntimeState(BaseRuntimeModel):
    """Aggregated runtime state snapshot for one operation.

    This object is the execution engine's local control-plane snapshot. It is
    designed for high-frequency updates, serialization and event-driven changes.
    """

    operation_id: str = Field(min_length=1)
    operation_status: RuntimeStatus = RuntimeStatus.CREATED

    execution: OperationRuntime                         #当前 operation 本身
    sessions: dict[str, SessionRuntime] = Field(default_factory=dict)      #当前已打开/可复用 session
    credentials: dict[str, CredentialRuntime] = Field(default_factory=dict)   #当前发现/验证过的凭据
    pivot_routes: dict[str, PivotRouteRuntime] = Field(default_factory=dict)  #当前发现/验证过的跳板路径
    budgets: BudgetRuntime = Field(default_factory=BudgetRuntime)             # 执行预算
    pending_events: list[RuntimeEventRef] = Field(default_factory=list)       # 待处理事件
    recent_outcomes: list[OutcomeCacheEntry] = Field(default_factory=list)     #最近执行结果缓存
    replan_requests: list[ReplanRequest] = Field(default_factory=list)          #重规划请求
    event_cursor: int = Field(default=0, ge=0)                               #事件游标
    last_updated: datetime = Field(default_factory=utc_now)                   #  最近更新时间

    @model_validator(mode="after")
    def validate_operation_identity(self) -> "RuntimeState":
        """Ensure the aggregate state and execution section use the same operation ID."""
        #RuntimeState 外层的 operation_id 必须和内部 execution.operation_id 一致
        if self.execution.operation_id != self.operation_id:
            raise ValueError("execution.operation_id must match operation_id")
        return self

    #新增一个 runtime event，并推进 event cursor
    def push_event(self, event_ref: RuntimeEventRef) -> RuntimeEventRef:
        """Append one pending runtime event and advance the local cursor."""

        self.pending_events.append(event_ref)
        self.event_cursor = max(self.event_cursor, event_ref.cursor)
        self.last_updated = utc_now()
        return event_ref
    
    #记录最近执行结果，默认最多保留 50 条
    def record_outcome(self, outcome: OutcomeCacheEntry, keep_last: int = 50) -> OutcomeCacheEntry:
        """Append an outcome cache entry and keep only the most recent entries."""

        self.recent_outcomes.append(outcome)
        if keep_last > 0 and len(self.recent_outcomes) > keep_last:
            self.recent_outcomes[:] = self.recent_outcomes[-keep_last:]
        self.last_updated = utc_now()
        return outcome
    
    #写入一次重规划请求
    def request_replan(self, request: ReplanRequest) -> ReplanRequest:
        """Append one runtime replan request."""

        self.replan_requests.append(request)
        self.last_updated = utc_now()
        return request


__all__ = [
    "BaseRuntimeModel",
    "BudgetRuntime",
    "CredentialKind",
    "CredentialRuntime",
    "CredentialStatus",
    "OperationRuntime",
    "OutcomeCacheEntry",
    "PivotRouteRuntime",
    "PivotRouteStatus",
    "ReplayPlanRuntime",
    "ReplayPlanStatus",
    "ReplanRequest",
    "RuntimeEventRef",
    "RuntimeState",
    "RuntimeStatus",
    "SessionRuntime",
    "SessionStatus",
    "utc_now",
]
