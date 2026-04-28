"""Structured agent-layer protocol models.

This module defines the input and output contracts exchanged between
task-executing workers and the rest of the orchestration system.

The protocol is intentionally ownership-aware:
- Workers never write KG / AG / TG directly.
- State Writer remains the only formal KG writer.
- Graph Projection remains the only formal AG updater.
- Task Builder / Scheduler remain the TG owners.
- Planner and Critic communicate only through structured outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import (
    AgentExecutionResult,
    AgentInput,
    AgentKind,
    AgentOutput,
    GraphRef as ProtocolGraphRef,
)
from src.core.models.ag import GraphRef
from src.core.models.tg import TaskType


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def new_protocol_id(prefix: str) -> str:
    """Return a short unique identifier for protocol objects."""

    return f"{prefix}-{uuid4().hex}"


class AgentRole(str, Enum):
    """Known agent roles in the orchestration system."""

    GOAL_WORKER = "goal_worker"
    RECON_WORKER = "recon_worker"
    ACCESS_WORKER = "access_worker"
    STATE_WRITER = "state_writer"
    GRAPH_PROJECTION = "graph_projection"
    TASK_BUILDER = "task_builder"
    SCHEDULER = "scheduler"
    PLANNER = "planner"
    CRITIC = "critic"


class AgentTaskIntent(str, Enum):
    """High-level intent of one worker request."""

    EXECUTE_TASK = "execute_task"
    COLLECT_EVIDENCE = "collect_evidence"
    VALIDATE_ACCESS = "validate_access"
    VALIDATE_GOAL = "validate_goal"


class AgentResultStatus(str, Enum):
    """Outcome status returned by a worker."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_REPLAN = "needs_replan"
    NOOP = "noop"


class FactWriteKind(str, Enum):
    """Formal write intent categories for the State Writer."""

    ENTITY_UPSERT = "entity_upsert"
    RELATION_UPSERT = "relation_upsert"
    ASSERTION = "assertion"


class ProjectionRequestKind(str, Enum):
    """Request categories consumed by the Graph Projection agent."""

    REFRESH_TARGETS = "refresh_targets"
    REFRESH_ACTIONS = "refresh_actions"
    REFRESH_LOCAL_FRONTIER = "refresh_local_frontier"


class RuntimeControlType(str, Enum):
    """Runtime-side actions requested by a worker result."""

    OPEN_SESSION = "open_session"
    EXTEND_SESSION = "extend_session"
    EXPIRE_SESSION = "expire_session"
    ACQUIRE_LOCKS = "acquire_locks"
    RELEASE_LOCKS = "release_locks"
    CONSUME_BUDGET = "consume_budget"
    CREATE_CHECKPOINT = "create_checkpoint"
    REQUEST_REPLAN = "request_replan"


class CriticSignalSeverity(str, Enum):
    """Severity level for worker-emitted critic-facing signals."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReplanScope(str, Enum):
    """Scope hint for downstream replanning."""

    LOCAL = "local"
    BRANCH = "branch"
    FULL = "full"


class BaseProtocolModel(BaseModel):
    """Shared validation settings for all agent protocol models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AgentExecutionContext(BaseProtocolModel):
    """Execution context passed to one worker.

    This context links the worker request back to the current TG task and
    Runtime State without embedding the full runtime snapshot.
    """

    operation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    task_type: TaskType
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1)
    assigned_worker_id: str | None = None
    session_id: str | None = None
    checkpoint_ref: str | None = None
    deadline: datetime | None = None
    resource_keys: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationRecord(BaseProtocolModel):
    """Structured observation produced by one worker run."""

    observation_id: str = Field(default_factory=lambda: new_protocol_id("obs"))
    category: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    refs: list[GraphRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class EvidenceArtifact(BaseProtocolModel):
    """Evidence artifact reference emitted by one worker result."""

    evidence_id: str = Field(default_factory=lambda: new_protocol_id("evidence"))
    kind: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    payload_ref: str = Field(min_length=1)
    refs: list[GraphRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FactWriteRequest(BaseProtocolModel):
    """Structured write intent consumed by the State Writer.

    Workers emit this request instead of mutating KG directly.
    """

    proposal_id: str = Field(default_factory=lambda: new_protocol_id("fact"))
    kind: FactWriteKind
    source_task_id: str = Field(min_length=1)
    subject_ref: GraphRef | None = None
    relation_type: str | None = None
    object_ref: GraphRef | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)


class ProjectionRequest(BaseProtocolModel):
    """Structured projection request consumed by the Graph Projection agent."""

    request_id: str = Field(default_factory=lambda: new_protocol_id("projection"))
    kind: ProjectionRequestKind
    source_task_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    invalidated_ref_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskCandidateProposal(BaseProtocolModel):
    """Structured task candidate proposal for Task Builder / Scheduler.

    Workers may suggest follow-up work, but they do not mutate TG directly.
    """

    candidate_id: str = Field(default_factory=lambda: new_protocol_id("task-candidate"))
    source_task_id: str = Field(min_length=1)
    task_type: TaskType
    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    target_refs: list[GraphRef] = Field(default_factory=list)
    resource_keys: set[str] = Field(default_factory=set)
    priority_hint: int = Field(default=50, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeBudgetDelta(BaseProtocolModel):
    """Budget deltas requested by one worker result."""

    time_sec: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    operations: int = Field(default=0, ge=0)
    noise: float = Field(default=0.0, ge=0.0)
    risk: float = Field(default=0.0, ge=0.0)


class RuntimeControlRequest(BaseProtocolModel):
    """Runtime-side control request emitted by a worker result."""

    request_id: str = Field(default_factory=lambda: new_protocol_id("runtime"))
    request_type: RuntimeControlType
    source_task_id: str = Field(min_length=1)
    session_id: str | None = None
    lock_keys: list[str] = Field(default_factory=list)
    lease_seconds: int | None = Field(default=None, ge=1)
    reuse_policy: Literal["exclusive", "shared_readonly", "shared"] | None = None
    checkpoint_id: str | None = None
    budget_delta: RuntimeBudgetDelta = Field(default_factory=RuntimeBudgetDelta)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointHint(BaseProtocolModel):
    """Suggestion to create or associate a checkpoint after one worker result."""

    checkpoint_id: str = Field(default_factory=lambda: new_protocol_id("checkpoint"))
    source_task_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    created_after_tasks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplanHint(BaseProtocolModel):
    """Structured replanning hint passed to Critic / Task Builder flows."""

    hint_id: str = Field(default_factory=lambda: new_protocol_id("replan"))
    source_task_id: str = Field(min_length=1)
    scope: ReplanScope = ReplanScope.LOCAL
    reason: str = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    invalidated_ref_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CriticSignal(BaseProtocolModel):
    """Worker-emitted structured signal for the Critic layer."""

    signal_id: str = Field(default_factory=lambda: new_protocol_id("critic"))
    source_task_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    severity: CriticSignalSeverity = CriticSignalSeverity.MEDIUM
    reason: str = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)
    invalidated_ref_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTaskRequest(BaseProtocolModel):
    """Formal input envelope for one worker task run."""

    request_id: str = Field(default_factory=lambda: new_protocol_id("request"))
    agent_role: AgentRole
    intent: AgentTaskIntent = AgentTaskIntent.EXECUTE_TASK
    context: AgentExecutionContext
    task_label: str = Field(min_length=1)
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    target_refs: list[GraphRef] = Field(default_factory=list)
    source_refs: list[GraphRef] = Field(default_factory=list)
    expected_output_refs: list[GraphRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentTaskResult(BaseProtocolModel):
    """Formal output envelope for one worker run."""

    result_id: str = Field(default_factory=lambda: new_protocol_id("result"))
    request_id: str = Field(min_length=1)
    agent_role: AgentRole
    operation_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    tg_node_id: str = Field(min_length=1)
    status: AgentResultStatus
    summary: str = Field(min_length=1)
    error_message: str | None = None
    observations: list[ObservationRecord] = Field(default_factory=list)
    evidence: list[EvidenceArtifact] = Field(default_factory=list)
    fact_write_requests: list[FactWriteRequest] = Field(default_factory=list)
    projection_requests: list[ProjectionRequest] = Field(default_factory=list)
    task_candidate_proposals: list[TaskCandidateProposal] = Field(default_factory=list)
    runtime_requests: list[RuntimeControlRequest] = Field(default_factory=list)
    checkpoint_hints: list[CheckpointHint] = Field(default_factory=list)
    replan_hints: list[ReplanHint] = Field(default_factory=list)
    critic_signals: list[CriticSignal] = Field(default_factory=list)
    outcome_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentResultAdapter:
    """集中式 worker 结果适配器。

    中文注释：
    - `AgentTaskResult` 是 worker 结果的 canonical result。
    - `AgentOutput` 只保留为 agent 层 transport，不再作为 phase-two 的正式落地协议。
    - 所有从 worker `AgentOutput` 到 `AgentTaskResult` 的转换都集中在这里，避免多处维护分叉逻辑。
    """

    @classmethod
    def to_task_result(
        cls,
        result: AgentTaskResult | AgentExecutionResult | AgentOutput,
        *,
        agent_input: AgentInput | None = None,
        agent_name: str | None = None,
        agent_kind: AgentKind | None = None,
    ) -> AgentTaskResult:
        if isinstance(result, AgentTaskResult):
            return result
        if isinstance(result, AgentExecutionResult):
            return cls.agent_output_to_task_result(
                agent_input=agent_input,
                agent_output=result.output,
                agent_name=result.agent_name,
                agent_kind=result.agent_kind,
            )
        return cls.agent_output_to_task_result(
            agent_input=agent_input,
            agent_output=result,
            agent_name=agent_name,
            agent_kind=agent_kind,
        )

    @classmethod
    def agent_output_to_task_result(
        cls,
        *,
        agent_input: AgentInput | None,
        agent_output: AgentOutput,
        agent_name: str | None,
        agent_kind: AgentKind | None,
    ) -> AgentTaskResult:
        if agent_kind is not None and agent_kind != AgentKind.WORKER:
            raise ValueError("AgentResultAdapter only adapts worker AgentOutput values")
        if agent_input is None:
            raise ValueError("agent_input is required when adapting AgentOutput into AgentTaskResult")

        explicit = agent_input.raw_payload.get("agent_task_result")
        if isinstance(explicit, AgentTaskResult):
            return explicit
        if isinstance(explicit, dict) and explicit:
            return AgentTaskResult.model_validate(explicit)

        task_id = agent_input.task_ref or cls._string(agent_input.raw_payload.get("task_id"))
        if not task_id:
            raise ValueError("worker result adaptation requires task_ref or raw_payload.task_id")

        evidence = [
            EvidenceArtifact(
                kind=str(item.get("result_type") or item.get("kind") or "worker_result"),
                summary=str(item.get("summary") or "worker evidence"),
                payload_ref=str(item.get("payload_ref") or f"runtime://worker-results/{task_id}"),
                refs=cls._graph_refs(item.get("refs")),
                metadata=cls._mapping(item.get("extra"))
                | {
                    key: value
                    for key, value in dict(item).items()
                    if key not in {"task_id", "source_agent", "result_type", "summary", "payload_ref", "refs", "extra"}
                },
            )
            for item in agent_output.evidence
            if isinstance(item, dict)
        ]
        observations = [
            ObservationRecord(
                category=str(item.get("category") or "worker_output"),
                summary=str(item.get("summary") or "worker observation"),
                confidence=float(item.get("confidence", 0.5)),
                refs=cls._graph_refs(item.get("refs")),
                payload=cls._mapping(item.get("payload")),
            )
            for item in agent_output.observations
            if isinstance(item, dict)
        ]
        outcome = agent_output.outcomes[0] if agent_output.outcomes else {}
        status = cls._status_from_output(agent_output=agent_output, outcome=outcome)
        summary = str(
            (outcome.get("summary") if isinstance(outcome, dict) else None)
            or (evidence[0].summary if evidence else None)
            or f"worker completed task {task_id}"
        )
        return AgentTaskResult(
            request_id=str(agent_input.decision_ref or f"{agent_name or 'worker'}-{task_id}"),
            agent_role=cls._agent_role(agent_input),
            operation_id=agent_input.context.operation_id,
            task_id=task_id,
            tg_node_id=str(agent_input.raw_payload.get("tg_node_id") or task_id),
            status=status,
            summary=summary,
            error_message="\n".join(agent_output.errors) or None,
            observations=observations,
            evidence=evidence,
            outcome_payload=dict(outcome) if isinstance(outcome, dict) else {},
            metadata={
                "adapted_from": "agent_output",
                "source_agent": agent_name,
            },
        )

    @staticmethod
    def _agent_role(agent_input: AgentInput) -> AgentRole:
        raw_role = agent_input.raw_payload.get("agent_role")
        if raw_role is not None:
            return raw_role if isinstance(raw_role, AgentRole) else AgentRole(str(raw_role))
        task_type = str(agent_input.raw_payload.get("task_type") or "").upper()
        if "GOAL" in task_type:
            return AgentRole.GOAL_WORKER
        if "PRIVILEGE" in task_type or "ACCESS" in task_type:
            return AgentRole.ACCESS_WORKER
        return AgentRole.RECON_WORKER

    @staticmethod
    def _status_from_output(*, agent_output: AgentOutput, outcome: dict[str, Any] | Any) -> AgentResultStatus:
        if agent_output.errors:
            if any("block" in str(err).lower() for err in agent_output.errors):
                return AgentResultStatus.BLOCKED
            return AgentResultStatus.FAILED
        if isinstance(outcome, dict) and not bool(outcome.get("success", True)):
            return AgentResultStatus.FAILED
        return AgentResultStatus.SUCCEEDED

    @staticmethod
    def _graph_refs(value: Any) -> list[GraphRef]:
        if not isinstance(value, list):
            return []
        refs: list[GraphRef] = []
        for item in value:
            if isinstance(item, GraphRef):
                refs.append(item)
            elif isinstance(item, ProtocolGraphRef):
                refs.append(GraphRef.model_validate(item.model_dump(mode="json")))
            elif isinstance(item, dict):
                try:
                    refs.append(GraphRef.model_validate(item))
                except Exception:
                    continue
        return refs

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = [
    "AgentResultAdapter",
    "AgentExecutionContext",
    "AgentResultStatus",
    "AgentRole",
    "AgentTaskIntent",
    "AgentTaskRequest",
    "AgentTaskResult",
    "BaseProtocolModel",
    "CheckpointHint",
    "CriticSignal",
    "CriticSignalSeverity",
    "EvidenceArtifact",
    "FactWriteKind",
    "FactWriteRequest",
    "ObservationRecord",
    "ProjectionRequest",
    "ProjectionRequestKind",
    "ReplanHint",
    "ReplanScope",
    "RuntimeBudgetDelta",
    "RuntimeControlRequest",
    "RuntimeControlType",
    "TaskCandidateProposal",
    "new_protocol_id",
    "utc_now",
]
