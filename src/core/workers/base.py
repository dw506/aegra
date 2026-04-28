"""Base abstractions for worker agents.

Workers execute tasks and emit structured outputs. They do not directly mutate
KG / AG / TG primary structures. Any graph or runtime effects must be expressed
through `AgentOutput` payloads and consumed by the proper owning component.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import OutcomeRecord
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.models.events import (
    AgentExecutionContext,
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    AgentTaskResult,
)
from src.core.models.runtime import RuntimeState, TaskRuntime
from src.core.models.tg import BaseTaskNode, TaskType


class WorkerCapability(str, Enum):
    """Capabilities that a worker agent may advertise."""

    RECON = "recon"
    CONTEXT_VALIDATION = "context_validation"
    ACCESS_VALIDATION = "access_validation"
    PRIVILEGE_VALIDATION = "privilege_validation"
    GOAL_VALIDATION = "goal_validation"


class WorkerTaskSpec(BaseModel):
    """Minimal task specification consumed by worker agents.

    Attributes:
        task_id: Runtime-facing task identifier.
        task_type: Logical task type name understood by the worker.
        input_bindings: Structured task inputs resolved by TG / scheduler.
        target_refs: Related KG / AG / TG / Runtime references.
        resource_keys: Runtime resource keys relevant to execution.
        constraints: Extra execution constraints such as session or approval
            requirements.
        timeout_seconds: Optional per-task execution timeout hint.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(min_length=1, description="Runtime-facing task identifier.")
    task_type: str = Field(min_length=1, description="Logical task type handled by the worker.")
    input_bindings: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured resolved input bindings for the task.",
    )
    target_refs: list[GraphRef] = Field(
        default_factory=list,
        description="Graph and runtime references associated with the task.",
    )
    resource_keys: list[str] = Field(
        default_factory=list,
        description="Runtime resource keys relevant to worker execution.",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Worker-facing execution constraints and hints.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Optional worker execution timeout in seconds.",
    )


class BaseWorkerAgent(BaseAgent):
    """Base class for worker agents.

    Worker agents have a fixed `worker` kind and a conservative default write
    profile:
    - no KG / AG / TG structural writes,
    - no KG / AG / TG state writes,
    - optional Runtime state deltas only when a subclass explicitly asks for
      them through `write_permission`.

    Concrete workers implement `supports_task()` and `execute_task()`.
    """

    capabilities: frozenset[WorkerCapability] = frozenset()

    def __init__(
        self,
        name: str,
        *,
        write_permission: WritePermission | None = None,
    ) -> None:
        """Initialize the worker with fixed worker kind and safe permissions."""

        super().__init__(
            name=name,
            kind=AgentKind.WORKER,
            write_permission=write_permission
            or WritePermission(
                scopes=[GraphScope.RUNTIME],
                allow_structural_write=False,
                allow_state_write=False,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Validate common worker input requirements."""

        super().validate_input(agent_input)
        if not agent_input.task_ref:
            raise ValueError("worker input requires task_ref")

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Standard `BaseAgent` entrypoint for worker agents."""

        task_spec = self._task_spec_from_input(agent_input)
        if not self.supports_task(task_spec):
            raise ValueError(f"{self.name} does not support task type '{task_spec.task_type}'")
        output = self.execute_task(task_spec, agent_input)
        self._enforce_worker_output_policy(output)
        return output

    @abstractmethod
    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        """Return True when this worker can execute the given task."""

    @abstractmethod
    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        """Execute one task spec and return a structured agent output."""

    def build_outcome(
        self,
        *,
        task_id: str,
        outcome_type: str,
        success: bool,
        summary: str,
        raw_result_ref: str | None = None,
        confidence: float = 0.5,
        refs: list[GraphRef] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> OutcomeRecord:
        """Build a standard worker outcome record.

        Worker outputs are expected to carry `OutcomeRecord` objects so downstream
        runtime, checkpoint and critic flows can consume a consistent shape.
        """

        return OutcomeRecord(
            source_agent=self.name,
            task_id=task_id,
            outcome_type=outcome_type,
            success=success,
            raw_result_ref=raw_result_ref,
            summary=summary,
            confidence=confidence,
            refs=refs or [],
            payload=payload or {},
        )

    def build_raw_result(
        self,
        *,
        task_id: str,
        result_type: str,
        summary: str,
        payload_ref: str,
        refs: list[GraphRef] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a lightweight raw-result pointer for `AgentOutput.evidence`.

        This method intentionally returns a reference object rather than any
        direct graph mutation payload.
        """

        return {
            "task_id": task_id,
            "source_agent": self.name,
            "result_type": result_type,
            "summary": summary,
            "payload_ref": payload_ref,
            "refs": [ref.model_dump(mode="json") for ref in refs or []],
            "extra": extra or {},
        }

    def describe_capabilities(self) -> dict[str, Any]:
        """Return the base capability summary with worker-specific fields."""

        data = super().describe_capabilities()
        data["worker_capabilities"] = sorted(cap.value for cap in self.capabilities)
        return data

    def _task_spec_from_input(self, agent_input: AgentInput) -> WorkerTaskSpec:
        """Build a worker task spec from a generic `AgentInput` envelope."""

        raw = dict(agent_input.raw_payload)
        return WorkerTaskSpec(
            task_id=agent_input.task_ref or raw.get("task_id") or "unknown-task",
            task_type=str(raw.get("task_type") or raw.get("type") or "unknown"),
            input_bindings=dict(raw.get("input_bindings", {})),
            target_refs=list(agent_input.graph_refs),
            resource_keys=list(raw.get("resource_keys", [])),
            constraints=dict(raw.get("constraints", {})),
            timeout_seconds=raw.get("timeout_seconds"),
        )

    def _enforce_worker_output_policy(self, output: AgentOutput) -> None:
        """Apply worker-specific output restrictions."""

        for delta in output.state_deltas:
            scope_value = delta.get("scope")
            if scope_value is None:
                raise PermissionError("worker state_deltas must declare scope")
            scope = scope_value if isinstance(scope_value, GraphScope) else GraphScope(str(scope_value).lower())
            if scope in {GraphScope.KG, GraphScope.AG, GraphScope.TG}:
                raise PermissionError("worker agents may not emit KG/AG/TG state_deltas directly")

        for outcome in output.outcomes:
            if not isinstance(outcome, dict):
                raise TypeError("worker outcomes must be serialized OutcomeRecord fragments")
            if "task_id" not in outcome or "outcome_type" not in outcome:
                raise ValueError("worker outcomes must include OutcomeRecord fields")


class WorkerExecutionError(RuntimeError):
    """Raised when a compatibility worker cannot complete a request."""


class WorkerBlockedError(RuntimeError):
    """Raised when a compatibility worker is blocked on a runtime prerequisite."""


class BaseWorker:
    """Compatibility worker base used by existing worker implementations.

    This wrapper preserves the earlier worker-specific request/result protocol in
    `src.core.models.events` while the new agent layer evolves in parallel.
    """

    agent_role: AgentRole
    supported_task_types: frozenset[TaskType] = frozenset()
    capabilities: frozenset[str] = frozenset()

    def supports_task(self, task: BaseTaskNode) -> bool:
        """Return True when this worker can handle the provided TG task."""

        return task.task_type in self.supported_task_types

    def build_request(
        self,
        task: BaseTaskNode,
        operation_id: str,
        task_runtime: TaskRuntime | None = None,
        session_id: str | None = None,
        assigned_worker_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTaskRequest:
        """Build the legacy worker request envelope from TG and Runtime data."""

        if not self.supports_task(task):
            raise ValueError(
                f"{self.agent_role.value} does not support task type {task.task_type.value}"
            )
        runtime = task_runtime or TaskRuntime(
            task_id=task.id,
            tg_node_id=task.id,
            attempt_count=task.attempt_count,
            max_attempts=task.max_attempts,
            deadline=task.deadline,
            resource_keys=set(task.resource_keys),
        )
        context = AgentExecutionContext(
            operation_id=operation_id,
            task_id=task.id,
            tg_node_id=runtime.tg_node_id or task.id,
            task_type=task.task_type,
            attempt_count=runtime.attempt_count,
            max_attempts=runtime.max_attempts,
            assigned_worker_id=assigned_worker_id or runtime.assigned_worker,
            session_id=session_id or runtime.metadata.get("session_id"),
            checkpoint_ref=runtime.checkpoint_ref,
            deadline=runtime.deadline or task.deadline,
            resource_keys=set(task.resource_keys) | set(runtime.resource_keys),
            metadata=dict(runtime.metadata),
        )
        return AgentTaskRequest(
            agent_role=self.agent_role,
            intent=self.default_intent(task.task_type),
            context=context,
            task_label=task.label,
            input_bindings=dict(task.input_bindings),
            target_refs=list(task.target_refs),
            source_refs=list(task.source_refs),
            expected_output_refs=list(task.expected_output_refs),
            metadata=metadata or {},
        )

    def execute_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Execute one compatibility worker request and always return a result."""

        try:
            self._validate_request(request)
            result = self.handle_task(request)
        except WorkerBlockedError as exc:
            return self._result(
                request,
                status=AgentResultStatus.BLOCKED,
                summary=str(exc) or "worker is blocked on runtime prerequisites",
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive envelope
            return self._result(
                request,
                status=AgentResultStatus.FAILED,
                summary=f"{self.agent_role.value} failed while handling task",
                error_message=str(exc),
            )
        self._validate_result(request, result)
        return result

    @abstractmethod
    def handle_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Handle one compatibility worker request."""

    def default_intent(self, task_type: TaskType) -> AgentTaskIntent:
        """Return the default protocol intent for one task type."""

        return AgentTaskIntent.EXECUTE_TASK

    def runtime_view_for_task(
        self,
        state: RuntimeState,
        task_id: str,
    ) -> TaskRuntime | None:
        """Return the runtime entry for one task when it exists."""

        return state.execution.tasks.get(task_id)

    def _validate_request(self, request: AgentTaskRequest) -> None:
        if request.agent_role != self.agent_role:
            raise ValueError(
                f"request agent_role {request.agent_role.value} does not match {self.agent_role.value}"
            )
        if request.context.task_type not in self.supported_task_types:
            raise ValueError(
                f"{self.agent_role.value} does not support task type {request.context.task_type.value}"
            )

    def _validate_result(self, request: AgentTaskRequest, result: AgentTaskResult) -> None:
        if result.request_id != request.request_id:
            raise ValueError("result.request_id must match request.request_id")
        if result.agent_role != self.agent_role:
            raise ValueError("result.agent_role must match worker role")
        if result.operation_id != request.context.operation_id:
            raise ValueError("result.operation_id must match request context")
        if result.task_id != request.context.task_id or result.tg_node_id != request.context.tg_node_id:
            raise ValueError("result task identity must match request context")

    def _result(
        self,
        request: AgentTaskRequest,
        *,
        status: AgentResultStatus,
        summary: str,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> AgentTaskResult:
        """Build a consistent compatibility worker result envelope."""

        return AgentTaskResult(
            request_id=request.request_id,
            agent_role=self.agent_role,
            operation_id=request.context.operation_id,
            task_id=request.context.task_id,
            tg_node_id=request.context.tg_node_id,
            status=status,
            summary=summary,
            error_message=error_message,
            metadata=metadata or {},
            **extra,
        )


__all__ = [
    "BaseWorker",
    "BaseWorkerAgent",
    "WorkerBlockedError",
    "WorkerCapability",
    "WorkerExecutionError",
    "WorkerTaskSpec",
]
