"""Scheduler agent that turns TG + Runtime state into assignment decisions.

The Scheduler reads Task Graph and Runtime State, identifies truly schedulable
tasks, chooses workers, and emits TG/Runtime deltas plus runtime-compatible
events. It does not write KG and does not perform global planning.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import DecisionRecord, StateDeltaRecord, new_record_id
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.models.runtime import RuntimeState, TaskRuntimeStatus, WorkerRuntime, WorkerStatus
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskStatus
from src.core.runtime.budgets import RuntimeBudgetManager
from src.core.runtime.events import (
    LockAcquiredEvent,
    RuntimeEventType,
    TaskQueuedEvent,
    WorkerAssignedEvent,
)
from src.core.runtime.runtime_queries import RuntimeQueryService
from src.core.runtime.scheduler import RuntimeScheduler, SchedulerTickResult


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class SchedulingContext(BaseModel):
    """Scheduling-side context and policy hints for one scheduler tick."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_assignments: int = Field(default=5, ge=1, le=100)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    available_workers: list[dict[str, Any]] = Field(default_factory=list)
    session_summary: dict[str, Any] = Field(default_factory=dict)
    lock_summary: dict[str, Any] = Field(default_factory=dict)
    budget_summary: dict[str, Any] = Field(default_factory=dict)
    retry_backoff_seconds: int = Field(default=30, ge=0)
    timeout_grace_seconds: int = Field(default=60, ge=0)
    policy_flags: dict[str, Any] = Field(default_factory=dict)


class SchedulingDecisionRecord(DecisionRecord):
    """Structured scheduler decision for assignment, deferral, or retry."""

    task_id: str = Field(min_length=1)
    worker_id: str | None = None
    session_id: str | None = None
    action: str = Field(min_length=1)
    accepted: bool = False
    required_resource_keys: list[str] = Field(default_factory=list)
    retry_after_seconds: int | None = Field(default=None, ge=0)


class SchedulerAgent(BaseAgent):
    """Read TG + Runtime and emit scheduling decisions without executing work."""

    def __init__(self, name: str = "scheduler_agent") -> None:
        self._queries = RuntimeQueryService()
        self._budgets = RuntimeBudgetManager()
        self._runtime_scheduler = RuntimeScheduler()
        super().__init__(
            name=name,
            kind=AgentKind.SCHEDULER,
            write_permission=WritePermission(
                scopes=[GraphScope.TG, GraphScope.RUNTIME],
                allow_structural_write=False,
                allow_state_write=True,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Require TG refs plus a TaskGraph snapshot and runtime inputs."""

        super().validate_input(agent_input)
        if not any(ref.graph == GraphScope.TG for ref in agent_input.graph_refs):
            raise ValueError("scheduler input requires at least one TG ref")
        self._resolve_task_graph(agent_input)
        self._resolve_scheduling_context(agent_input)

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Find schedulable tasks, assign workers, and emit TG/Runtime deltas."""

        task_graph = self._resolve_task_graph(agent_input)
        runtime_state = self._resolve_runtime_state(agent_input)
        context = self._resolve_scheduling_context(agent_input)

        schedulable_tasks = self._find_schedulable_tasks(
            task_graph=task_graph,
            runtime_state=runtime_state,
            context=context,
        )
        decisions: list[SchedulingDecisionRecord] = []
        emitted_events: list[dict[str, Any]] = []
        state_deltas: list[dict[str, Any]] = []
        assigned_count = 0
        reserved_workers: set[str] = set()

        for task in schedulable_tasks:
            blockers = self._check_runtime_blockers(
                task=task,
                runtime_state=runtime_state,
                context=context,
                reserved_workers=reserved_workers,
            )
            if blockers:
                decisions.append(
                    self._emit_assignment_decision(
                        task=task,
                        action="defer",
                        accepted=False,
                        rationale="; ".join(blockers),
                    )
                )
                continue
            if assigned_count >= context.max_assignments:
                decisions.append(
                    self._emit_assignment_decision(
                        task=task,
                        action="defer",
                        accepted=False,
                        rationale="max assignments reached for this scheduling tick",
                    )
                )
                continue

            worker = self._select_worker_for_task(
                task=task,
                runtime_state=runtime_state,
                context=context,
                reserved_workers=reserved_workers,
            )
            if worker is None:
                decisions.append(
                    self._emit_assignment_decision(
                        task=task,
                        action="defer",
                        accepted=False,
                        rationale="no idle worker available",
                    )
                )
                continue

            session_id = self._select_session_for_task(task=task, runtime_state=runtime_state)
            reserved_workers.add(worker.worker_id)
            assigned_count += 1

            decision = self._emit_assignment_decision(
                task=task,
                action="assign",
                accepted=True,
                rationale="task is ready in TG and admissible under runtime constraints",
                worker_id=worker.worker_id,
                session_id=session_id,
            )
            decisions.append(decision)
            state_deltas.extend(
                self._build_assignment_state_deltas(
                    task=task,
                    worker=worker,
                    session_id=session_id,
                )
            )
            emitted_events.extend(
                self._build_assignment_events(
                    task=task,
                    worker=worker,
                    session_id=session_id,
                    operation_id=agent_input.context.operation_id,
                )
            )

        retryable_tasks = self._find_retryable_tasks(task_graph=task_graph)
        for task in retryable_tasks:
            retry_decision = self._build_retry_decision(task=task, context=context)
            decisions.append(retry_decision)
            state_deltas.append(self._build_retry_runtime_delta(task=task, context=context))

        logs = [
            f"found {len(schedulable_tasks)} ready TG task(s) for scheduling evaluation",
            f"assigned {assigned_count} task(s) during this tick",
            f"emitted {len(emitted_events)} runtime-compatible event(s)",
            f"generated {len(state_deltas)} TG/Runtime state delta(s)",
            "scheduler only updates TG/Runtime surfaces and never writes KG facts",
        ]
        if retryable_tasks:
            logs.append(f"generated retry suggestions for {len(retryable_tasks)} failed task(s)")

        return AgentOutput(
            decisions=[record.to_agent_output_fragment() for record in decisions],
            emitted_events=emitted_events,
            state_deltas=state_deltas,
            logs=logs,
        )

    def schedule_with_runtime_scheduler(
        self,
        *,
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
    ) -> SchedulerTickResult:
        """Compatibility wrapper around the existing `RuntimeScheduler.tick()`."""

        return self._runtime_scheduler.tick(task_graph=task_graph, runtime_state=runtime_state)

    def _find_schedulable_tasks(
        self,
        *,
        task_graph: TaskGraph,
        runtime_state: RuntimeState | None,
        context: SchedulingContext,
    ) -> list[BaseTaskNode]:
        """Return tasks that are TG-ready and worth scheduling evaluation."""

        ready_tasks = [
            task
            for task in task_graph.find_schedulable_tasks()
            if isinstance(task, BaseTaskNode) and task.status == TaskStatus.READY
        ]
        if runtime_state is None:
            return ready_tasks

        compatible_ids = set(
            self._runtime_scheduler.select_schedulable_tasks(task_graph=task_graph, runtime_state=runtime_state)
        )
        selected = [task for task in ready_tasks if task.id in compatible_ids]

        # Preserve visibility into tasks that are TG-ready but blocked by runtime checks.
        missing = [task for task in ready_tasks if task.id not in compatible_ids]
        return [*selected, *missing]

    def _check_runtime_blockers(
        self,
        *,
        task: BaseTaskNode,
        runtime_state: RuntimeState | None,
        context: SchedulingContext,
        reserved_workers: set[str],
    ) -> list[str]:
        """Return runtime blockers that prevent task dispatch."""

        blockers: list[str] = []
        if task.status != TaskStatus.READY:
            blockers.append("task is not ready")
        if task.deadline is not None and task.deadline <= utc_now():
            blockers.append("task deadline already expired")
        if task.approval_required and not bool(context.policy_flags.get("approval_granted", False)):
            blockers.append("approval gate not satisfied")

        if runtime_state is not None:
            if self._queries.is_task_blocked_by_runtime(
                runtime_state,
                task_id=task.id,
                required_resource_keys=task.resource_keys,
            ):
                blockers.append("runtime lock conflict")
            if self._budgets.would_exceed_budget(
                runtime_state,
                operations=1,
                noise=task.estimated_noise,
                risk=task.estimated_risk,
            ):
                blockers.append("runtime budget exhausted")
            if not self._select_worker_for_task(
                task=task,
                runtime_state=runtime_state,
                context=context,
                reserved_workers=reserved_workers,
            ):
                blockers.append("no idle worker available")
            return blockers

        active_locks = set(str(item) for item in context.lock_summary.get("active_lock_keys", []))
        if active_locks & set(task.resource_keys):
            blockers.append("runtime lock conflict")
        remaining = self._coerce_mapping(context.runtime_summary.get("remaining_budgets") or context.budget_summary)
        if self._summary_budget_exhausted(task=task, remaining=remaining):
            blockers.append("runtime budget exhausted")
        available_worker_ids = {
            str(worker.get("worker_id"))
            for worker in context.available_workers
            if str(worker.get("status", "idle")).lower() == WorkerStatus.IDLE.value
        }
        if not available_worker_ids - reserved_workers:
            blockers.append("no idle worker available")
        return blockers

    def _select_worker_for_task(
        self,
        *,
        task: BaseTaskNode,
        runtime_state: RuntimeState | None,
        context: SchedulingContext,
        reserved_workers: set[str],
    ) -> WorkerRuntime | None:
        """Select one worker for the task from runtime state or provided summaries."""

        available: list[WorkerRuntime] = []
        if runtime_state is not None:
            available = self._queries.find_idle_workers(runtime_state)
        else:
            available = [
                WorkerRuntime.model_validate(worker)
                for worker in context.available_workers
                if str(worker.get("status", "idle")).lower() == WorkerStatus.IDLE.value
            ]

        if task.assigned_agent:
            for worker in available:
                if worker.worker_id == task.assigned_agent and worker.worker_id not in reserved_workers:
                    return worker

        for worker in available:
            if worker.worker_id in reserved_workers:
                continue
            return worker
        return None

    def _emit_assignment_decision(
        self,
        *,
        task: BaseTaskNode,
        action: str,
        accepted: bool,
        rationale: str,
        worker_id: str | None = None,
        session_id: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> SchedulingDecisionRecord:
        """Emit one scheduling decision record."""

        target_refs = [GraphRef(graph=GraphScope.TG, ref_id=task.id, ref_type="Task")]
        if worker_id:
            target_refs.append(GraphRef(graph=GraphScope.RUNTIME, ref_id=worker_id, ref_type="WorkerRuntime"))
        if session_id:
            target_refs.append(GraphRef(graph=GraphScope.RUNTIME, ref_id=session_id, ref_type="SessionRuntime"))
        return SchedulingDecisionRecord(
            source_agent=self.name,
            summary=f"Scheduler {action} decision for task {task.id}",
            confidence=0.8 if accepted else 0.6,
            refs=target_refs,
            payload={
                "task_type": task.task_type.value,
                "resource_keys": sorted(task.resource_keys),
                "estimated_cost": task.estimated_cost,
                "estimated_risk": task.estimated_risk,
                "estimated_noise": task.estimated_noise,
            },
            decision_type="task_assignment",
            score=task.goal_relevance,
            target_refs=target_refs,
            rationale=rationale,
            task_id=task.id,
            worker_id=worker_id,
            session_id=session_id,
            action=action,
            accepted=accepted,
            required_resource_keys=sorted(task.resource_keys),
            retry_after_seconds=retry_after_seconds,
        )

    def _find_retryable_tasks(self, *, task_graph: TaskGraph) -> list[BaseTaskNode]:
        """Return failed TG tasks that still have retry budget."""

        return [
            task
            for task in task_graph.find_retryable_tasks()
            if isinstance(task, BaseTaskNode)
        ]

    def _build_retry_decision(
        self,
        *,
        task: BaseTaskNode,
        context: SchedulingContext,
    ) -> SchedulingDecisionRecord:
        """Emit a retry/backoff suggestion for one failed task."""

        backoff = max(context.retry_backoff_seconds, getattr(task.retry_policy, "backoff_seconds", 0))
        return self._emit_assignment_decision(
            task=task,
            action="suggest_retry",
            accepted=False,
            rationale="task failed but remains retryable under TG policy",
            retry_after_seconds=backoff,
        )

    def _build_retry_runtime_delta(
        self,
        *,
        task: BaseTaskNode,
        context: SchedulingContext,
    ) -> dict[str, Any]:
        """Build one runtime-side retry suggestion delta."""

        retry_at = utc_now() + timedelta(seconds=max(context.retry_backoff_seconds, getattr(task.retry_policy, "backoff_seconds", 0)))
        return StateDeltaRecord(
            source_agent=self.name,
            summary=f"Suggest retry/backoff for task {task.id}",
            graph_scope=GraphScope.RUNTIME,
            delta_type="retry_suggestion",
            target_ref=GraphRef(graph=GraphScope.RUNTIME, ref_id=task.id, ref_type="TaskRuntime"),
            patch={
                "task_id": task.id,
                "retry_after_seconds": max(context.retry_backoff_seconds, getattr(task.retry_policy, "backoff_seconds", 0)),
                "retry_not_before": retry_at.isoformat(),
            },
            payload={"patch_kind": "runtime_retry"},
        ).to_agent_output_fragment()

    def _build_assignment_state_deltas(
        self,
        *,
        task: BaseTaskNode,
        worker: WorkerRuntime,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        """Build TG and Runtime state deltas for one accepted assignment."""

        deltas = [
            StateDeltaRecord(
                source_agent=self.name,
                summary=f"Queue TG task {task.id}",
                graph_scope=GraphScope.TG,
                delta_type="task_status_update",
                target_ref=GraphRef(graph=GraphScope.TG, ref_id=task.id, ref_type="Task"),
                patch={
                    "status": TaskStatus.QUEUED.value,
                    "assigned_agent": worker.worker_id,
                    "reason": "scheduled for runtime dispatch",
                },
                payload={"patch_kind": "tg_task_state"},
            ).to_agent_output_fragment(),
            StateDeltaRecord(
                source_agent=self.name,
                summary=f"Queue runtime task {task.id}",
                graph_scope=GraphScope.RUNTIME,
                delta_type="task_runtime_update",
                target_ref=GraphRef(graph=GraphScope.RUNTIME, ref_id=task.id, ref_type="TaskRuntime"),
                patch={
                    "task_id": task.id,
                    "tg_node_id": task.id,
                    "status": TaskRuntimeStatus.QUEUED.value,
                    "assigned_worker": worker.worker_id,
                    "resource_keys": sorted(task.resource_keys),
                    "session_id": session_id,
                },
                payload={"patch_kind": "runtime_task_state"},
            ).to_agent_output_fragment(),
            StateDeltaRecord(
                source_agent=self.name,
                summary=f"Reserve worker {worker.worker_id} for task {task.id}",
                graph_scope=GraphScope.RUNTIME,
                delta_type="worker_runtime_update",
                target_ref=GraphRef(graph=GraphScope.RUNTIME, ref_id=worker.worker_id, ref_type="WorkerRuntime"),
                patch={
                    "worker_id": worker.worker_id,
                    "status": WorkerStatus.BUSY.value,
                    "current_task_id": task.id,
                    "current_load": max(worker.current_load, 0) + 1,
                },
                payload={"patch_kind": "runtime_worker_state"},
            ).to_agent_output_fragment(),
        ]
        for lock_key in sorted(task.resource_keys):
            deltas.append(
                StateDeltaRecord(
                    source_agent=self.name,
                    summary=f"Reserve lock {lock_key} for task {task.id}",
                    graph_scope=GraphScope.RUNTIME,
                    delta_type="lock_reservation",
                    target_ref=GraphRef(graph=GraphScope.RUNTIME, ref_id=lock_key, ref_type="ResourceLock"),
                    patch={
                        "lock_key": lock_key,
                        "owner_type": "task",
                        "owner_id": task.id,
                        "status": "active",
                    },
                    payload={"patch_kind": "runtime_lock_state"},
                ).to_agent_output_fragment()
            )
        return deltas

    def _build_assignment_events(
        self,
        *,
        task: BaseTaskNode,
        worker: WorkerRuntime,
        session_id: str | None,
        operation_id: str,
    ) -> list[dict[str, Any]]:
        """Build runtime-compatible events for one accepted assignment."""

        events: list[dict[str, Any]] = [
            TaskQueuedEvent(
                operation_id=operation_id,
                task_id=task.id,
                tg_node_id=task.id,
                queue_name="default",
                deadline=task.deadline,
                priority=task.priority,
            ).model_dump(mode="json"),
            WorkerAssignedEvent(
                operation_id=operation_id,
                worker_id=worker.worker_id,
                task_id=task.id,
                queue_name="default",
                current_load=max(worker.current_load, 0) + 1,
            ).model_dump(mode="json"),
        ]
        for lock_key in sorted(task.resource_keys):
            events.append(
                LockAcquiredEvent(
                    operation_id=operation_id,
                    lock_key=lock_key,
                    owner_type="task",
                    owner_id=task.id,
                ).model_dump(mode="json")
            )
        if session_id:
            events.append(
                {
                    "event_id": new_record_id("runtime-event"),
                    "event_type": RuntimeEventType.SESSION_HEARTBEAT.value,
                    "operation_id": operation_id,
                    "session_id": session_id,
                    "created_at": utc_now().isoformat(),
                    "payload": {"reason": "scheduler_selected_session"},
                }
            )
        return events

    def _resolve_task_graph(self, agent_input: AgentInput) -> TaskGraph:
        """Resolve a TG snapshot from raw payload."""

        raw_graph = agent_input.raw_payload.get("tg_graph") or agent_input.raw_payload.get("task_graph")
        if isinstance(raw_graph, TaskGraph):
            return raw_graph
        if isinstance(raw_graph, dict):
            return TaskGraph.from_dict(raw_graph)
        raise ValueError("scheduler input requires raw_payload.tg_graph or equivalent TaskGraph snapshot")

    def _resolve_runtime_state(self, agent_input: AgentInput) -> RuntimeState | None:
        """Resolve an optional concrete runtime state snapshot."""

        raw_state = agent_input.raw_payload.get("runtime_state")
        if isinstance(raw_state, RuntimeState):
            return raw_state
        if isinstance(raw_state, dict) and raw_state:
            return RuntimeState.model_validate(raw_state)
        return None

    def _resolve_scheduling_context(self, agent_input: AgentInput) -> SchedulingContext:
        """Normalize runtime and scheduler hints into one context object."""

        runtime_state = self._resolve_runtime_state(agent_input)
        summary_from_state = (
            self._queries.build_scheduler_view(runtime_state)
            if runtime_state is not None
            else {}
        )
        available_workers = (
            [worker.model_dump(mode="json") for worker in self._queries.find_idle_workers(runtime_state)]
            if runtime_state is not None
            else self._coerce_list_of_mappings(agent_input.raw_payload.get("available_workers"))
        )
        payload = {
            "runtime_summary": summary_from_state | self._coerce_mapping(agent_input.raw_payload.get("runtime_summary")),
            "available_workers": available_workers,
            "session_summary": self._coerce_mapping(agent_input.raw_payload.get("session_summary")),
            "lock_summary": self._coerce_mapping(agent_input.raw_payload.get("lock_summary")),
            "budget_summary": self._coerce_mapping(agent_input.raw_payload.get("budget_summary")),
            **self._coerce_mapping(agent_input.raw_payload.get("scheduling_context")),
        }
        return SchedulingContext.model_validate(payload)

    def _select_session_for_task(
        self,
        *,
        task: BaseTaskNode,
        runtime_state: RuntimeState | None,
    ) -> str | None:
        """Reuse the existing runtime scheduler session selection logic when possible."""

        if runtime_state is None:
            return None
        return self._runtime_scheduler._select_session_for_task(runtime_state, task)

    @staticmethod
    def _summary_budget_exhausted(
        *,
        task: BaseTaskNode,
        remaining: dict[str, Any],
    ) -> bool:
        """Apply a minimal budget check using summary fields only."""

        noise_remaining = remaining.get("noise_budget_remaining")
        risk_remaining = remaining.get("risk_budget_remaining")
        operation_remaining = remaining.get("operation_budget_remaining")
        if noise_remaining is not None and float(noise_remaining) < task.estimated_noise:
            return True
        if risk_remaining is not None and float(risk_remaining) < task.estimated_risk:
            return True
        if operation_remaining is not None and int(operation_remaining) < 1:
            return True
        return False

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        """Return a shallow mapping copy or an empty mapping."""

        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_list_of_mappings(value: Any) -> list[dict[str, Any]]:
        """Normalize input into a list of mappings."""

        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [dict(item) for item in items if isinstance(item, dict)]


__all__ = [
    "SchedulerAgent",
    "SchedulingContext",
    "SchedulingDecisionRecord",
]
