"""Top-level orchestration entry point."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.app.settings import AppSettings
from src.core.agents.agent_pipeline import AgentPipeline, PipelineCycleResult, PipelineStepResult
from src.core.agents.agent_protocol import AgentKind, GraphRef as AgentGraphRef, GraphScope
from src.core.agents.pipeline_builders import AgentPipelineAssemblyOptions, build_optional_agent_pipeline
from src.core.models.events import AgentTaskResult
from src.core.models.runtime import OperationRuntime, RuntimeState, RuntimeStatus, utc_now
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskStatus
from src.core.runtime.observability import append_operation_log, mark_clean_shutdown, mark_unclean_shutdown, record_phase_checkpoint
from src.core.runtime.llm_history import (
    LLMDecisionHistoryRecord,
    append_llm_decision_history,
    ensure_llm_decision_history,
    recent_llm_decision_history,
)
from src.core.runtime.result_applier import PhaseTwoApplyResult, PhaseTwoResultApplier
from src.core.runtime.store import FileRuntimeStore, InMemoryRuntimeStore, RuntimeStore


class TargetHost(BaseModel):
    """Small inventory record used by the first-stage control plane."""

    model_config = ConfigDict(extra="forbid")

    address: str = Field(min_length=1)
    hostname: str | None = None
    platform: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationSummary(BaseModel):
    """Compact operation summary returned by the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    operation_status: RuntimeStatus
    runtime_task_count: int = 0
    worker_count: int = 0
    target_count: int = 0
    last_cycle_phase: str | None = None
    unclean_shutdown: bool = False
    audit_event_count: int = 0
    pending_event_count: int = 0
    last_updated: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationCycleResult(BaseModel):
    """一次 operation 主循环的结构化结果。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    operation_id: str
    cycle_index: int = Field(ge=1)
    planning: PipelineCycleResult | None = None
    execution: PipelineCycleResult | None = None
    feedback: PipelineCycleResult | None = None
    apply_results: list[PhaseTwoApplyResult] = Field(default_factory=list)
    selected_task_ids: list[str] = Field(default_factory=list)
    applied_task_ids: list[str] = Field(default_factory=list)
    stopped: bool = False
    stop_reason: str | None = None
    runtime_state: RuntimeState


class AppOrchestrator:
    """Application-layer orchestration facade for phase-one control flows."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        runtime_store: RuntimeStore | None = None,
        pipeline: AgentPipeline | None = None,
        result_applier: PhaseTwoResultApplier | None = None,
    ) -> None:
        self.settings = settings or AppSettings.from_env()
        self.runtime_store = runtime_store or self._build_runtime_store(self.settings)
        self.pipeline = pipeline or self._build_default_pipeline(self.settings)
        self.result_applier = result_applier or PhaseTwoResultApplier()

    def create_operation(self, operation_id: str, metadata: dict[str, Any] | None = None) -> RuntimeState:
        """Create a new operation with control-plane metadata attached."""

        runtime_policy = self.settings.load_runtime_policy()
        operation_metadata = {
            "control_plane": {
                "audit_enabled": self.settings.audit_enabled,
                "audit_persist_enabled": self.settings.audit_persist_enabled,
                # 中文注释：
                # 日志治理参数跟随 operation 一起固化到 runtime metadata，
                # 这样后续 resume、export 和不同 store 后端都能复用同一份配置。
                "audit_max_entries": self.settings.audit_max_entries,
                "operation_log_max_entries": self.settings.operation_log_max_entries,
                "audit_redaction_enabled": self.settings.audit_redaction_enabled,
                "recovery_enabled": self.settings.recovery_enabled,
                "max_concurrent_workers": self.settings.max_concurrent_workers,
                "default_operation_budget": self.settings.default_operation_budget,
                "default_scan_timeout_sec": self.settings.default_scan_timeout_sec,
                "llm_advisors": self._llm_advisor_status(self.settings),
            },
            # 中文注释：
            # operation metadata 中只落稳定 JSON 结构，避免把 Pydantic 对象直接塞进 state。
            "runtime_policy": runtime_policy.to_runtime_metadata(),
            "target_inventory": [],
            "target_count": 0,
            "llm_decision_history": [],
        }
        if metadata:
            operation_metadata.update(metadata)
        state = RuntimeState(
            operation_id=operation_id,
            operation_status=RuntimeStatus.CREATED,
            execution=OperationRuntime(
                operation_id=operation_id,
                status=RuntimeStatus.CREATED,
                metadata=operation_metadata,
            ),
        )
        created = self.runtime_store.create_operation(operation_id, initial_state=state)
        ensure_llm_decision_history(created)
        self._log_operation_event(
            created,
            event_type="operation_created",
            runtime_policy=runtime_policy.to_runtime_metadata(),
        )
        self.runtime_store.save_state(created)
        return created.model_copy(deep=True)

    def import_targets(self, operation_id: str, targets: list[TargetHost]) -> RuntimeState:
        """Persist the current target inventory in operation metadata."""

        state = self.runtime_store.snapshot(operation_id)
        inventory = {target.address: target for target in targets}
        ordered_targets = [inventory[address] for address in sorted(inventory)]
        state.execution.metadata["target_inventory"] = [
            target.model_dump(mode="json")
            for target in ordered_targets
        ]
        state.execution.metadata["target_count"] = len(ordered_targets)
        state.execution.summary = f"{len(ordered_targets)} targets imported"
        state.last_updated = utc_now()
        self._log_operation_event(
            state,
            event_type="targets_imported",
            target_count=len(ordered_targets),
        )
        self.runtime_store.save_state(state)
        return state.model_copy(deep=True)

    def start_operation(self, operation_id: str) -> RuntimeState:
        """Mark an operation ready for the first planning/execution cycle."""

        started_at = utc_now()
        state = self.runtime_store.snapshot(operation_id)
        state.operation_status = RuntimeStatus.READY
        state.execution.status = RuntimeStatus.READY
        state.execution.started_at = state.execution.started_at or started_at
        state.execution.metadata["last_control_cycle"] = {
            "cycle_type": "bootstrap",
            "started_at": started_at.isoformat(),
            "status": RuntimeStatus.READY.value,
        }
        state.execution.summary = "bootstrap control cycle initialized"
        state.last_updated = started_at
        self._log_operation_event(state, event_type="operation_started", status=RuntimeStatus.READY.value)
        self.runtime_store.save_state(state)
        return state.model_copy(deep=True)

    def get_operation_state(self, operation_id: str) -> RuntimeState:
        """Return the current runtime state for one operation."""

        return self.runtime_store.snapshot(operation_id)

    def get_operation_summary(self, operation_id: str) -> OperationSummary:
        """Return a compact operation summary for API consumers."""

        state = self.runtime_store.snapshot(operation_id)
        recovery = self._mapping(state.execution.metadata.get("recovery"))
        last_phase_checkpoint = self._mapping(state.execution.metadata.get("last_phase_checkpoint"))
        return OperationSummary(
            operation_id=state.operation_id,
            operation_status=state.operation_status,
            runtime_task_count=len(state.execution.tasks),
            worker_count=len(state.workers),
            target_count=int(state.execution.metadata.get("target_count", 0)),
            last_cycle_phase=str(last_phase_checkpoint.get("phase")) if last_phase_checkpoint.get("phase") is not None else None,
            unclean_shutdown=bool(recovery.get("unclean_shutdown", False)),
            audit_event_count=len(state.execution.metadata.get("audit_log", [])),
            pending_event_count=len(state.pending_events),
            last_updated=state.last_updated.isoformat(),
            metadata=dict(state.execution.metadata),
        )

    def list_operations(self) -> list[OperationSummary]:
        """Return summaries for all known operations."""

        return [
            self.get_operation_summary(operation_id)
            for operation_id in self.runtime_store.list_operation_ids()
        ]

    def recover_operation(self, operation_id: str, *, reason: str = "manual_recover") -> RuntimeState:
        """Normalize runtime state without forcing the operation back to ready."""

        state = self.runtime_store.recover_operation(operation_id, reason=reason)
        # 中文注释：
        # recover 和 resume 分开：recover 只做保守整理，便于运维先收口脏状态，
        # 是否继续执行交给后续 resume 或人工确认。
        self._log_operation_event(
            state,
            event_type="operation_recovered",
            recovery_reason=reason,
            recovery_metadata=dict(state.execution.metadata.get("recovery", {})),
        )
        self.runtime_store.save_state(state)
        return state.model_copy(deep=True)

    def export_audit_report(self, operation_id: str) -> dict[str, Any]:
        """Export the persisted audit report for one operation."""

        return self.runtime_store.export_audit_report(operation_id)

    def get_llm_decision_history(self, operation_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent operation-level LLM decision history entries."""

        state = self.runtime_store.snapshot(operation_id)
        return recent_llm_decision_history(state, limit=limit)

    def record_llm_decision_cycle(
        self,
        operation_id: str,
        *,
        cycle_index: int,
        cycle: PipelineCycleResult,
    ) -> list[dict[str, Any]]:
        """Persist LLM decision history from an explicitly executed pipeline cycle."""

        state = self.runtime_store.snapshot(operation_id)
        records = self._extract_llm_decision_history(cycle_index=cycle_index, cycle=cycle)
        if records:
            append_llm_decision_history(state, records)
            self._log_operation_event(
                state,
                event_type="llm_decision_history_recorded",
                cycle_index=cycle_index,
                record_count=len(records),
            )
            self.runtime_store.save_state(state)
        return recent_llm_decision_history(state, limit=len(records) if records else 20)

    def get_health_status(self) -> dict[str, Any]:
        """Return a lightweight liveness view for operational checks."""

        return {
            "status": "ok",
            "runtime_store_backend": self.settings.runtime_store_backend,
            "operation_count": len(self.runtime_store.list_operation_ids()),
            "llm_advisors": self._llm_advisor_status(self.settings),
        }

    def get_readiness_status(self) -> dict[str, Any]:
        """Return a lightweight readiness view for operational checks."""

        return {
            "status": "ready",
            "runtime_store_backend": self.settings.runtime_store_backend,
            "recovery_enabled": self.settings.recovery_enabled,
            "llm_advisors": self._llm_advisor_status(self.settings),
        }

    def run_operation_cycle(
        self,
        operation_id: str,
        *,
        graph_refs: list[AgentGraphRef],
        planner_payload: dict[str, Any],
        scheduler_payload: dict[str, Any] | None = None,
        feedback_payload: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> OperationCycleResult:
        """运行一次统一主循环。

        中文注释：
        这里把 plan -> schedule -> execute -> apply -> feedback 串起来，
        AppOrchestrator 不再只是 RuntimeState 门面，而是 operation 级驱动入口。
        """

        pipeline = self._require_pipeline()
        state = self.runtime_store.snapshot(operation_id)
        if self.settings.recovery_enabled and self._needs_recovery(state):
            # 中文注释：
            # 自动恢复统一走 store 层，确保 memory/file backend 都会同步刷新恢复快照。
            state = self.runtime_store.recover_operation(operation_id, reason="unclean_shutdown")
            resume_summary = dict(state.execution.metadata.get("recovery", {}))
            self._log_operation_event(state, event_type="operation_resumed", **resume_summary)
        cycle_index = self._next_cycle_index(state)
        state.operation_status = RuntimeStatus.RUNNING
        state.execution.status = RuntimeStatus.RUNNING
        mark_unclean_shutdown(state, cycle_index=cycle_index)
        self._log_operation_event(state, event_type="cycle_started", cycle_index=cycle_index)
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="cycle_started",
            status="running",
        )

        planning = self._run_planning_phase(
            pipeline=pipeline,
            operation_id=operation_id,
            graph_refs=graph_refs,
            planner_payload=planner_payload,
            context=context,
        )
        self._log_operation_event(
            state,
            event_type="planning_completed",
            cycle_index=cycle_index,
            success=planning.success,
            step_count=len(planning.steps),
        )
        self._append_llm_decision_history_from_cycle(
            state,
            cycle_index=cycle_index,
            cycle=planning,
        )
        task_graph = self._task_graph_from_planning(planning=planning, state=state)
        state.execution.metadata["task_graph"] = task_graph.to_dict()
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="planning_completed",
            status="completed" if planning.success else "failed",
            step_count=len(planning.steps),
            success=planning.success,
        )

        execution = self._run_execution_phase(
            pipeline=pipeline,
            operation_id=operation_id,
            graph_refs=graph_refs,
            task_graph=task_graph,
            runtime_state=state,
            scheduler_payload=scheduler_payload,
            context=context,
        )
        selected_task_ids = self._selected_task_ids(execution)
        self._log_operation_event(
            state,
            event_type="execution_completed",
            cycle_index=cycle_index,
            success=execution.success,
            selected_task_ids=selected_task_ids,
        )
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="execution_completed",
            status="completed" if execution.success else "failed",
            selected_task_ids=selected_task_ids,
            step_count=len(execution.steps),
            success=execution.success,
        )

        apply_results: list[PhaseTwoApplyResult] = []
        applied_task_ids: list[str] = []
        recent_outcomes: list[dict[str, Any]] = []
        # 中文注释：
        # worker step 的 `AgentOutput` 会先通过 pipeline 内部的统一适配层收敛成
        # `AgentTaskResult`，这样 orchestrator 不再维护第二份转换逻辑。
        for task_result in pipeline.worker_task_results(execution):
            applied_task_ids.append(task_result.task_id)
            applied = self.result_applier.apply(
                task_result,
                state,
                kg_ref=self._kg_ref(graph_refs),
                goal_context=self._mapping((feedback_payload or {}).get("goal_context")),
                policy_context=self._mapping((feedback_payload or {}).get("policy_context")),
            )
            apply_results.append(applied)
            recent_outcomes.append(self._recent_outcome_entry(task_result))
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="apply_completed",
            status="completed",
            selected_task_ids=selected_task_ids,
            applied_task_ids=applied_task_ids,
            runtime_event_count=sum(len(item.runtime_event_refs) for item in apply_results),
        )

        feedback = self._run_feedback_phase(
            pipeline=pipeline,
            operation_id=operation_id,
            graph_refs=graph_refs,
            task_graph=task_graph,
            runtime_state=state,
            feedback_payload=feedback_payload,
            recent_outcomes=recent_outcomes,
            context=context,
        )
        self._mark_applied_tasks(task_graph, applied_task_ids)
        self._log_operation_event(
            state,
            event_type="feedback_completed",
            cycle_index=cycle_index,
            success=feedback.success,
            applied_task_ids=applied_task_ids,
        )
        self._append_llm_decision_history_from_cycle(
            state,
            cycle_index=cycle_index,
            cycle=feedback,
        )
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="feedback_completed",
            status="completed" if feedback.success else "failed",
            selected_task_ids=selected_task_ids,
            applied_task_ids=applied_task_ids,
            runtime_event_count=sum(len(item.runtime_event_refs) for item in apply_results),
            step_count=len(feedback.steps),
            success=feedback.success,
        )

        summary = self._persist_cycle_summary(
            state,
            cycle_index=cycle_index,
            planning=planning,
            execution=execution,
            feedback=feedback,
            apply_results=apply_results,
            applied_task_ids=applied_task_ids,
            task_graph=task_graph,
        )
        mark_clean_shutdown(state, cycle_index=cycle_index)
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="cycle_completed",
            status="completed",
            selected_task_ids=selected_task_ids,
            applied_task_ids=applied_task_ids,
            runtime_event_count=sum(len(item.runtime_event_refs) for item in apply_results),
            stopped=summary["stopped"],
            stop_reason=summary["stop_reason"],
            persist=False,
        )
        self.runtime_store.save_state(state)
        return OperationCycleResult(
            operation_id=operation_id,
            cycle_index=cycle_index,
            planning=planning,
            execution=execution,
            feedback=feedback,
            apply_results=apply_results,
            selected_task_ids=selected_task_ids,
            applied_task_ids=list(applied_task_ids),
            stopped=summary["stopped"],
            stop_reason=summary["stop_reason"],
            runtime_state=state.model_copy(deep=True),
        )

    def run_until_quiescent(
        self,
        operation_id: str,
        *,
        graph_refs: list[AgentGraphRef],
        planner_payload: dict[str, Any],
        scheduler_payload: dict[str, Any] | None = None,
        feedback_payload: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_cycles: int = 5,
    ) -> list[OperationCycleResult]:
        """持续运行主循环，直到静止或达到上限。"""

        results: list[OperationCycleResult] = []
        for _ in range(max_cycles):
            cycle_result = self.run_operation_cycle(
                operation_id,
                graph_refs=graph_refs,
                planner_payload=planner_payload,
                scheduler_payload=scheduler_payload,
                feedback_payload=feedback_payload,
                context=context,
            )
            results.append(cycle_result)
            if cycle_result.stopped:
                break
        return results

    def resume_operation(self, operation_id: str, *, reason: str = "manual_resume") -> RuntimeState:
        """Normalize in-flight runtime state so the operation can resume safely."""

        state = self.runtime_store.recover_operation(operation_id, reason=reason)
        summary = dict(state.execution.metadata.get("recovery", {}))
        state.operation_status = RuntimeStatus.READY
        state.execution.status = RuntimeStatus.READY
        self._log_operation_event(state, event_type="operation_resumed", **summary)
        self.runtime_store.save_state(state)
        return state.model_copy(deep=True)

    def _run_planning_phase(
        self,
        *,
        pipeline: AgentPipeline,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        planner_payload: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> PipelineCycleResult:
        """运行 planning 阶段。"""

        return pipeline.run_planning_cycle(
            operation_id=operation_id,
            graph_refs=graph_refs,
            planner_payload=planner_payload,
            context=context,
        )

    def _run_execution_phase(
        self,
        *,
        pipeline: AgentPipeline,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
        scheduler_payload: dict[str, Any] | None,
        context: dict[str, Any] | None,
    ) -> PipelineCycleResult:
        """运行 scheduling + worker execution 阶段。"""

        payload = {
            **self._mapping(scheduler_payload),
            "tg_graph": task_graph.to_dict(),
            "runtime_state": runtime_state.model_dump(mode="json"),
        }
        # 中文注释：
        # 调度器里的 worker_id 更接近 runtime worker 槽位，不一定等于 agent registry 名称。
        # 当仓库里只注册了一个 worker agent 时，这里显式指定它，避免 execution 阶段因为名称不一致而中断。
        return pipeline.run_execution_cycle(
            operation_id=operation_id,
            graph_refs=graph_refs,
            scheduler_payload=payload,
            worker_agent=self._default_worker_agent_name(pipeline, scheduler_payload),
            context=context,
        )

    def _run_feedback_phase(
        self,
        *,
        pipeline: AgentPipeline,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
        feedback_payload: dict[str, Any] | None,
        recent_outcomes: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> PipelineCycleResult:
        """运行 feedback 阶段。

        中文注释：
        这里不重复消费 worker step 做 perception/state-writer，而是把 apply 之后的
        runtime/tg 摘要交给 critic，避免重复写入。
        """

        payload = {
            **self._mapping(feedback_payload),
            "tg_graph": task_graph.to_dict(),
            "runtime_state": runtime_state.model_dump(mode="json"),
            "runtime_summary": self._runtime_summary(runtime_state),
            "recent_outcomes": list(recent_outcomes),
        }
        return pipeline.run_feedback_cycle(
            operation_id=operation_id,
            graph_refs=graph_refs,
            worker_steps=[],
            feedback_payload=payload,
            context=context,
        )

    def _persist_cycle_summary(
        self,
        state: RuntimeState,
        *,
        cycle_index: int,
        planning: PipelineCycleResult,
        execution: PipelineCycleResult,
        feedback: PipelineCycleResult,
        apply_results: list[PhaseTwoApplyResult],
        applied_task_ids: list[str],
        task_graph: TaskGraph,
    ) -> dict[str, Any]:
        """把本轮主循环摘要回写到 RuntimeState。"""

        stopped = not self._selected_task_ids(execution) and not state.replan_requests
        stop_reason = "no schedulable work and no replan request" if stopped else None
        applied_summaries = [
            {
                "task_id": task_id,
                "runtime_event_count": len(item.runtime_event_refs),
                "kg_delta_count": len(item.kg_state_deltas),
                "ag_delta_count": len(item.ag_state_deltas),
            }
            for task_id, item in zip(applied_task_ids, apply_results, strict=False)
        ]
        summary = {
            "cycle_index": cycle_index,
            "started_at": utc_now().isoformat(),
            "planning_success": planning.success,
            "execution_success": execution.success,
            "feedback_success": feedback.success,
            "selected_task_ids": self._selected_task_ids(execution),
            "applied_results": applied_summaries,
            "replan_request_count": len(state.replan_requests),
            "llm_advisors": self._llm_advisor_status(self.settings),
            "llm_decision_history_count": len(state.execution.metadata.get("llm_decision_history", [])),
            "stopped": stopped,
            "stop_reason": stop_reason,
        }
        history = state.execution.metadata.setdefault("control_cycle_history", [])
        history.append(summary)
        state.execution.metadata["last_control_cycle"] = summary
        state.execution.metadata["task_graph"] = task_graph.to_dict()
        state.execution.summary = f"control cycle {cycle_index} completed"
        state.operation_status = RuntimeStatus.COMPLETED if stopped else RuntimeStatus.READY
        state.execution.status = state.operation_status
        state.last_updated = utc_now()
        self._log_operation_event(
            state,
            event_type="cycle_completed",
            cycle_index=cycle_index,
            stopped=stopped,
            stop_reason=stop_reason,
        )
        return summary

    def _append_llm_decision_history_from_cycle(
        self,
        state: RuntimeState,
        *,
        cycle_index: int,
        cycle: PipelineCycleResult,
    ) -> None:
        records = self._extract_llm_decision_history(cycle_index=cycle_index, cycle=cycle)
        if not records:
            return
        append_llm_decision_history(state, records)
        self._log_operation_event(
            state,
            event_type="llm_decision_history_recorded",
            cycle_index=cycle_index,
            cycle_name=cycle.cycle_name,
            record_count=len(records),
        )

    def _extract_llm_decision_history(
        self,
        *,
        cycle_index: int,
        cycle: PipelineCycleResult,
    ) -> list[LLMDecisionHistoryRecord]:
        records: list[LLMDecisionHistoryRecord] = []
        seen: set[tuple[str, str, bool, str | None]] = set()
        for step in cycle.steps:
            if step.agent_kind not in {AgentKind.PLANNER, AgentKind.CRITIC, AgentKind.SUPERVISOR}:
                continue
            for container in (
                step.agent_output.decisions,
                step.agent_output.replan_requests,
            ):
                for item in container:
                    for payload in self._iter_llm_payloads(item):
                        record = self._history_record_from_payload(
                            cycle_index=cycle_index,
                            agent_kind=step.agent_kind,
                            payload=payload,
                        )
                        if record is None:
                            continue
                        key = (
                            record.agent_kind,
                            record.decision_type,
                            record.accepted,
                            record.rejected_reason,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        records.append(record)
            for record in self._history_records_from_logs(
                cycle_index=cycle_index,
                agent_kind=step.agent_kind,
                logs=step.agent_output.logs,
            ):
                key = (
                    record.agent_kind,
                    record.decision_type,
                    record.accepted,
                    record.rejected_reason,
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
        return records

    def _history_record_from_payload(
        self,
        *,
        cycle_index: int,
        agent_kind: AgentKind,
        payload: dict[str, Any],
    ) -> LLMDecisionHistoryRecord | None:
        validation = self._mapping(payload.get("llm_decision_validation"))
        if not validation and isinstance(payload.get("validation"), dict):
            validation = self._mapping(payload.get("validation"))
        if not validation:
            return None
        accepted = bool(validation.get("accepted"))
        reason = str(validation.get("reason") or "") or None
        decision = self._mapping(payload.get("llm_decision")) or self._mapping(payload.get("decision"))
        decision_type = str(
            decision.get("decision_type")
            or payload.get("decision_type")
            or self._default_llm_decision_type(agent_kind)
        )
        return LLMDecisionHistoryRecord(
            cycle_index=cycle_index,
            agent_kind=agent_kind.value,
            advisor_type=self._advisor_type(agent_kind, observed=True),
            enabled=self._llm_advisor_enabled(agent_kind, observed=True),
            configured=self.settings.to_packy_llm_config() is not None,
            decision_type=decision_type,
            accepted=accepted,
            rejected_reason=None if accepted else reason,
            model=self._llm_model(),
        )

    def _history_records_from_logs(
        self,
        *,
        cycle_index: int,
        agent_kind: AgentKind,
        logs: list[str],
    ) -> list[LLMDecisionHistoryRecord]:
        records: list[LLMDecisionHistoryRecord] = []
        marker = "llm"
        rejected_marker = "rejected:"
        for log in logs:
            lowered = log.lower()
            if marker not in lowered or rejected_marker not in lowered:
                continue
            reason = log.split(rejected_marker, 1)[1].strip()
            records.append(
                LLMDecisionHistoryRecord(
                    cycle_index=cycle_index,
                    agent_kind=agent_kind.value,
                    advisor_type=self._advisor_type(agent_kind, observed=True),
                    enabled=self._llm_advisor_enabled(agent_kind, observed=True),
                    configured=self.settings.to_packy_llm_config() is not None,
                    decision_type=self._default_llm_decision_type(agent_kind),
                    accepted=False,
                    rejected_reason=reason or "llm decision rejected",
                    model=self._llm_model(),
                )
            )
        return records

    def _iter_llm_payloads(self, value: Any) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if "llm_decision_validation" in value or (
                "validation" in value and ("llm_decision" in value or "decision" in value)
            ):
                payloads.append(value)
            for item in value.values():
                payloads.extend(self._iter_llm_payloads(item))
        elif isinstance(value, list):
            for item in value:
                payloads.extend(self._iter_llm_payloads(item))
        return payloads

    def _llm_advisor_enabled(self, agent_kind: AgentKind, *, observed: bool = False) -> bool:
        if agent_kind == AgentKind.PLANNER:
            return self.settings.enable_planner_llm_advisor or observed
        if agent_kind == AgentKind.CRITIC:
            return self.settings.enable_critic_llm_advisor or observed
        if agent_kind == AgentKind.SUPERVISOR:
            return self.settings.enable_supervisor_llm_advisor or observed
        return observed

    def _advisor_type(self, agent_kind: AgentKind, *, observed: bool = False) -> str:
        if not self._llm_advisor_enabled(agent_kind, observed=observed):
            return "none"
        return "packy" if self.settings.to_packy_llm_config() is not None else "injected"

    @staticmethod
    def _default_llm_decision_type(agent_kind: AgentKind) -> str:
        if agent_kind == AgentKind.PLANNER:
            return "planner_strategy_decision"
        if agent_kind == AgentKind.CRITIC:
            return "critic_finding_review"
        if agent_kind == AgentKind.SUPERVISOR:
            return "supervisor_strategy"
        return "llm_decision"

    def _llm_model(self) -> str | None:
        config = self.settings.to_packy_llm_config()
        return config.model if config is not None else None

    def _task_graph_from_planning(self, *, planning: PipelineCycleResult, state: RuntimeState) -> TaskGraph:
        payload = state.execution.metadata.get("task_graph")
        current = TaskGraph.from_dict(payload) if isinstance(payload, dict) and payload else TaskGraph()
        patch_payload = {"nodes": [], "edges": []}
        for delta in planning.final_output.state_deltas:
            patch = dict(delta.get("patch", {}))
            if "node" in patch:
                patch_payload["nodes"].append(dict(patch["node"]))
            if "edge" in patch:
                patch_payload["edges"].append(dict(patch["edge"]))
        if not patch_payload["nodes"] and not patch_payload["edges"]:
            return current
        return TaskGraph.from_dict(patch_payload)

    @staticmethod
    def _mark_applied_tasks(task_graph: TaskGraph, task_ids: list[str]) -> None:
        """把已经执行并完成 apply 的 task 标记为 succeeded，避免下一轮重复调度。"""

        for task_id in task_ids:
            node = task_graph.get_node(task_id)
            if isinstance(node, BaseTaskNode):
                node.status = TaskStatus.SUCCEEDED

    def _selected_task_ids(self, execution: PipelineCycleResult) -> list[str]:
        for step in execution.steps:
            if step.agent_kind.value != "scheduler":
                continue
            return [
                str(decision.get("task_id"))
                for decision in step.agent_output.decisions
                if bool(decision.get("accepted"))
            ]
        return []

    @staticmethod
    def _worker_steps(execution: PipelineCycleResult) -> list[PipelineStepResult]:
        return [step for step in execution.steps if step.agent_kind.value == "worker"]

    @staticmethod
    def _recent_outcome_entry(task_result: AgentTaskResult) -> dict[str, Any]:
        outcome = dict(task_result.outcome_payload)
        payload_ref = str(
            outcome.get("raw_result_ref")
            or (task_result.evidence[0].payload_ref if task_result.evidence else f"runtime://worker-results/{task_result.task_id}")
        )
        # 中文注释：
        # Critic 读取的是 Runtime outcome cache 风格的摘要，而不是 worker 原始 outcome 全量字段。
        # 这里把执行结果压缩成稳定的最小结构，避免 feedback 阶段直接吃到底层执行器细节。
        return {
            "outcome_id": str(outcome.get("id") or f"outcome::{task_result.task_id}"),
            "task_id": task_result.task_id,
            "outcome_type": str(outcome.get("outcome_type") or outcome.get("kind") or "worker_outcome"),
            "summary": str(outcome.get("summary") or task_result.summary),
            "payload_ref": payload_ref,
            "metadata": {
                "status": task_result.status.value,
                "confidence": outcome.get("confidence"),
                "source_agent": outcome.get("source_agent"),
            },
        }

    @staticmethod
    def _default_worker_agent_name(
        pipeline: AgentPipeline,
        scheduler_payload: dict[str, Any] | None,
    ) -> str | None:
        payload = dict(scheduler_payload or {})
        explicit = payload.get("worker_agent")
        if isinstance(explicit, str) and explicit:
            return explicit
        workers = pipeline.registry.list_by_kind(AgentKind.WORKER)
        if len(workers) == 1:
            return workers[0].name
        return None

    @staticmethod
    def _runtime_summary(state: RuntimeState) -> dict[str, Any]:
        return {
            "operation_status": state.operation_status.value,
            "task_count": len(state.execution.tasks),
            "pending_event_count": len(state.pending_events),
            "replan_request_count": len(state.replan_requests),
        }

    @staticmethod
    def _kg_ref(graph_refs: list[AgentGraphRef]) -> AgentGraphRef | None:
        for ref in graph_refs:
            if ref.graph == GraphScope.KG:
                return ref
        return None

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _next_cycle_index(state: RuntimeState) -> int:
        history = state.execution.metadata.get("control_cycle_history", [])
        return len(history) + 1

    @staticmethod
    def _needs_recovery(state: RuntimeState) -> bool:
        recovery = state.execution.metadata.get("recovery", {})
        return isinstance(recovery, dict) and bool(recovery.get("unclean_shutdown"))

    @staticmethod
    def _log_operation_event(state: RuntimeState, *, event_type: str, **payload: Any) -> None:
        append_operation_log(
            state,
            event_type=event_type,
            operation_status=state.operation_status.value,
            **payload,
        )

    def _checkpoint_phase(
        self,
        state: RuntimeState,
        *,
        cycle_index: int,
        phase: str,
        status: str,
        selected_task_ids: list[str] | None = None,
        applied_task_ids: list[str] | None = None,
        runtime_event_count: int | None = None,
        step_count: int | None = None,
        success: bool | None = None,
        stopped: bool | None = None,
        stop_reason: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """记录阶段性 checkpoint，并在需要时立刻轻量落盘。"""

        checkpoint = record_phase_checkpoint(
            state,
            cycle_index=cycle_index,
            phase=phase,
            status=status,
            selected_task_ids=selected_task_ids,
            applied_task_ids=applied_task_ids,
            runtime_event_count=runtime_event_count,
            step_count=step_count,
            success=success,
            stopped=stopped,
            stop_reason=stop_reason,
        )
        # 中文注释：
        # 这里的 save 不是事务提交，只是把当前最小可信边界尽快落盘，
        # 尤其 apply 完成后可以避免结果已写入 state 但 crash 后完全丢失。
        if persist:
            self.runtime_store.save_state(state)
        return checkpoint

    def _require_pipeline(self) -> AgentPipeline:
        if self.pipeline is None:
            raise ValueError("AppOrchestrator.run_operation_cycle requires an AgentPipeline instance")
        return self.pipeline

    @staticmethod
    def _build_default_pipeline(settings: AppSettings) -> AgentPipeline:
        # 中文注释：
        # 运行时默认总是装配一套标准 pipeline；
        # planner advisor 是否启用、以及使用哪个 LLM 配置，统一由 settings 决定。
        llm_client_config = settings.to_packy_llm_config()
        enabled_advisors: list[str] = []
        if settings.enable_planner_llm_advisor:
            enabled_advisors.append("enable_planner_llm_advisor")
        if settings.enable_critic_llm_advisor:
            enabled_advisors.append("enable_critic_llm_advisor")
        if settings.enable_supervisor_llm_advisor:
            enabled_advisors.append("enable_supervisor_llm_advisor")
        if enabled_advisors and llm_client_config is None:
            raise ValueError(f"{', '.join(enabled_advisors)} require llm_api_key in AppSettings")
        return build_optional_agent_pipeline(
            options=AgentPipelineAssemblyOptions(
                enable_packy_planner_advisor=settings.enable_planner_llm_advisor,
                enable_packy_critic_advisor=settings.enable_critic_llm_advisor,
                enable_packy_supervisor_advisor=settings.enable_supervisor_llm_advisor,
            ),
            llm_client_config=llm_client_config,
        )

    @staticmethod
    def _llm_advisor_status(settings: AppSettings) -> dict[str, Any]:
        config = settings.to_packy_llm_config()
        return {
            "planner_enabled": settings.enable_planner_llm_advisor,
            "critic_enabled": settings.enable_critic_llm_advisor,
            "supervisor_enabled": settings.enable_supervisor_llm_advisor,
            "configured": config is not None,
            "model": config.model if config is not None else None,
            "base_url": config.base_url if config is not None else None,
        }

    @staticmethod
    def _build_runtime_store(settings: AppSettings) -> RuntimeStore:
        if settings.runtime_store_backend == "memory":
            return InMemoryRuntimeStore()
        return FileRuntimeStore(settings.runtime_store_dir)


__all__ = ["AppOrchestrator", "OperationCycleResult", "OperationSummary", "TargetHost"]
