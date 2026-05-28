"""Top-level orchestration entry point."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.app.settings import AppSettings
from src.app.llm_decision_observer import LLMDecisionObserver
from src.core.agents.agent_pipeline import AgentPipeline, PipelineCycleResult, PipelineStepResult
from src.core.agents.agent_protocol import AgentContext, AgentInput, AgentKind, AgentOutput, GraphRef as AgentGraphRef, GraphScope
from src.core.agents.graph_context import GraphContextBuilder
from src.core.agents.packy_llm import PackyLLMClient
from src.core.agents.pipeline_builders import AgentPipelineAssemblyOptions, build_optional_agent_pipeline
from src.core.execution.configured_mcp_client import ConfiguredMCPClient
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.graph_initializer import GraphInitializer
from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_merge import merge_task_graphs
from src.core.models.ag import AttackGraph
from src.core.models.events import AgentTaskResult
from src.core.models.runtime import OperationRuntime, ReplanRequest, RuntimeState, RuntimeStatus, utc_now
from src.core.models.scope import Asset, Engagement
from src.core.models.tg import BaseTaskNode, DependencyType, TaskGraph, TaskStatus
from src.core.planning.llm_mission_planner_advisor import LLMMissionPlannerAdvisor
from src.core.planning.mission_planner_agent import MissionPlannerAgent
from src.core.planning.stage_task_builder import StageTaskGraphBuilder
from src.core.runtime.audit_report import build_operation_audit_report
from src.core.runtime.observability import append_operation_log, mark_clean_shutdown, mark_unclean_shutdown, record_phase_checkpoint
from src.core.runtime.report_generator import ReportFormat, ReportGenerator
from src.core.runtime.llm_history import (
    LLMDecisionHistoryRecord,
    append_llm_decision_history,
    ensure_llm_decision_history,
    recent_llm_decision_history,
)
from src.core.runtime.result_applier import PhaseTwoApplyResult, PhaseTwoResultApplier
from src.core.scheduling.scheduler import schedule_ready_tasks
from src.core.scheduling.stage_scheduler import schedule_ready_stage_tasks
from src.core.stage.adapters import StageResultAdapter
from src.core.stage.llm_stage_advisor import LLMStageAdvisor
from src.core.stage.registry import StageAgentRegistry
from src.core.runtime.store import FileRuntimeStore, InMemoryRuntimeStore, RuntimeStore
from src.core.visualization.graph_publisher import graph_delta_publisher
from src.core.workers.llm_worker import LLMWorkerAgent
from src.core.workers.llm_worker_advisor import LLMWorkerAdvisor


class TargetHost(BaseModel):
    """Inventory record for host/domain/cidr/url/service target import."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    kind: str = "host"
    value: str | None = None
    hostname: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    protocol: str | None = None
    url: str | None = None
    platform: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_asset(self) -> Asset:
        value = self.value or self.address or self.url or self.hostname
        if not value:
            raise ValueError("target requires one of value, address, url or hostname")
        return Asset(
            asset_id=str(self.metadata.get("asset_id")) if self.metadata.get("asset_id") else None,
            kind=self.kind,  # type: ignore[arg-type]
            value=value,
            address=self.address,
            hostname=self.hostname,
            port=self.port,
            protocol=self.protocol,
            url=self.url,
            platform=self.platform,
            tags=list(self.tags),
            metadata=dict(self.metadata),
        )


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
        graph_memory_store: GraphMemoryStore | None = None,
        pipeline: AgentPipeline | None = None,
        result_applier: PhaseTwoResultApplier | None = None,
        graph_projector: AttackGraphProjector | None = None,
    ) -> None:
        self.settings = settings or AppSettings.from_env()
        self.runtime_store = runtime_store or self._build_runtime_store(self.settings)
        self.graph_memory_store = graph_memory_store or GraphMemoryStore(self.settings.runtime_store_dir)
        self.pipeline = pipeline or self._build_default_pipeline(self.settings)
        self.result_applier = result_applier or PhaseTwoResultApplier(attack_graph_projector=graph_projector)
        self.mcp_client = (
            ConfiguredMCPClient.from_sources(
                config_path=self.settings.mcp_config_path,
                config_json=self.settings.mcp_config_json,
            )
            if self.settings.mcp_enabled
            else None
        )
        llm_client_config = self.settings.to_packy_llm_config()
        stage_llm_client = (
            PackyLLMClient(llm_client_config)
            if llm_client_config is not None and self._planner_llm_enabled(self.settings)
            else None
        )
        stage_advisor = LLMStageAdvisor(client=stage_llm_client) if stage_llm_client is not None else None
        self.stage_registry = StageAgentRegistry.default(
            advisor=stage_advisor,
            mcp_client=self.mcp_client,
            default_timeout_seconds=self.settings.mcp_default_timeout_seconds,
        )
        mission_advisor = (
            LLMMissionPlannerAdvisor(client=stage_llm_client)
            if stage_llm_client is not None
            else None
        )
        self.mission_planner = MissionPlannerAgent(advisor=mission_advisor)
        self.stage_task_builder = StageTaskGraphBuilder()

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
                "agent_architecture": self._agent_architecture_metadata(self.settings),
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
        assets = [target.to_asset() for target in targets]
        inventory = {asset.normalized_value: asset for asset in assets}
        ordered_targets = [inventory[address] for address in sorted(inventory)]
        state.execution.metadata["target_inventory"] = [
            target.model_dump(mode="json")
            for target in ordered_targets
        ]
        policy_payload = dict(state.execution.metadata.get("runtime_policy", {}))
        policy_payload["engagement"] = Engagement(
            engagement_id=f"engagement::{operation_id}",
            assets=ordered_targets,
            scope_rules=[
                {
                    "rule_id": f"allow::{index}",
                    "action": "allow",
                    "kind": asset.kind,
                    "value": asset.normalized_value,
                    "reason": "imported target",
                }
                for index, asset in enumerate(ordered_targets)
            ],
        ).model_dump(mode="json")
        state.execution.metadata["runtime_policy"] = policy_payload
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
        graph_initialization = self._ensure_graph_initialized_from_targets(state)
        if graph_initialization is not None:
            state.execution.metadata["graph_initialization"] = graph_initialization
            kg, ag, tg, graph_runtime = self._load_graph_memory(operation_id)
            state.execution.metadata["graph_memory"] = self._graph_memory_metadata(
                kg=kg,
                ag=ag,
                tg=tg,
                loaded_runtime=graph_runtime is not None,
            )
            self._log_operation_event(
                state,
                event_type="graph_memory_initialized",
                target=graph_initialization["target"],
                initial_task_count=len(graph_initialization["initial_task_ids"]),
            )
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

    def get_operation_audit_report(
        self,
        operation_id: str,
        *,
        limit: int = 100,
        agent_kind: str | None = None,
        accepted: bool | None = None,
    ) -> dict[str, Any]:
        """Return the operation-level LLM/control observability report."""

        state = self.runtime_store.snapshot(operation_id)
        return build_operation_audit_report(
            state,
            limit=limit,
            agent_kind=agent_kind,
            accepted=accepted,
        )

    def get_control_cycle_history(self, operation_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent operation control cycle records."""

        state = self.runtime_store.snapshot(operation_id)
        report = build_operation_audit_report(state, limit=limit)
        return list(report["control_cycle_history"])

    def get_llm_decision_history(
        self,
        operation_id: str,
        *,
        limit: int = 20,
        agent_kind: str | None = None,
        accepted: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent operation-level LLM decision history entries."""

        state = self.runtime_store.snapshot(operation_id)
        report = build_operation_audit_report(
            state,
            limit=limit,
            agent_kind=agent_kind,
            accepted=accepted,
        )
        return list(report["llm_decision_history"])

    def list_findings(self, operation_id: str) -> list[dict[str, Any]]:
        """Return sanitized findings for one operation."""

        state = self.runtime_store.snapshot(operation_id)
        report = ReportGenerator().build_report(state)
        return list(report["findings"])

    def list_evidence(self, operation_id: str) -> list[dict[str, Any]]:
        """Return sanitized evidence artifacts for one operation."""

        state = self.runtime_store.snapshot(operation_id)
        report = ReportGenerator().build_report(state)
        return list(report["evidence"])

    def get_findings_graph(self, operation_id: str) -> dict[str, Any]:
        """Return a lightweight traceability graph for findings and evidence."""

        state = self.runtime_store.snapshot(operation_id)
        return ReportGenerator().graph(state)

    def export_findings_report(self, operation_id: str, *, format: ReportFormat = "json") -> dict[str, Any] | str:
        """Export a sanitized findings report."""

        state = self.runtime_store.snapshot(operation_id)
        return ReportGenerator().export(state, format=format)

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

        state = self.runtime_store.snapshot(operation_id)
        kg, ag, task_graph, graph_runtime = self._load_graph_memory(operation_id)
        if self.settings.recovery_enabled and self._needs_recovery(state):
            # 中文注释：
            # 自动恢复统一走 store 层，确保 memory/file backend 都会同步刷新恢复快照。
            state = self.runtime_store.recover_operation(operation_id, reason="unclean_shutdown")
            resume_summary = dict(state.execution.metadata.get("recovery", {}))
            self._log_operation_event(state, event_type="operation_resumed", **resume_summary)
        cycle_index = self._next_cycle_index(state)
        if task_graph.list_nodes():
            state.execution.metadata["task_graph"] = task_graph.to_dict()
        state.execution.metadata["graph_memory"] = self._graph_memory_metadata(
            kg=kg,
            ag=ag,
            tg=task_graph,
            loaded_runtime=graph_runtime is not None,
        )
        state.execution.metadata["agent_architecture"] = self._agent_architecture_metadata(self.settings)
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

        planning = self._run_stage_planning_phase(
            operation_id=operation_id,
            graph_refs=graph_refs,
            planner_payload=planner_payload,
            kg=kg,
            ag=ag,
            task_graph=task_graph,
            runtime_state=state,
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
        task_graph = TaskGraph.from_dict(state.execution.metadata.get("task_graph") or task_graph.to_dict())
        state.execution.metadata["task_graph"] = task_graph.to_dict()
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="planning_completed",
            status="completed" if planning.success else "failed",
            step_count=len(planning.steps),
            success=planning.success,
        )

        execution = self._run_stage_execution_phase(
            operation_id=operation_id,
            graph_refs=graph_refs,
            kg=kg,
            ag=ag,
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
        for task_result in self._execution_task_results(execution):
            applied_task_ids.append(task_result.task_id)
            applied = self.result_applier.apply(
                task_result,
                state,
                kg_ref=self._kg_ref(graph_refs),
                kg_store=kg,
                attack_graph=ag,
                task_graph=task_graph,
                goal_context=self._mapping((feedback_payload or {}).get("goal_context")),
                policy_context=self._mapping((feedback_payload or {}).get("policy_context")),
            )
            if applied.ag_graph is not None:
                ag = AttackGraph.from_dict(applied.ag_graph)
            if applied.tg_graph is not None:
                task_graph = TaskGraph.from_dict(applied.tg_graph)
            for delta in applied.visual_graph_deltas:
                graph_delta_publisher.publish_nowait(delta)
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
        self._save_graph_memory(
            operation_id=operation_id,
            kg=kg,
            ag=ag,
            tg=task_graph,
            runtime_state=state,
        )

        feedback = PipelineCycleResult(
            cycle_name="feedback_disabled",
            operation_id=operation_id,
            success=True,
            logs=["feedback critic phase disabled for the main execution chain"],
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
        state.execution.metadata["graph_memory"] = self._graph_memory_metadata(
            kg=kg,
            ag=ag,
            tg=task_graph,
            loaded_runtime=True,
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
        self._save_graph_memory(
            operation_id=operation_id,
            kg=kg,
            ag=ag,
            tg=task_graph,
            runtime_state=state,
        )
        self.graph_memory_store.save_snapshot(operation_id, cycle_index)
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
        max_replans: int = 3,
        consecutive_llm_rejections: int = 3,
        stop_when_quiescent: bool = True,
    ) -> list[OperationCycleResult]:
        """持续运行主循环，直到静止或达到上限。"""

        results: list[OperationCycleResult] = []
        llm_rejection_count = 0
        supervisor_replan_count = 0
        for _ in range(max_cycles):
            state_before = self.runtime_store.snapshot(operation_id)
            if self._budget_guard_triggered(state_before):
                guard_cycle_index = self._next_cycle_index(state_before)
                self._record_control_strategy(
                    state_before,
                    cycle_index=guard_cycle_index,
                    strategy="budget_guard",
                    accepted=True,
                    reason="budget guard triggered",
                )
                self._pause_for_review(
                    state_before,
                    reason="budget guard triggered",
                    cycle_index=guard_cycle_index,
                )
                self.runtime_store.save_state(state_before)
                break
            cycle_result = self.run_operation_cycle(
                operation_id,
                graph_refs=graph_refs,
                planner_payload=planner_payload,
                scheduler_payload=scheduler_payload,
                feedback_payload=feedback_payload,
                context=context,
            )
            results.append(cycle_result)
            supervisor_control = self._apply_supervisor_control_strategy(
                operation_id=operation_id,
                graph_refs=graph_refs,
                cycle_result=cycle_result,
                context=context,
                max_replans=max_replans,
                supervisor_replan_count=supervisor_replan_count,
            )
            if supervisor_control.get("llm_rejected"):
                llm_rejection_count += 1
            elif supervisor_control.get("llm_accepted"):
                llm_rejection_count = 0
            if supervisor_control.get("replan_requested"):
                supervisor_replan_count += 1
            if llm_rejection_count >= consecutive_llm_rejections:
                state = self.runtime_store.snapshot(operation_id)
                self._record_control_strategy(
                    state,
                    cycle_index=cycle_result.cycle_index,
                    strategy="deterministic_fallback",
                    accepted=False,
                    reason="consecutive llm rejections threshold reached",
                )
                self.runtime_store.save_state(state)
                llm_rejection_count = 0
            if supervisor_control.get("stop"):
                break
            if stop_when_quiescent and cycle_result.stopped:
                break
        return results

    def stop_operation(self, operation_id: str, *, reason: str = "manual_stop") -> RuntimeState:
        """Request a conservative operation stop without dispatching new work."""

        state = self.runtime_store.snapshot(operation_id)
        state.operation_status = RuntimeStatus.CANCELLED
        state.execution.status = RuntimeStatus.CANCELLED
        state.execution.metadata["stop_request"] = {
            "reason": reason,
            "requested_at": utc_now().isoformat(),
        }
        state.execution.summary = f"operation stopped: {reason}"
        state.last_updated = utc_now()
        self._log_operation_event(
            state,
            event_type="operation_stopped",
            stop_reason=reason,
            status=RuntimeStatus.CANCELLED.value,
        )
        self.runtime_store.save_state(state)
        return state.model_copy(deep=True)

    def _apply_supervisor_control_strategy(
        self,
        *,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        cycle_result: OperationCycleResult,
        context: dict[str, Any] | None,
        max_replans: int,
        supervisor_replan_count: int,
    ) -> dict[str, Any]:
        """Supervisor control is temporarily disabled in the main loop."""

        del operation_id, graph_refs, cycle_result, context, max_replans, supervisor_replan_count
        return {}

    def resume_operation(self, operation_id: str, *, reason: str = "manual_resume") -> RuntimeState:
        """Normalize in-flight runtime state so the operation can resume safely."""

        state = self.runtime_store.recover_operation(operation_id, reason=reason)
        summary = dict(state.execution.metadata.get("recovery", {}))
        state.operation_status = RuntimeStatus.READY
        state.execution.status = RuntimeStatus.READY
        self._log_operation_event(state, event_type="operation_resumed", **summary)
        self.runtime_store.save_state(state)
        return state.model_copy(deep=True)

    def _supervisor_payload_from_cycle(
        self,
        state: RuntimeState,
        cycle_result: OperationCycleResult,
    ) -> dict[str, Any]:
        last_control_cycle = self._mapping(state.execution.metadata.get("last_control_cycle"))
        return {
            "runtime_summary": self._runtime_summary(state),
            "last_control_cycle": last_control_cycle,
            "planner_summary": {
                "success": cycle_result.planning.success if cycle_result.planning is not None else False,
                "step_count": len(cycle_result.planning.steps) if cycle_result.planning is not None else 0,
            },
            "critic_summary": {
                "success": cycle_result.feedback.success if cycle_result.feedback is not None else False,
                "finding_count": self._critic_finding_count(cycle_result.feedback),
                "replan_request_count": len(state.replan_requests),
            },
            "budget_summary": self._budget_summary(state),
        }

    def _validated_supervisor_strategy(self, cycle: PipelineCycleResult) -> dict[str, Any] | None:
        for decision in cycle.final_output.decisions:
            payload = self._mapping(decision.get("payload"))
            validation = self._mapping(payload.get("llm_decision_validation"))
            if not bool(payload.get("control_only")):
                continue
            if not bool(payload.get("llm_adopted")) or not bool(validation.get("accepted")):
                continue
            supervisor_decision = self._mapping(payload.get("supervisor_decision"))
            strategy = supervisor_decision.get("strategy")
            if strategy not in {
                "continue_planning",
                "continue_execution",
                "request_replan",
                "pause_for_review",
                "stop_when_quiescent",
            }:
                continue
            return {
                "strategy": str(strategy),
                "rationale": supervisor_decision.get("rationale"),
                "requires_human_review": bool(supervisor_decision.get("requires_human_review", False)),
            }
        return None

    def _record_control_strategy(
        self,
        state: RuntimeState,
        *,
        cycle_index: int,
        strategy: str,
        accepted: bool,
        reason: str,
    ) -> None:
        record = {
            "cycle_index": cycle_index,
            "strategy": strategy,
            "accepted": accepted,
            "reason": reason,
            "created_at": utc_now().isoformat(),
        }
        state.execution.metadata["last_supervisor_control_strategy"] = record
        history = state.execution.metadata.setdefault("control_cycle_history", [])
        if history and isinstance(history[-1], dict) and history[-1].get("cycle_index") == cycle_index:
            history[-1]["supervisor_control_strategy"] = record
        else:
            history.append(
                {
                    "cycle_index": cycle_index,
                    "cycle_type": "control_guard",
                    "stopped": True,
                    "stop_reason": reason,
                    "supervisor_control_strategy": record,
                }
            )
        self._log_operation_event(
            state,
            event_type="supervisor_control_strategy_recorded",
            cycle_index=cycle_index,
            strategy=strategy,
            accepted=accepted,
            reason=reason,
        )

    def _pause_for_review(self, state: RuntimeState, *, reason: str, cycle_index: int) -> None:
        state.operation_status = RuntimeStatus.PAUSED
        state.execution.status = RuntimeStatus.PAUSED
        state.execution.metadata["pause_reason"] = reason
        state.execution.metadata["pause_cycle_index"] = cycle_index
        state.last_updated = utc_now()
        self._log_operation_event(
            state,
            event_type="operation_paused_for_review",
            cycle_index=cycle_index,
            reason=reason,
        )

    def _request_supervisor_replan(self, state: RuntimeState, *, cycle_index: int, rationale: str) -> None:
        request = ReplanRequest(
            request_id=f"supervisor-replan-{cycle_index}-{len(state.replan_requests) + 1}",
            reason="supervisor requested existing replan flow",
            scope="local",
            metadata={
                "source": "supervisor",
                "cycle_index": cycle_index,
                "rationale": rationale,
            },
        )
        state.request_replan(request)
        self._log_operation_event(
            state,
            event_type="supervisor_replan_requested",
            cycle_index=cycle_index,
            request_id=request.request_id,
        )

    @staticmethod
    def _budget_summary(state: RuntimeState) -> dict[str, Any]:
        budgets = state.budgets
        guards = {
            "operation": AppOrchestrator._budget_exhausted(budgets.operation_budget_used, budgets.operation_budget_max),
            "time": AppOrchestrator._budget_exhausted(budgets.time_budget_used_sec, budgets.time_budget_max_sec),
            "token": AppOrchestrator._budget_exhausted(budgets.token_budget_used, budgets.token_budget_max),
            "noise": AppOrchestrator._budget_exhausted(budgets.noise_budget_used, budgets.noise_budget_max),
            "risk": AppOrchestrator._budget_exhausted(budgets.risk_budget_used, budgets.risk_budget_max),
        }
        return {
            "operation_budget_used": budgets.operation_budget_used,
            "operation_budget_max": budgets.operation_budget_max,
            "time_budget_used_sec": budgets.time_budget_used_sec,
            "time_budget_max_sec": budgets.time_budget_max_sec,
            "token_budget_used": budgets.token_budget_used,
            "token_budget_max": budgets.token_budget_max,
            "noise_budget_used": budgets.noise_budget_used,
            "noise_budget_max": budgets.noise_budget_max,
            "risk_budget_used": budgets.risk_budget_used,
            "risk_budget_max": budgets.risk_budget_max,
            "requires_human_review": any(guards.values()),
            "guards": guards,
        }

    @staticmethod
    def _budget_guard_triggered(state: RuntimeState) -> bool:
        return bool(AppOrchestrator._budget_summary(state)["requires_human_review"])

    @staticmethod
    def _budget_exhausted(used: float | int, maximum: float | int | None) -> bool:
        return maximum is not None and used >= maximum

    @staticmethod
    def _critic_finding_count(feedback: PipelineCycleResult | None) -> int:
        if feedback is None:
            return 0
        count = 0
        for decision in feedback.final_output.decisions:
            payload = decision.get("payload") if isinstance(decision, dict) else None
            if isinstance(payload, dict) and "recommendation" in payload:
                count += 1
        return count

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

    def _load_graph_memory(self, operation_id: str) -> tuple[KnowledgeGraph, AttackGraph, TaskGraph, RuntimeState | None]:
        """Load KG / AG / TG / Runtime snapshots for the current operation cycle."""

        kg = self.graph_memory_store.load_kg(operation_id)
        ag = self.graph_memory_store.load_ag(operation_id)
        tg = self.graph_memory_store.load_tg(operation_id)
        runtime = self.graph_memory_store.load_runtime(operation_id)
        return kg, ag, tg, runtime

    def _ensure_graph_initialized_from_targets(self, state: RuntimeState) -> dict[str, Any] | None:
        """Initialize KG / AG / TG from imported targets when graph memory is empty."""

        kg, ag, tg, _runtime = self._load_graph_memory(state.operation_id)
        if kg.list_nodes() or ag.list_nodes() or tg.list_nodes():
            return None

        target = self._initial_graph_target(state)
        if target is None:
            return None

        result = GraphInitializer(self.graph_memory_store).initialize(
            operation_id=state.operation_id,
            target=target,
            persist=True,
        )
        return {
            "operation_id": result.operation_id,
            "target": result.target.raw,
            "target_kind": result.target.kind,
            "host_id": result.host_id,
            "goal_id": result.goal_id,
            "scope_id": result.scope_id,
            "initial_action_ids": list(result.initial_action_ids),
            "initial_task_ids": list(result.initial_task_ids),
        }

    @staticmethod
    def _initial_graph_target(state: RuntimeState) -> str | None:
        """Return the first imported target value suitable for graph initialization."""

        inventory = state.execution.metadata.get("target_inventory", [])
        if not isinstance(inventory, list):
            return None
        for item in inventory:
            if not isinstance(item, dict):
                continue
            for key in ("url", "value", "address", "hostname"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _planner_payload_with_graph_memory(
        self,
        planner_payload: dict[str, Any],
        *,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        tg: TaskGraph,
        runtime_state: RuntimeState,
        enable_graph_llm_planning: bool = False,
    ) -> dict[str, Any]:
        """Attach persisted graph memory snapshots to planner input."""

        policy_context = self._mapping(
            planner_payload.get("policy_context")
            or runtime_state.execution.metadata.get("runtime_policy")
        )
        planning_context = {
            **self._mapping(planner_payload.get("planning_context")),
        }
        if enable_graph_llm_planning:
            planning_context.setdefault("enable_graph_llm_planning", True)
        graph_context = GraphContextBuilder().build(
            knowledge_graph=kg,
            attack_graph=ag,
            task_graph=tg,
            runtime_state=runtime_state,
            policy_context=policy_context,
        )
        return {
            **dict(planner_payload),
            "kg_graph": kg.to_dict(),
            "ag_graph": ag.to_dict(),
            "tg_graph": tg.to_dict(),
            "task_graph": tg.to_dict(),
            "runtime_state": runtime_state.model_dump(mode="json"),
            "policy_context": policy_context,
            "planning_context": planning_context,
            "graph_context": graph_context.model_dump(mode="json"),
        }

    def _save_graph_memory(
        self,
        *,
        operation_id: str,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        tg: TaskGraph,
        runtime_state: RuntimeState,
    ) -> None:
        """Persist KG / AG / TG / Runtime graph memory snapshots."""

        self.graph_memory_store.save_kg(operation_id, kg)
        self.graph_memory_store.save_ag(operation_id, ag)
        self.graph_memory_store.save_tg(operation_id, tg)
        self.graph_memory_store.save_runtime(operation_id, runtime_state)

    @staticmethod
    def _graph_memory_metadata(
        *,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        tg: TaskGraph,
        loaded_runtime: bool,
    ) -> dict[str, Any]:
        return {
            "kg_version": kg.version,
            "ag_version": ag.version,
            "tg_version": tg.version,
            "source_kg_version": ag.source_kg_version,
            "source_ag_version": tg.source_ag_version,
            "frontier_version": tg.frontier_version,
            "loaded_runtime": loaded_runtime,
        }

    def _run_execution_phase(
        self,
        *,
        pipeline: AgentPipeline,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        kg: KnowledgeGraph,
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
        scheduler_payload: dict[str, Any] | None,
        context: dict[str, Any] | None,
    ) -> PipelineCycleResult:
        """运行 scheduling + worker execution 阶段。"""

        task_graph.refresh_blocked_states()
        ready_tasks = task_graph.find_schedulable_tasks()
        if self.settings.debug_scheduler_io:
            self._log_operation_event(
                runtime_state,
                event_type="scheduler_debug_input",
                ready_tasks=[
                    {
                        "id": task.id,
                        "type": task.task_type.value,
                        "status": task.status.value,
                        "source_action_id": task.source_action_id,
                        "target_refs": [ref.key() for ref in task.target_refs],
                        "resource_keys": sorted(task.resource_keys),
                        "assigned_agent": task.assigned_agent,
                    }
                    for task in ready_tasks
                ],
                draft_tasks=self._draft_task_debug_records(task_graph),
                task_graph_nodes=[
                    {
                        "id": node.id,
                        "kind": getattr(node, "kind", None),
                        "type": getattr(getattr(node, "task_type", None), "value", None),
                        "status": getattr(getattr(node, "status", None), "value", None),
                    }
                    for node in task_graph.list_nodes()
                ],
            )
        payload = {
            **self._mapping(scheduler_payload),
            "kg_graph": kg.to_dict(),
            "tg_graph": task_graph.to_dict(),
            "runtime_state": runtime_state.model_dump(mode="json"),
        }
        scheduling_result = schedule_ready_tasks(
            task_graph=task_graph,
            runtime_state=runtime_state,
            context=self._mapping(payload.get("scheduling_context")),
            operation_id=operation_id,
            graph_refs=graph_refs,
            payload=payload,
        )
        # 中文注释：
        # 调度器里的 worker_id 更接近 runtime worker 槽位，不一定等于 agent registry 名称。
        # 当仓库里只注册了一个 worker agent 时，这里显式指定它，避免 execution 阶段因为名称不一致而中断。
        execution = pipeline.run_worker_execution_cycle(
            operation_id=operation_id,
            graph_refs=graph_refs,
            scheduler_payload=payload,
            scheduler_output=scheduling_result.to_agent_output(),
            worker_agent=self._default_worker_agent_name(pipeline, scheduler_payload),
            context=context,
        )
        if self.settings.debug_scheduler_io:
            self._log_operation_event(
                runtime_state,
                event_type="scheduler_debug_output",
                success=execution.success,
                decisions=execution.final_output.decisions,
                logs=execution.final_output.logs,
                errors=execution.final_output.errors,
            )
        return execution

    def _run_stage_planning_phase(
        self,
        *,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        planner_payload: dict[str, Any],
        kg: KnowledgeGraph,
        ag: AttackGraph,
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
        context: dict[str, Any] | None,
    ) -> PipelineCycleResult:
        goal = self._mission_goal(planner_payload=planner_payload, context=context)
        graph_context = self._stage_graph_context(
            operation_id=operation_id,
            graph_refs=graph_refs,
            kg=kg,
            ag=ag,
            tg=task_graph,
            runtime_state=runtime_state,
            extra=planner_payload,
        )
        planner_result = self.mission_planner.run(
            goal=goal,
            graph_context=graph_context,
            policy_context=self._mapping(planner_payload.get("policy_context")),
        )
        created_ids = self.stage_task_builder.upsert_stage_tasks(
            task_graph,
            planner_result.stage_tasks,
            dependencies=planner_result.dependencies,
        )
        runtime_state.execution.metadata["task_graph"] = task_graph.to_dict()
        runtime_state.execution.metadata["stage_planning"] = planner_result.model_dump(mode="json")
        output = AgentOutput(
            decisions=[
                {
                    "id": f"stage-plan-{operation_id}",
                    "accepted": True,
                    "action": "propose_stage_tasks",
                    "stage_task_ids": [task.task_id for task in planner_result.stage_tasks],
                    "created_task_ids": created_ids,
                }
            ],
            logs=[planner_result.summary or f"stage planner proposed {len(planner_result.stage_tasks)} task(s)"],
        )
        now = utc_now()
        return PipelineCycleResult(
            cycle_name="stage_planning",
            operation_id=operation_id,
            success=True,
            steps=[
                PipelineStepResult(
                    step_name="stage_planner",
                    agent_name="mission_planner_agent",
                    agent_kind=AgentKind.PLANNER,
                    success=True,
                    agent_input=AgentInput(
                        graph_refs=graph_refs,
                        context=AgentContext(operation_id=operation_id),
                        raw_payload=planner_payload,
                    ),
                    agent_output=output,
                    started_at=now,
                    finished_at=now,
                    duration_ms=0,
                )
            ],
            final_output=output,
            logs=[f"stage planning created {len(created_ids)} task(s)"],
        )

    def _run_stage_execution_phase(
        self,
        *,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        kg: KnowledgeGraph,
        ag: AttackGraph,
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
        scheduler_payload: dict[str, Any] | None,
        context: dict[str, Any] | None,
    ) -> PipelineCycleResult:
        del scheduler_payload, context
        ready_stage_tasks = schedule_ready_stage_tasks(task_graph, runtime_state)
        tool_catalog = self.mcp_client.list_tools() if self.mcp_client is not None else {"available": False, "error": "MCP is not configured"}
        graph_context = self._stage_graph_context(
            operation_id=operation_id,
            graph_refs=graph_refs,
            kg=kg,
            ag=ag,
            tg=task_graph,
            runtime_state=runtime_state,
            extra={},
        )
        runtime_context = runtime_state.model_dump(mode="json")
        steps: list[PipelineStepResult] = []
        decisions: list[dict[str, Any]] = []
        for index, stage_task in enumerate(ready_stage_tasks, start=1):
            started_at = utc_now()
            agent = self.stage_registry.resolve(stage_task.stage_type)
            stage_result = agent.run(
                task=stage_task,
                graph_context=graph_context,
                runtime_context=runtime_context,
                tool_catalog=tool_catalog,
            )
            task_result = StageResultAdapter.to_task_result(stage_result)
            output = AgentOutput(
                outcomes=[
                    {
                        "task_id": task_result.task_id,
                        "outcome_type": stage_result.stage_type.value,
                        "success": task_result.status.value == "succeeded",
                        "summary": task_result.summary,
                        "payload": {"agent_task_result": task_result.model_dump(mode="json")},
                    }
                ],
                logs=[stage_result.summary],
                errors=[] if task_result.status.value in {"succeeded", "needs_replan"} else [task_result.summary],
            )
            agent_input = AgentInput(
                graph_refs=graph_refs,
                task_ref=stage_task.task_id,
                context=AgentContext(operation_id=operation_id, runtime_state_ref=operation_id),
                raw_payload={
                    "task_id": stage_task.task_id,
                    "task_type": stage_task.stage_type.value,
                    "tg_node_id": stage_task.task_id,
                    "agent_task_result": task_result.model_dump(mode="json"),
                },
            )
            finished_at = utc_now()
            steps.append(
                PipelineStepResult(
                    step_name=f"stage_agent[{index}]",
                    agent_name=agent.agent_name,
                    agent_kind=AgentKind.WORKER,
                    success=task_result.status.value in {"succeeded", "needs_replan"},
                    agent_input=agent_input,
                    agent_output=output,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=max(0, int((finished_at - started_at).total_seconds() * 1000)),
                )
            )
            decisions.append(
                {
                    "id": f"stage-schedule-{stage_task.task_id}",
                    "accepted": True,
                    "action": "assign",
                    "task_id": stage_task.task_id,
                    "worker_id": agent.agent_name,
                    "stage_type": stage_task.stage_type.value,
                }
            )
        final_output = AgentOutput(decisions=decisions, logs=[f"stage execution ran {len(steps)} stage task(s)"])
        for step in steps:
            final_output.outcomes.extend(step.agent_output.outcomes)
            final_output.logs.extend(step.agent_output.logs)
            final_output.errors.extend(step.agent_output.errors)
        return PipelineCycleResult(
            cycle_name="stage_execution",
            operation_id=operation_id,
            success=not final_output.errors,
            steps=steps,
            final_output=final_output,
            logs=[f"stage execution selected {len(ready_stage_tasks)} ready task(s)"],
            errors=list(final_output.errors),
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
        if self.settings.enable_critic_llm_advisor:
            payload.setdefault("enable_legacy_graph_critic", True)
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

        goal_satisfied = self._goal_satisfied(state)
        stopped = goal_satisfied or (not self._selected_task_ids(execution) and not state.replan_requests)
        stop_reason = (
            "goal satisfied by stage result"
            if goal_satisfied
            else "no schedulable work and no replan request"
            if stopped
            else None
        )
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
            "architecture": "graph_driven_multi_agent_multihost",
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

    @staticmethod
    def _goal_satisfied(state: RuntimeState) -> bool:
        if bool(state.execution.metadata.get("goal_satisfied")):
            return True
        for outcome in state.recent_outcomes:
            payload = outcome.metadata.get("outcome_payload")
            if not isinstance(payload, dict):
                continue
            stage_result = payload.get("stage_result")
            if not isinstance(stage_result, dict):
                continue
            if stage_result.get("stage_type") == "GOAL_STAGE" and stage_result.get("status") == "succeeded":
                hints = stage_result.get("runtime_hints")
                if not isinstance(hints, dict) or bool(hints.get("goal_satisfied", True)):
                    state.execution.metadata["goal_satisfied"] = True
                    return True
        return False

    @staticmethod
    def _mission_goal(*, planner_payload: dict[str, Any], context: dict[str, Any] | None) -> str:
        candidates = [
            planner_payload.get("mission_goal"),
            planner_payload.get("goal"),
            planner_payload.get("objective"),
            AppOrchestrator._mapping(context).get("mission_goal"),
            AppOrchestrator._mapping(context).get("goal"),
        ]
        for candidate in candidates:
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()
        return "Validate the authorized mission goal"

    @staticmethod
    def _stage_graph_context(
        *,
        operation_id: str,
        graph_refs: list[AgentGraphRef],
        kg: KnowledgeGraph,
        ag: AttackGraph,
        tg: TaskGraph,
        runtime_state: RuntimeState,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "operation_id": operation_id,
            "target_refs": [
                {
                    "graph": ref.graph.value if hasattr(ref.graph, "value") else str(ref.graph),
                    "ref_id": ref.ref_id,
                    "ref_type": ref.ref_type,
                    "label": None,
                }
                for ref in graph_refs
                if (ref.graph.value if hasattr(ref.graph, "value") else str(ref.graph)) in {"kg", "ag", "tg", "query"}
            ],
            "kg_graph": kg.to_dict(),
            "ag_graph": ag.to_dict(),
            "tg_graph": tg.to_dict(),
            "runtime_state": runtime_state.model_dump(mode="json"),
            "extra": dict(extra or {}),
        }

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

    @staticmethod
    def _execution_task_results(execution: PipelineCycleResult) -> list[AgentTaskResult]:
        results: list[AgentTaskResult] = []
        for step in execution.steps:
            if step.agent_kind != AgentKind.WORKER:
                continue
            explicit = step.agent_input.raw_payload.get("agent_task_result")
            if isinstance(explicit, AgentTaskResult):
                results.append(explicit)
            elif isinstance(explicit, dict):
                results.append(AgentTaskResult.model_validate(explicit))
        return results

    def _extract_llm_decision_history(
        self,
        *,
        cycle_index: int,
        cycle: PipelineCycleResult,
    ) -> list[LLMDecisionHistoryRecord]:
        return LLMDecisionObserver(self.settings).extract(cycle_index=cycle_index, cycle=cycle)

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
        return merge_task_graphs(current, TaskGraph.from_dict(patch_payload))

    @staticmethod
    def _mark_applied_tasks(task_graph: TaskGraph, task_ids: list[str]) -> None:
        """把已经执行并完成 apply 的 task 标记为 succeeded，避免下一轮重复调度。"""

        for task_id in task_ids:
            if task_id not in task_graph._nodes:
                continue
            node = task_graph.get_node(task_id)
            if isinstance(node, BaseTaskNode):
                task_graph.mark_task_status(task_id, TaskStatus.SUCCEEDED, reason=node.reason)

    def _selected_task_ids(self, execution: PipelineCycleResult) -> list[str]:
        if execution.final_output.decisions:
            return [
                str(decision.get("task_id"))
                for decision in execution.final_output.decisions
                if bool(decision.get("accepted")) and decision.get("task_id") is not None
            ]
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
    def _draft_task_debug_records(task_graph: TaskGraph) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for node in task_graph.list_nodes(status=TaskStatus.DRAFT):
            if isinstance(node, BaseTaskNode):
                dependency_predecessors = task_graph.predecessors(node.id, DependencyType.DEPENDS_ON)
                draft_blockers = []
                unsatisfied_dependencies = [
                    f"{pred.id}:{pred.status.value}"
                    for pred in dependency_predecessors
                    if pred.status != TaskStatus.SUCCEEDED
                ]
                if unsatisfied_dependencies:
                    draft_blockers.append(
                        "upstream dependencies not succeeded: "
                        + ", ".join(unsatisfied_dependencies)
                    )
                if node.gate_ids:
                    draft_blockers.append("gate_ids uncleared: " + ", ".join(sorted(node.gate_ids)))
                if not draft_blockers:
                    draft_blockers.append("draft after refresh_blocked_states without TG dependency blocker")
                records.append(
                    {
                        "id": node.id,
                        "type": node.task_type.value,
                        "status": node.status.value,
                        "reason": node.reason,
                        "draft_blockers": draft_blockers,
                        "preconditions": [ref.key() for ref in node.precondition_refs],
                        "targets": [ref.key() for ref in node.target_refs],
                        "source_action_id": node.source_action_id,
                        "gate_ids": sorted(node.gate_ids),
                        "resource_keys": sorted(node.resource_keys),
                        "predecessors": [
                            {
                                "id": pred.id,
                                "type": pred.task_type.value,
                                "status": pred.status.value,
                            }
                            for pred in task_graph.predecessors(node.id)
                        ],
                    }
                )
        return records

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
        if settings.enable_planner_rank_llm_advisor:
            enabled_advisors.append("enable_planner_rank_llm_advisor")
        if settings.enable_graph_llm_planner_advisor:
            enabled_advisors.append("enable_graph_llm_planner_advisor")
        if settings.enable_critic_llm_advisor:
            enabled_advisors.append("enable_critic_llm_advisor")
        if settings.enable_supervisor_llm_advisor:
            enabled_advisors.append("enable_supervisor_llm_advisor")
        if enabled_advisors and llm_client_config is None:
            raise ValueError(f"{', '.join(enabled_advisors)} require llm_api_key in AppSettings")
        mcp_client = (
            ConfiguredMCPClient.from_sources(
                config_path=settings.mcp_config_path,
                config_json=settings.mcp_config_json,
            )
            if settings.mcp_enabled
            else None
        )
        llm_worker_advisor = (
            LLMWorkerAdvisor(client=PackyLLMClient(llm_client_config))
            if llm_client_config is not None
            else None
        )
        return build_optional_agent_pipeline(
            options=AgentPipelineAssemblyOptions(
                enable_packy_planner_advisor=AppOrchestrator._planner_rank_llm_enabled(settings),
                enable_graph_llm_planner_advisor=AppOrchestrator._graph_llm_planner_enabled(settings),
                enable_packy_critic_advisor=False,
                enable_packy_supervisor_advisor=False,
                include_scheduler=False,
                include_critic=False,
                include_supervisor=False,
                include_recon_worker=False,
                include_llm_worker=True,
            ),
            llm_client_config=llm_client_config,
            llm_worker_agent=LLMWorkerAgent(
                advisor=llm_worker_advisor,
                mcp_client=mcp_client,
                default_timeout_seconds=settings.mcp_default_timeout_seconds,
            ),
        )

    @staticmethod
    def _llm_advisor_status(settings: AppSettings) -> dict[str, Any]:
        config = settings.to_packy_llm_config()
        return {
            "planner_enabled": AppOrchestrator._planner_llm_enabled(settings),
            "planner_rank_enabled": AppOrchestrator._planner_rank_llm_enabled(settings),
            "graph_planner_enabled": AppOrchestrator._graph_llm_planner_enabled(settings),
            "critic_enabled": settings.enable_critic_llm_advisor,
            "supervisor_enabled": settings.enable_supervisor_llm_advisor,
            "configured": config is not None,
            "model": config.model if config is not None else None,
            "base_url": config.base_url if config is not None else None,
        }

    @staticmethod
    def _agent_architecture_metadata(settings: AppSettings) -> dict[str, Any]:
        """Stable description of the graph-driven multi-agent runtime."""

        llm_configured = settings.to_packy_llm_config() is not None
        llm_stage_enabled = llm_configured and AppOrchestrator._planner_llm_enabled(settings)
        return {
            "architecture": "graph_driven_multi_agent_multihost",
            "graph_layer": {
                "components": ["KG", "AG", "TG", "runtime"],
                "llm_enabled": False,
            },
            "planner_agent": {
                "implementation": "MissionPlannerAgent",
                "enabled": True,
                "reads": ["KG", "AG", "TG", "runtime"],
                "writes": ["TG stage tasks"],
                "reasoning_backend": "llm" if llm_stage_enabled else "deterministic",
            },
            "execution_layer": {
                "enabled": True,
                "reasoning_backend": "llm" if llm_stage_enabled else "deterministic",
                "agents": [
                    "ReconAgent",
                    "VulnAnalysisAgent",
                    "ExploitAgent",
                    "AccessPivotAgent",
                    "GoalAgent",
                ],
            },
            "stage_result_writeback": {
                "adapter": "StageResultAdapter",
                "applier": "PhaseTwoResultApplier",
                "llm_enabled": False,
                "writes": ["KG facts", "AG state", "TG status/candidates", "runtime sessions/credentials/pivots"],
            },
        }

    @staticmethod
    def _planner_rank_llm_enabled(settings: AppSettings) -> bool:
        return settings.enable_planner_rank_llm_advisor or settings.enable_planner_llm_advisor

    @staticmethod
    def _graph_llm_planner_enabled(settings: AppSettings) -> bool:
        return settings.enable_graph_llm_planner_advisor or settings.enable_planner_llm_advisor

    @staticmethod
    def _planner_llm_enabled(settings: AppSettings) -> bool:
        return (
            settings.enable_planner_llm_advisor
            or settings.enable_planner_rank_llm_advisor
            or settings.enable_graph_llm_planner_advisor
        )

    @staticmethod
    def _build_runtime_store(settings: AppSettings) -> RuntimeStore:
        if settings.runtime_store_backend == "memory":
            return InMemoryRuntimeStore()
        return FileRuntimeStore(settings.runtime_store_dir)


__all__ = ["AppOrchestrator", "OperationCycleResult", "OperationSummary", "TargetHost"]
