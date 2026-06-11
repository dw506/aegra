"""Top-level orchestration entry point."""

from __future__ import annotations

from typing import Any
import inspect
import importlib

from pydantic import BaseModel, ConfigDict, Field

from src.app.settings import AppSettings
from src.app.llm_decision_observer import LLMDecisionObserver
from src.core.agents.pipeline_results import PipelineCycleResult
from src.core.agents.agent_protocol import GraphRef as AgentGraphRef
from src.core.agents.graph_context import TwoGraphContextBuilder
from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.configured_mcp_client import ConfiguredMCPClient
from src.core.graph.graph_initializer import GraphInitializer
from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState, RuntimeStatus, utc_now
from src.core.models.scope import Asset, Engagement
from src.core.planning.llm_mission_planner_advisor import LLMMissionPlannerAdvisor
from src.core.planning.mission_planner_agent import MissionPlannerAgent
from src.core.planning.models import PlannerDecision
from src.core.runtime.audit_report import build_operation_audit_report
from src.core.runtime.observability import (
    append_operation_log,
    mark_clean_shutdown,
    mark_unclean_shutdown,
    prepare_state_for_resume,
    record_phase_checkpoint,
)
from src.core.runtime.report_generator import ReportFormat, ReportGenerator
from src.core.runtime.llm_history import (
    LLMDecisionHistoryRecord,
    append_llm_decision_history,
    ensure_llm_decision_history,
    recent_llm_decision_history,
)
from src.core.runtime.result_applier import PhaseTwoApplyResult, PhaseTwoResultApplier
from src.core.runtime.attack_log_extractor import AttackLogExtractor
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.dispatcher import StageDispatcher
from src.core.stage.registry import StageAgentRegistry
from src.core.runtime.store import FileRuntimeStore, InMemoryRuntimeStore, RuntimeStore
from src.core.visualization.graph_publisher import graph_delta_publisher
from src.integrations.mcp_lab.catalog import build_default_lab_tool_catalog

#把用户输入的目标转换成标准 Asset
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
        kind = self._normalized_kind(value)
        return Asset(
            asset_id=str(self.metadata.get("asset_id")) if self.metadata.get("asset_id") else None,
            kind=kind,  # type: ignore[arg-type]
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

    def _normalized_kind(self, value: str) -> str:
        kind = str(self.kind or "").strip().lower()
        if kind in {"network", "cidr_range"}:
            return "cidr"
        if kind == "host_or_network_or_url":
            if "://" in value:
                return "url"
            if "/" in value:
                return "cidr"
            return "host"
        return kind or "host"

#返回 operation 简要状态
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
    planner_decision: PlannerDecision | None = None
    planning_success: bool = False
    stage_result: dict[str, Any] | None = None
    execution_success: bool = False
    apply_results: list[PhaseTwoApplyResult] = Field(default_factory=list)
    selected_task_ids: list[str] = Field(default_factory=list)
    applied_task_ids: list[str] = Field(default_factory=list)
    stopped: bool = False
    stop_reason: str | None = None
    runtime_state: RuntimeState

#表示 /run 最终返回结果
class OperationRunSummary(BaseModel):
    """Stable final result contract returned to API, scripts and users."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    status: str
    stop_reason: str
    success: bool
    success_condition_progress: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    findings_url: str
    evidence_url: str
    graph_url: str
    audit_url: str


class AppOrchestrator:
    """Application-layer orchestration facade for phase-one control flows."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        runtime_store: RuntimeStore | None = None,
        graph_memory_store: GraphMemoryStore | None = None,
        result_applier: PhaseTwoResultApplier | None = None,
    ) -> None:
        self.settings = settings or AppSettings.from_env()
        self.runtime_store = runtime_store or self._build_runtime_store(self.settings)
        self.graph_memory_store = graph_memory_store or GraphMemoryStore(self.settings.runtime_store_dir)
        self.result_applier = result_applier or PhaseTwoResultApplier()
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
            if llm_client_config is not None
            else None
        )
        self.stage_registry = StageAgentRegistry.default(
            llm_client=stage_llm_client,
            mcp_client=self.mcp_client,
            default_timeout_seconds=self.settings.mcp_default_timeout_seconds,
        )
        mission_advisor = (
            LLMMissionPlannerAdvisor(client=stage_llm_client)
            if stage_llm_client is not None
            else None
        )
        self.mission_planner = MissionPlannerAgent(advisor=mission_advisor)

    def create_operation(self, operation_id: str, metadata: dict[str, Any] | None = None) -> RuntimeState:
        """Create a new operation with control-plane metadata attached."""

        runtime_policy = self.settings.load_runtime_policy()
        lab_profile = self.settings.load_lab_profile()
        public_lab_profile = self._public_lab_profile_metadata(lab_profile)
        operation_metadata = {        #某一次 operation 的附加运行上下文信息
            "control_plane": {            #运行控制参数
                "audit_enabled": self.settings.audit_enabled,               #用于决定 operation 中是否记录审计事件
                "audit_persist_enabled": self.settings.audit_persist_enabled,
                
                # 日志治理参数跟随 operation 一起固化到 runtime metadata，
                # 这样后续 resume、export 和不同 store 后端都能复用同一份配置。
                "audit_max_entries": self.settings.audit_max_entries,
                "operation_log_max_entries": self.settings.operation_log_max_entries,  #operation 普通运行日志最多保留多少条
                "audit_redaction_enabled": self.settings.audit_redaction_enabled,      #是否开启审计脱敏
                "recovery_enabled": self.settings.recovery_enabled,
                "max_concurrent_workers": self.settings.max_concurrent_workers,        #最大并发 worker 数
                "default_operation_budget": self.settings.default_operation_budget,
                "default_scan_timeout_sec": self.settings.default_scan_timeout_sec,
                "llm_advisors": self._llm_advisor_status(self.settings),  
                "agent_architecture": self._agent_architecture_metadata(self.settings),            #当前 agent 架构信息
            },
          
            # operation metadata 中只落稳定 JSON 结构，避免把 Pydantic 对象直接塞进 state。
            "runtime_policy": runtime_policy.to_runtime_metadata(),              #保存运行策略和授权范围
            "lab_profile": public_lab_profile,
            "target_inventory": [],                       #用户导入的目标列表
            "target_count": 0,
            "llm_decision_history": [],                  #LLM 决策历史

        }
        if metadata:
            operation_metadata.update(metadata)
        operation_metadata["lab_activation"] = self._lab_activation_metadata(lab_profile)
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
    #导入目标
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
    
    #启动 operation 流程
    def start_operation(self, operation_id: str) -> RuntimeState:     
        """Mark an operation ready for the first planning/execution cycle."""

        started_at = utc_now()
        state = self.runtime_store.snapshot(operation_id)
        graph_initialization = self._ensure_graph_initialized_from_targets(state)     #初始化 KG / AG
        if graph_initialization is not None:
            state.execution.metadata["graph_initialization"] = graph_initialization
            kg, ag, graph_runtime = self._load_graph_memory(operation_id)
            state.execution.metadata["graph_memory"] = {
                "kg_version": kg.version,
                "ag_version": ag.version,
                "source_kg_version": ag.source_kg_version,
                "loaded_runtime": graph_runtime is not None,
            }
            self._log_operation_event(
                state,
                event_type="graph_memory_initialized",
                target=graph_initialization["target"],
                initial_action_count=len(graph_initialization["initial_action_ids"]),
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

    def get_operation_run_summary(
        self,
        operation_id: str,
        *,
        cycle_results: list[OperationCycleResult] | None = None,
        max_cycles: int | None = None,
    ) -> OperationRunSummary:
        """Return the fixed final result contract for `/run` and automation scripts.

        Operation success is owned by the orchestrator summary layer. MCP tools
        and GoalAgent may contribute evidence and hints, but they do not emit
        the operation-level success contract directly.
        """

        state = self.runtime_store.snapshot(operation_id)
        progress = self._mapping(state.execution.metadata.get("success_condition_progress"))
        success = state.operation_status == RuntimeStatus.COMPLETED and (
            bool(progress.get("all_required_satisfied"))
            or self._goal_satisfied(state, cycle_results or [])
        )
        status = self._result_status(state=state, success=success, progress=progress)
        stop_reason = self._result_stop_reason(
            state=state,
            success=success,
            cycle_results=cycle_results or [],
            max_cycles=max_cycles,
        )
        return OperationRunSummary(
            operation_id=operation_id,
            status=status,
            stop_reason=stop_reason,
            success=success,
            success_condition_progress=progress,
            evidence_ids=self._success_condition_evidence_ids(progress),
            findings_url=f"/operations/{operation_id}/findings",
            evidence_url=f"/operations/{operation_id}/evidence",
            graph_url=f"/operations/{operation_id}/graph",
            audit_url=f"/operations/{operation_id}/audit-report",
        )

    def list_operations(self) -> list[OperationSummary]:
        """Return summaries for all known operations."""

        summaries: list[OperationSummary] = []
        for operation_id in self.runtime_store.list_operation_ids():
            try:
                summaries.append(self.get_operation_summary(operation_id))
            except Exception as exc:
                summaries.append(
                    OperationSummary(
                        operation_id=operation_id,
                        operation_status=RuntimeStatus.FAILED,
                        last_updated="1970-01-01T00:00:00+00:00",
                        metadata={
                            "load_error": {
                                "error_type": exc.__class__.__name__,
                                "message": str(exc),
                            }
                        },
                    )
                )
        return summaries

    def recover_operation(self, operation_id: str, *, reason: str = "manual_recover") -> RuntimeState:
        """Normalize runtime state without forcing the operation back to ready."""

        state = self.runtime_store.recover_operation(operation_id, reason=reason)
        state.operation_status = RuntimeStatus.PAUSED
        state.execution.status = RuntimeStatus.PAUSED
        recovery = state.execution.metadata.setdefault("recovery", {})
        if isinstance(recovery, dict):
            recovery["lifecycle_state"] = "ready_to_resume"
            recovery["active_cycle_id"] = None
            recovery["active_stage_id"] = None
            recovery["runtime_lock"] = "released"
        state.execution.metadata["lifecycle_state"] = "ready_to_resume"
        state.execution.summary = f"operation recovered: {reason}"
        state.last_updated = utc_now()
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

    def run_operation_cycle(          #单轮主循环
        self,
        operation_id: str,
        *,
        graph_refs: list[AgentGraphRef],
        planner_payload: dict[str, Any],
        feedback_payload: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> OperationCycleResult:
        del feedback_payload

        state = self.runtime_store.snapshot(operation_id)
        kg = self.graph_memory_store.load_kg(operation_id)
        ag = self.graph_memory_store.load_ag(operation_id)
        stored_runtime = self.graph_memory_store.load_runtime(operation_id)
        if stored_runtime is not None:
            state = stored_runtime
        if self.settings.recovery_enabled and self._needs_recovery(state):
            state = self.runtime_store.recover_operation(operation_id, reason="unclean_shutdown")
            resume_summary = dict(state.execution.metadata.get("recovery", {}))
            self._log_operation_event(state, event_type="operation_resumed", **resume_summary)

        cycle_index = self._next_cycle_index(state)
        txt_logger = TxtTraceLogger(operation_id)
        operation_trace_logger = TxtTraceLogger.operation_trace(operation_id)
        txt_logger.write_block(
            "SYSTEM",
            "operation cycle started",
            {
                "operation_id": operation_id,
                "cycle": cycle_index,
                "policy_gate": "audit_only_non_blocking",
                "json_run_file": "disabled_or_secondary",
                "txt_trace": True,
                "graph_write": True,
                "result_applier": "enabled",
                "attack_log_extractor": "enabled",
            },
        )
        operation_trace_logger.write_block(
            "CYCLE_START",
            "operation cycle started",
            {
                "operation_id": operation_id,
                "cycle_index": cycle_index,
                "policy_gate": "audit_only_non_blocking",
                "txt_trace": True,
            },
        )
        state.execution.metadata["graph_memory"] = {
            "kg_version": kg.version,
            "ag_version": ag.version,
            "source_kg_version": ag.source_kg_version,
            "loaded_runtime": stored_runtime is not None,
        }
        state.execution.metadata["agent_architecture"] = self._agent_architecture_metadata(self.settings)
        state.operation_status = RuntimeStatus.RUNNING
        state.execution.status = RuntimeStatus.RUNNING
        mark_unclean_shutdown(state, cycle_index=cycle_index)
        self._log_operation_event(state, event_type="cycle_started", cycle_index=cycle_index)
        txt_logger.write("CYCLE", f"cycle={cycle_index} started")
        self._checkpoint_phase(state, cycle_index=cycle_index, phase="cycle_started", status="running")

        goal = self._mission_goal(planner_payload=planner_payload, context=context)
        policy_context = self._mapping(
            planner_payload.get("policy_context")
            or state.execution.metadata.get("runtime_policy")
        )
        blackbox_policy_context = self._blackbox_policy_context(policy_context, state)
        tool_catalog = self._load_tool_catalog()
        state.execution.metadata["tool_catalog"] = tool_catalog
        txt_logger.write_block(
            "MISSION",
            "mission context",
            {
                "mission_goal": goal,
                "targets": state.execution.metadata.get("target_inventory", []),
                "scope": blackbox_policy_context,
                "tool_catalog": tool_catalog,
            },
        )
        planner_payload_for_snapshot = {**planner_payload, "policy_context": blackbox_policy_context}
        graph_snapshot = self._kg_ag_runtime_snapshot(
            operation_id=operation_id,
            cycle_index=cycle_index,
            graph_refs=graph_refs,
            kg=kg,
            ag=ag,
            runtime_state=state,
            extra=planner_payload_for_snapshot,
        )
        graph_snapshot["agent_capabilities"] = self.stage_registry.capability_summary()
        graph_snapshot["mcp_tool_catalog"] = tool_catalog
        # Surface the authoritative success-condition gate so PlannerAgent can read
        # eligible_for_stop before deciding whether to emit stop_success.
        graph_snapshot["success_condition_progress"] = self._mapping(
            state.execution.metadata.get("success_condition_progress")
        )
        try:
            decision = self.mission_planner.run(
                goal=goal,
                graph_context=graph_snapshot,
                policy_context=blackbox_policy_context,
                recent_stage_results=self._planner_recent_stage_results(state),
            )
            operation_trace_logger.write_block(
                "PLANNER_DECISION",
                "planner decision",
                {
                    "cycle_index": cycle_index,
                    "decision": decision.decision,
                    "selected_agent": decision.selected_agent,
                    "selected_stage": decision.selected_stage,
                    "objective": decision.objective,
                    "task_brief": decision.task_brief,
                    "max_steps": decision.max_steps,
                },
            )
        except Exception as exc:
            return self._fail_operation_cycle(
                operation_id=operation_id,
                cycle_index=cycle_index,
                state=state,
                kg=kg,
                ag=ag,
                phase="planning",
                exc=exc,
                txt_logger=txt_logger,
            )
        if decision.operation_id != operation_id:
            original_operation_id = decision.operation_id
            decision.operation_id = operation_id
            self._log_operation_event(
                state,
                event_type="planner_decision_operation_id_normalized",
                cycle_index=cycle_index,
                original_operation_id=original_operation_id,
                normalized_operation_id=operation_id,
            )
        decision.cycle_index = cycle_index
        txt_logger.write_block(
            "PLANNER",
            "planner decision",
            {
                "cycle": cycle_index,
                "selected_agent": decision.selected_agent,
                "selected_stage": decision.selected_stage,
                "objective": decision.objective,
                "risk_level": decision.risk_level,
                "confidence": decision.confidence,
                "reasoning_summary": decision.reasoning_summary,
                "success_criteria": list(decision.success_criteria),
                "target_refs": list(decision.target_refs),
            },
        )
        planning_apply = self.result_applier.apply_planner_decision(decision, state, kg, ag)
        for delta in planning_apply.visual_graph_deltas:
            graph_delta_publisher.publish_nowait(delta)

        planning_success = decision.decision not in {"stop_failed"}
        apply_results: list[PhaseTwoApplyResult] = [planning_apply]
        stage_result_payload: dict[str, Any] | None = None
        execution_success = True
        applied_ids: list[str] = []
        stopped = decision.decision in {"stop_success", "stop_failed", "pause_for_review"}
        stop_reason = decision.stop_condition or (decision.decision if stopped else None)

        if decision.decision == "dispatch_agent":
            try:
                stage_result = StageDispatcher(self.stage_registry).dispatch(
                    decision,
                    kg_snapshot=graph_snapshot,
                    ag_process_history={},
                    runtime_context=self._blackbox_runtime_context(state),
                    policy_context=blackbox_policy_context,
                    mcp_tool_catalog=tool_catalog,
                )
                if stage_result.operation_id != operation_id:
                    original_operation_id = stage_result.operation_id
                    stage_result.operation_id = operation_id
                    stage_result.runtime_hints = {
                        **dict(stage_result.runtime_hints),
                        "normalized_operation_id_from": original_operation_id,
                    }
                    self._log_operation_event(
                        state,
                        event_type="stage_result_operation_id_normalized",
                        cycle_index=cycle_index,
                        stage_task_id=stage_result.stage_task_id,
                        original_operation_id=original_operation_id,
                        normalized_operation_id=operation_id,
                    )
                stage_apply = self.result_applier.apply_stage_result(stage_result, state, kg, ag)
            except Exception as exc:
                return self._fail_operation_cycle(
                    operation_id=operation_id,
                    cycle_index=cycle_index,
                    state=state,
                    kg=kg,
                    ag=ag,
                    phase="stage_dispatch",
                    exc=exc,
                    txt_logger=txt_logger,
                    planner_decision=decision,
                    planning_success=planning_success,
                    apply_results=apply_results,
                )
            stage_graph_write_payload = {
                "cycle": cycle_index,
                "stage": stage_result.stage_type,
                "agent": stage_result.agent_name,
                "stage_status": stage_result.status,
                "kg_delta_count": len(stage_apply.kg_state_deltas or []),
                "kg_write_diagnostics": dict(stage_apply.kg_write_diagnostics or {}),
                "ag_delta_count": len(stage_apply.ag_state_deltas or []),
                "tg_delta_count": 0,
                "runtime_event_count": len(stage_apply.runtime_event_refs or []),
                "created_ag_nodes": len((stage_apply.ag_graph or {}).get("nodes", [])) if isinstance(stage_apply.ag_graph, dict) else 0,
                "created_ag_edges": len((stage_apply.ag_graph or {}).get("edges", [])) if isinstance(stage_apply.ag_graph, dict) else 0,
                "evidence_refs": list(stage_result.evidence_refs),
            }
            txt_logger.write_block("GRAPH_WRITE", "graph write completed", stage_graph_write_payload)
            operation_trace_logger.write_block("GRAPH_WRITE", "ResultApplier graph write completed", stage_graph_write_payload)
            extraction = AttackLogExtractor().extract(
                decision,
                stage_result,
                stage_result.tool_trace,
                [event.model_dump(mode="json") for event in state.pending_events[-20:]],
                [],
            )
            log_apply = self.result_applier.apply_log_extraction(extraction, state, ag)
            attack_log_payload = {
                "cycle": cycle_index,
                "stage": stage_result.stage_type,
                "agent": stage_result.agent_name,
                "status": stage_result.status,
                "ag_node_count": len(extraction.ag_nodes),
                "ag_edge_count": len(extraction.ag_edges),
                "node_roles": [
                    getattr(node, "node_type", None).value if hasattr(getattr(node, "node_type", None), "value") else str(getattr(node, "node_type", ""))
                    for node in extraction.ag_nodes
                ],
                "evidence_refs": list(extraction.evidence_refs),
            }
            txt_logger.write_block("ATTACK_LOG_EXTRACT", "attack graph extraction completed", attack_log_payload)
            operation_trace_logger.write_block("ATTACK_LOG_EXTRACT", "attack graph extraction completed", attack_log_payload)
            attack_log_graph_payload = {
                "cycle": cycle_index,
                "stage": stage_result.stage_type,
                "agent": stage_result.agent_name,
                "stage_status": stage_result.status,
                "kg_delta_count": 0,
                "ag_delta_count": len(extraction.ag_nodes) + len(extraction.ag_edges),
                "tg_delta_count": 0,
                "runtime_event_count": len(log_apply.runtime_event_refs or []),
                "created_ag_nodes": len(extraction.ag_nodes),
                "created_ag_edges": len(extraction.ag_edges),
                "evidence_refs": list(extraction.evidence_refs),
            }
            txt_logger.write_block("GRAPH_WRITE", "attack log graph write completed", attack_log_graph_payload)
            operation_trace_logger.write_block("GRAPH_WRITE", "attack log graph write completed", attack_log_graph_payload)
            for applied in (stage_apply, log_apply):
                for delta in applied.visual_graph_deltas:
                    graph_delta_publisher.publish_nowait(delta)
            apply_results.extend([stage_apply, log_apply])
            applied_ids.append(stage_result.stage_task_id)
            stage_result_payload = stage_result.model_dump(mode="json")
            execution_success = stage_result.status not in {"failed", "blocked"}
            self._update_success_condition_progress(state=state, stage_result=stage_result)
            if self._goal_satisfied(state):
                state.execution.metadata["goal_satisfied"] = True
            if stage_result.status in {"failed", "blocked"}:
                stop_reason = stage_result.summary
                state.execution.metadata["needs_replan"] = True
                state.execution.metadata["last_stage_stop_reason"] = {
                    "status": stage_result.status,
                    "summary": stage_result.summary,
                    "stage_task_id": stage_result.stage_task_id,
                    "recorded_at": utc_now().isoformat(),
                }
            else:
                stop_reason = None
            stopped = False

        summary = {
            "cycle_index": cycle_index,
            "started_at": utc_now().isoformat(),
            "architecture": "planner_stage_agent_graph_main_path",
            "planning_success": planning_success,
            "execution_success": execution_success,
            "selected_agent": decision.selected_agent,
            "selected_stage": decision.selected_stage,
            "applied_results": len(apply_results),
            "stopped": stopped,
            "stop_reason": stop_reason,
        }
        history = state.execution.metadata.setdefault("control_cycle_history", [])
        history.append(summary)
        state.execution.metadata["last_control_cycle"] = summary
        state.execution.summary = f"control cycle {cycle_index} completed"
        if decision.decision == "stop_success":
            state.operation_status = RuntimeStatus.COMPLETED
        elif decision.decision == "stop_failed":
            state.operation_status = RuntimeStatus.FAILED
        elif decision.decision == "pause_for_review":
            state.operation_status = RuntimeStatus.PAUSED
        else:
            state.operation_status = RuntimeStatus.READY
        state.execution.status = state.operation_status
        state.last_updated = utc_now()
        self._log_operation_event(
            state,
            event_type="cycle_completed",
            cycle_index=cycle_index,
            stopped=stopped,
            stop_reason=stop_reason,
        )
        txt_logger.write_block(
            "CYCLE",
            "cycle completed",
            {
                "cycle": cycle_index,
                "selected_agent": decision.selected_agent,
                "selected_stage": decision.selected_stage,
                "status": state.operation_status.value,
                "stop_reason": stop_reason,
            },
        )
        operation_trace_logger.write_block(
            "CYCLE_END",
            "cycle completed",
            {
                "cycle": cycle_index,
                "selected_agent": decision.selected_agent,
                "selected_stage": decision.selected_stage,
                "status": state.operation_status.value,
                "stop_reason": stop_reason,
                "applied_results": len(apply_results),
            },
        )
        mark_clean_shutdown(state, cycle_index=cycle_index)
        self._mark_operation_cycle_clean(state, cycle_index=cycle_index)
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="cycle_completed",
            status="completed",
            applied_task_ids=applied_ids,
            stopped=stopped,
            stop_reason=stop_reason,
            persist=False,
        )
        self.runtime_store.save_state(state)
        self.graph_memory_store.save_kg(operation_id, kg)
        self.graph_memory_store.save_ag(operation_id, ag)
        self.graph_memory_store.save_runtime(operation_id, state)
        self.graph_memory_store.save_snapshot(operation_id, cycle_index)
        return OperationCycleResult(
            operation_id=operation_id,
            cycle_index=cycle_index,
            planner_decision=decision,
            planning_success=planning_success,
            stage_result=stage_result_payload,
            execution_success=execution_success,
            apply_results=apply_results,
            selected_task_ids=applied_ids,
            applied_task_ids=list(applied_ids),
            stopped=stopped,
            stop_reason=stop_reason,
            runtime_state=state.model_copy(deep=True),
        )

    @staticmethod
    def _mark_operation_cycle_clean(state: RuntimeState, *, cycle_index: int) -> None:
        """Clear stale recovery markers after a cycle reaches a stable end state."""

        now = utc_now().isoformat()
        recovery = state.execution.metadata.setdefault("recovery", {})
        if isinstance(recovery, dict):
            recovery["lifecycle_state"] = "cycle_completed"
            recovery["active_cycle_id"] = None
            recovery["active_stage_id"] = None
            recovery["runtime_lock"] = "released"
            recovery["unclean_shutdown"] = False
            recovery["last_phase"] = "cycle_completed"
            recovery["last_phase_status"] = "completed"
            recovery["last_cycle_completed_at"] = now
            recovery["last_cycle_index"] = cycle_index
            recovery.pop("last_error", None)
        state.execution.metadata["lifecycle_state"] = "cycle_completed"
        state.execution.metadata["active_cycle_id"] = None
        state.execution.metadata["active_stage_id"] = None
        state.execution.metadata["runtime_lock"] = "released"
        state.execution.metadata["unclean_shutdown"] = False
        state.execution.metadata.pop("last_error", None)

    def run_until_quiescent(
        self,
        operation_id: str,
        *,
        graph_refs: list[AgentGraphRef],
        planner_payload: dict[str, Any],
        feedback_payload: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_cycles: int = 5,
        max_replans: int = 3,
        consecutive_llm_rejections: int = 3,
        stop_when_quiescent: bool = True,
    ) -> list[OperationCycleResult]:
        """持续运行主循环，直到静止或达到上限。"""

        results: list[OperationCycleResult] = []
        planner_replan_count = 0
        # consecutive_llm_rejections is retained for API request compatibility; the
        # supervisor-driven rejection/replan path is not part of the stage-agent main loop.
        del consecutive_llm_rejections
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
                feedback_payload=feedback_payload,
                context=context,
            )
            results.append(cycle_result)
            planner_decision = self._cycle_planner_decision(cycle_result)
            if planner_decision in {"stop_success", "stop_failed", "pause_for_review"} or cycle_result.stopped:
                break
            if planner_decision == "replan":
                planner_replan_count += 1
                if planner_replan_count > max_replans:
                    state = self.runtime_store.snapshot(operation_id)
                    self._record_control_strategy(
                        state,
                        cycle_index=cycle_result.cycle_index,
                        strategy="max_replans",
                        accepted=False,
                        reason="planner replan limit reached",
                    )
                    self.runtime_store.save_state(state)
                    break
            elif planner_decision in {"dispatch_agent"}:
                planner_replan_count = 0
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

    def resume_operation(self, operation_id: str, *, reason: str = "manual_resume") -> RuntimeState:
        """Normalize in-flight runtime state so the operation can resume safely."""

        state = self.runtime_store.recover_operation(operation_id, reason=reason)
        summary = dict(state.execution.metadata.get("recovery", {}))
        state.operation_status = RuntimeStatus.READY
        state.execution.status = RuntimeStatus.READY
        self._log_operation_event(state, event_type="operation_resumed", **summary)
        self.runtime_store.save_state(state)
        return state.model_copy(deep=True)

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

    def _load_graph_memory(self, operation_id: str) -> tuple[KnowledgeGraph, AttackGraph, RuntimeState | None]:
        """Load KG / AG / Runtime snapshots for operation bootstrap."""

        kg = self.graph_memory_store.load_kg(operation_id)
        ag = self.graph_memory_store.load_ag(operation_id)
        runtime = self.graph_memory_store.load_runtime(operation_id)
        return kg, ag, runtime

    def _ensure_graph_initialized_from_targets(self, state: RuntimeState) -> dict[str, Any] | None:
        """Initialize KG / AG from imported targets when graph memory is empty."""

        kg, ag, _runtime = self._load_graph_memory(state.operation_id)
        if kg.list_nodes() or ag.list_nodes():
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

    def _update_success_condition_progress(
        self,
        *,
        state: RuntimeState,
        stage_result: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate configured success conditions against the latest stage signals.

        Generic, environment-agnostic producer for
        ``metadata["success_condition_progress"]``. Condition names come from the
        active profile's ``success_conditions.require_all`` list; their meaning is
        resolved by deterministic stage-signal detectors (never by an LLM, never
        using raw secrets). The result accumulates across cycles: once a condition
        is satisfied it stays satisfied, and evidence ids are unioned in.

        ``eligible_for_stop`` is computed here and is the only authoritative gate
        that lets PlannerAgent emit ``stop_success``.
        """

        success_conditions = self._mapping(self.settings.load_lab_profile().get("success_conditions"))
        require_all = [str(name) for name in success_conditions.get("require_all", []) if str(name).strip()]
        if not require_all:
            return self._mapping(state.execution.metadata.get("success_condition_progress"))

        progress = self._mapping(state.execution.metadata.get("success_condition_progress"))
        conditions = {k: dict(v) for k, v in self._mapping(progress.get("conditions")).items() if isinstance(v, dict)}
        signals = self._stage_condition_signals(state=state, stage_result=stage_result)

        for name in require_all:
            existing = self._mapping(conditions.get(name))
            satisfied = bool(existing.get("satisfied"))
            evidence_ids = [str(e) for e in existing.get("evidence_ids", []) if str(e).strip()]
            newly = signals.get(name)
            if newly is not None:
                satisfied = True
                for evidence_id in newly:
                    if evidence_id and evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)
            conditions[name] = {"satisfied": satisfied, "evidence_ids": evidence_ids}

        missing = [name for name in require_all if not conditions.get(name, {}).get("satisfied")]
        all_required_satisfied = not missing
        new_progress = {
            "conditions": conditions,
            "missing": missing,
            "all_required_satisfied": all_required_satisfied,
            "eligible_for_stop": all_required_satisfied,
            "recommended_planner_action": "stop_success" if all_required_satisfied else "continue",
        }
        state.execution.metadata["success_condition_progress"] = new_progress
        return new_progress

    def _stage_condition_signals(
        self,
        *,
        state: RuntimeState,
        stage_result: Any | None,
    ) -> dict[str, list[str]]:
        """Map one stage result to the success-condition names it newly satisfies."""

        if stage_result is None:
            return {}

        stage_type = str(getattr(stage_result, "stage_type", "") or "").upper()
        status = str(getattr(stage_result, "status", "") or "").lower()
        findings = getattr(stage_result, "findings", None) or []
        observations = getattr(stage_result, "observations", None) or []
        discovered_entities = getattr(stage_result, "discovered_entities", None) or []
        runtime_hints = self._mapping(getattr(stage_result, "runtime_hints", None))
        evidence_ids = self._stage_evidence_ids(stage_result)
        signal_text = " ".join(
            [
                str(getattr(stage_result, "summary", "") or ""),
                *[
                    str(item.get("type") or item.get("kind") or item.get("category") or item.get("summary") or "")
                    for item in findings
                    if isinstance(item, dict)
                ],
                *[
                    str(item.get("type") or item.get("kind") or item.get("category") or item.get("summary") or "")
                    for item in observations
                    if isinstance(item, dict)
                ],
                *[
                    str(item.get("type") or item.get("kind") or item.get("entity_type") or item.get("summary") or "")
                    for item in discovered_entities
                    if isinstance(item, dict)
                ],
                " ".join(str(key) for key, value in runtime_hints.items() if value),
            ]
        ).lower()
        finding_text = " ".join(
            str(item.get("type") or item.get("kind") or item.get("summary") or "")
            for item in findings
            if isinstance(item, dict)
        ).lower()
        zone = self._stage_zone(stage_result=stage_result, finding_text=signal_text)
        has_pivot_route = self._has_pivot_route(state) or bool(runtime_hints.get("register_pivot_route"))

        is_service_discovery = "recon" in stage_type or "service" in signal_text or "discover" in signal_text
        is_vuln_candidate = "vuln" in stage_type or "candidate" in signal_text or "vulnerab" in signal_text
        is_exploit = "exploit" in stage_type or "exploit" in signal_text
        is_session = bool(runtime_hints.get("session_id") or runtime_hints.get("capability")) or "pivot" in stage_type or "access" in stage_type
        is_controlled_proof = (
            bool(runtime_hints.get("exploit_executed") or runtime_hints.get("validated") or runtime_hints.get("marker_found"))
            or "controlled_proof" in signal_text
            or "validated access" in signal_text
            or "validated_access" in signal_text
        )
        is_post_access = (
            "post_access" in signal_text
            or "post-access" in signal_text
            or "postaccess" in signal_text
            or "artifact observed" in signal_text
            or bool(runtime_hints.get("post_access_observed"))
        )
        is_goal_check = (
            "goal" in stage_type
            or bool(runtime_hints.get("goal_satisfied"))
            or "goalcheck" in signal_text.replace(" ", "")
        )

        signals: dict[str, list[str]] = {}

        def emit(name: str) -> None:
            signals[name] = list(evidence_ids)

        def emit_with_evidence(name: str, ids: list[str] | None = None) -> None:
            clean_name = str(name or "").strip()
            if not clean_name:
                return
            clean_ids = [str(ref) for ref in (ids or evidence_ids) if str(ref).strip()]
            existing = signals.setdefault(clean_name, [])
            for ref in clean_ids:
                if ref not in existing:
                    existing.append(ref)

        generic_conditions = runtime_hints.get("satisfied_conditions") or runtime_hints.get("condition_signals")
        if isinstance(generic_conditions, list):
            for item in generic_conditions:
                if isinstance(item, dict):
                    refs = item.get("evidence_ids") or item.get("evidence_refs")
                    emit_with_evidence(
                        str(item.get("condition") or item.get("name") or ""),
                        [str(ref) for ref in refs] if isinstance(refs, list) else None,
                    )
                else:
                    emit_with_evidence(str(item))
        elif isinstance(generic_conditions, dict):
            for name, refs in generic_conditions.items():
                emit_with_evidence(
                    str(name),
                    [str(ref) for ref in refs] if isinstance(refs, list) else None,
                )

        if is_service_discovery and zone != "restricted":
            emit("dmz_service_discovered")
            emit("entry_zone_service_discovered")
        if is_service_discovery and zone == "restricted" and has_pivot_route:
            emit("internal_service_discovered_after_authorized_route")
            emit("restricted_zone_service_discovered")
        if "recon" in stage_type and ("host" in finding_text or "host" in status):
            emit("entry_zone_host_discovered")
        if "fingerprint" in stage_type or "fingerprint" in finding_text:
            emit("service_fingerprint_recorded")
        if is_vuln_candidate:
            emit("vulnerability_candidate_recorded")
        if is_exploit:
            emit("exploit_attempt_recorded")
            if status in {"succeeded", "success"} or "success" in finding_text:
                emit("exploit_success_recorded")
        if is_controlled_proof:
            emit("at_least_one_controlled_proof_or_validated_access")
        if is_session:
            emit("capability_or_session_recorded")
        if is_post_access:
            emit("post_access_observation_recorded")
        if has_pivot_route:
            emit("pivot_route_recorded")
        if is_goal_check:
            emit("goal_check_recorded")
            if bool(runtime_hints.get("goal_satisfied")):
                emit("final_goal_proof_valid")
        return signals

    @staticmethod
    def _stage_evidence_ids(stage_result: Any) -> list[str]:
        """Resolve evidence ids for a stage: evidence_refs first, then tool raw refs."""

        evidence_ids = [str(ref) for ref in (getattr(stage_result, "evidence_refs", None) or []) if str(ref).strip()]
        if not evidence_ids:
            for trace in getattr(stage_result, "tool_trace", None) or []:
                ref = getattr(trace, "raw_output_ref", None)
                if ref and str(ref).strip():
                    evidence_ids.append(str(ref))
        return evidence_ids

    @staticmethod
    def _stage_zone(*, stage_result: Any, finding_text: str) -> str:
        """Classify the stage's target zone as 'restricted' or 'entry' (generic)."""

        stage_type = str(getattr(stage_result, "stage_type", "") or "").lower()
        runtime_hints = getattr(stage_result, "runtime_hints", None) or {}
        zone_ref = str(
            (runtime_hints.get("zone_ref") if isinstance(runtime_hints, dict) else None) or ""
        ).lower()
        if zone_ref in {"restricted", "internal"}:
            return "restricted"
        if "internal" in stage_type or "restricted" in finding_text or "internal" in finding_text:
            return "restricted"
        return "entry"

    @staticmethod
    def _has_pivot_route(state: RuntimeState) -> bool:
        """Return whether the runtime already recorded at least one pivot route."""

        routes = getattr(state, "pivot_routes", None)
        if routes:
            return True
        metadata_routes = state.execution.metadata.get("pivot_routes")
        return bool(metadata_routes)

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
    def _kg_ag_runtime_snapshot(
        *,
        operation_id: str,
        cycle_index: int,
        graph_refs: list[AgentGraphRef],
        kg: KnowledgeGraph,
        ag: AttackGraph,
        runtime_state: RuntimeState,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        del graph_refs
        policy_context = AppOrchestrator._mapping(extra.get("policy_context") or runtime_state.execution.metadata.get("runtime_policy"))
        current_goal = str(
            extra.get("mission_goal")
            or extra.get("goal")
            or extra.get("objective")
            or runtime_state.execution.summary
            or ""
        )
        snapshot = TwoGraphContextBuilder().build(
            knowledge_graph=kg,
            attack_graph=ag,
            runtime_state=runtime_state,
            policy_context=policy_context,
            current_goal=current_goal,
        )
        snapshot["operation_id"] = operation_id
        snapshot["cycle_index"] = cycle_index
        return snapshot

    def _extract_llm_decision_history(
        self,
        *,
        cycle_index: int,
        cycle: PipelineCycleResult,
    ) -> list[LLMDecisionHistoryRecord]:
        return LLMDecisionObserver(self.settings).extract(cycle_index=cycle_index, cycle=cycle)

    @staticmethod
    def _planner_recent_stage_results(state: RuntimeState, *, limit: int = 8) -> list[dict[str, Any]]:
        """Return compact StageResult records for Planner Agent replanning context."""

        results: list[dict[str, Any]] = []
        for outcome in state.recent_outcomes[-limit:]:
            payload = outcome.metadata.get("outcome_payload")
            if not isinstance(payload, dict):
                continue
            stage_result = payload.get("stage_result")
            if not isinstance(stage_result, dict):
                continue
            results.append(
                {
                    "task_id": outcome.task_id,
                    "outcome_id": outcome.outcome_id,
                    "outcome_type": outcome.outcome_type,
                    "summary": outcome.summary,
                    "payload_ref": outcome.payload_ref,
                    "status": outcome.metadata.get("status"),
                    "stage_result": stage_result,
                    "runtime_hints": payload.get("runtime_hints") if isinstance(payload.get("runtime_hints"), dict) else {},
                    "writeback_hints": payload.get("writeback_hints") if isinstance(payload.get("writeback_hints"), dict) else {},
                    "graph_update_intents": payload.get("graph_update_intents") if isinstance(payload.get("graph_update_intents"), list) else [],
                    "task_candidates": payload.get("task_candidates") if isinstance(payload.get("task_candidates"), list) else [],
                }
            )
        return results

    @staticmethod
    def _blackbox_policy_context(policy_context: dict[str, Any], state: RuntimeState) -> dict[str, Any]:
        targets = [
            dict(item)
            for item in state.execution.metadata.get("target_inventory", [])
            if isinstance(item, dict)
        ]
        risk_policy = AppOrchestrator._mapping(policy_context.get("risk_policy"))
        safe_risk_policy = {
            key: risk_policy[key]
            for key in (
                "max_risk_level",
                "block_destructive",
                "block_file_write",
                "block_reverse_callback",
                "block_command_execution",
                "block_active_exploit",
                "require_approval_for_active_exploit",
            )
            if key in risk_policy
        }
        return {
            "authorized_targets": targets,
            "allow_fingerprint": bool(policy_context.get("allow_fingerprint", True)),
            "allow_safe_probe": bool(policy_context.get("allow_safe_probe", True)),
            "deny_egress": bool(policy_context.get("deny_egress", False)),
            "risk_policy": safe_risk_policy,
            "scope_source": "imported_targets",
        }

    @staticmethod
    def _blackbox_runtime_context(state: RuntimeState) -> dict[str, Any]:
        metadata = state.execution.metadata
        recent_outcomes = [
            {
                "task_id": outcome.task_id,
                "outcome_type": outcome.outcome_type,
                "summary": outcome.summary,
                "payload_ref": outcome.payload_ref,
            }
            for outcome in state.recent_outcomes[-8:]
        ]
        return {
            "operation_id": state.operation_id,
            "operation_status": state.operation_status.value,
            "execution_status": state.execution.status.value,
            "target_inventory": [
                dict(item)
                for item in metadata.get("target_inventory", [])
                if isinstance(item, dict)
            ],
            "cycle_count": len(metadata.get("control_cycle_history", [])),
            "goal_state": AppOrchestrator._mapping(metadata.get("goal_state")),
            "finding_count": len(metadata.get("findings", [])) if isinstance(metadata.get("findings"), list) else 0,
            "evidence_artifact_count": len(metadata.get("evidence_artifacts", []))
            if isinstance(metadata.get("evidence_artifacts"), list)
            else 0,
            "recent_outcomes": recent_outcomes,
            "replan_request_count": len(state.replan_requests),
        }

    @staticmethod
    def _cycle_planner_decision(cycle_result: OperationCycleResult) -> str | None:
        if cycle_result.planner_decision is None:
            return None
        decision = cycle_result.planner_decision.decision
        return str(decision) if decision is not None else None

    def _load_tool_catalog(self) -> dict[str, Any]:
        if self.mcp_client is None:
            catalog = build_default_lab_tool_catalog()
            catalog["pentest-tools"]["available"] = False
            catalog["pentest-tools"]["error"] = "MCP is not configured"
            return catalog
        return self.mcp_client.list_tools() or build_default_lab_tool_catalog()

    @staticmethod
    def _lab_activation_metadata(lab_profile: dict[str, Any]) -> dict[str, Any]:
        """Describe lab activation using main-process visible profile state.

        Do not depend on AEGRA_MCP_TOOLSET here. That variable may only exist in
        the MCP server subprocess environment, while the orchestrator always has
        access to the loaded public lab profile.
        """

        profile_id = str(lab_profile.get("profile_id") or "")
        full_pentest_profiles = {
            "full-vulhub-multihost-pentest",
            "full-chain-autonomous-pentest-lab",
        }
        return {
            "full_pentest_active": profile_id in full_pentest_profiles,
            "profile_id": profile_id or None,
            "source": "lab_profile.profile_id",
        }

    @staticmethod
    def _public_lab_profile_metadata(lab_profile: dict[str, Any]) -> dict[str, Any]:
        """Expose non-topology lab metadata without leaking hidden targets."""

        public: dict[str, Any] = {}
        for key in ("profile_id", "mode"):
            value = lab_profile.get(key)
            if value:
                public[key] = value
        return public

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _success_condition_evidence_ids(progress: dict[str, Any]) -> list[str]:
        conditions = progress.get("conditions")
        if not isinstance(conditions, dict):
            return []
        evidence_ids: set[str] = set()
        for condition in conditions.values():
            if not isinstance(condition, dict):
                continue
            for evidence_id in condition.get("evidence_ids") or []:
                if evidence_id:
                    evidence_ids.add(str(evidence_id))
        return sorted(evidence_ids)

    @staticmethod
    def _result_status(
        *,
        state: RuntimeState,
        success: bool,
        progress: dict[str, Any],
    ) -> str:
        if success:
            return "success"
        if state.operation_status == RuntimeStatus.FAILED:
            return "failed"
        if state.operation_status in {RuntimeStatus.PAUSED, RuntimeStatus.CANCELLED}:
            return "blocked"
        if progress:
            return "partial"
        return "failed" if state.operation_status == RuntimeStatus.COMPLETED else "partial"

    @staticmethod
    def _result_stop_reason(
        *,
        state: RuntimeState,
        success: bool,
        cycle_results: list[OperationCycleResult],
        max_cycles: int | None,
    ) -> str:
        if success:
            return "success_conditions_satisfied"
        if state.operation_status == RuntimeStatus.FAILED:
            return "failed"
        if state.operation_status in {RuntimeStatus.PAUSED, RuntimeStatus.CANCELLED}:
            return "blocked"
        last = cycle_results[-1] if cycle_results else None
        if last is not None and last.stop_reason:
            return str(last.stop_reason)
        if max_cycles is not None and len(cycle_results) >= max_cycles:
            return "max_cycles"
        control = AppOrchestrator._mapping(state.execution.metadata.get("last_control_cycle"))
        if control.get("stop_reason"):
            return str(control["stop_reason"])
        return "blocked"

    @staticmethod
    def _goal_satisfied(state: RuntimeState, cycle_results: list[OperationCycleResult] | None = None) -> bool:
        goal_state = AppOrchestrator._mapping(state.execution.metadata.get("goal_state"))
        if bool(goal_state.get("goal_satisfied")) or str(goal_state.get("status", "")).lower() in {"satisfied", "completed"}:
            return True
        for result in reversed(cycle_results or []):
            if result.stop_reason == "goal_satisfied":
                return True
            decision = AppOrchestrator._cycle_planner_decision(result)
            if decision == "stop_success":
                return True
        control = AppOrchestrator._mapping(state.execution.metadata.get("last_control_cycle"))
        return str(control.get("stop_reason") or "").lower() == "goal_satisfied"

    @staticmethod
    def _next_cycle_index(state: RuntimeState) -> int:
        history = state.execution.metadata.get("control_cycle_history", [])
        return len(history) + 1

    @staticmethod
    def _needs_recovery(state: RuntimeState) -> bool:
        recovery = state.execution.metadata.get("recovery", {})
        return isinstance(recovery, dict) and bool(recovery.get("unclean_shutdown"))

    def _fail_operation_cycle(
        self,
        *,
        operation_id: str,
        cycle_index: int,
        state: RuntimeState,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        phase: str,
        exc: Exception,
        txt_logger: TxtTraceLogger | None = None,
        planner_decision: PlannerDecision | None = None,
        planning_success: bool = False,
        apply_results: list[PhaseTwoApplyResult] | None = None,
    ) -> OperationCycleResult:
        error_type = "llm_transport_error" if "llm_transport_error" in str(exc) else exc.__class__.__name__
        error_record = {
            "cycle_index": cycle_index,
            "phase": phase,
            "error_type": error_type,
            "message": str(exc),
            "retryable": error_type in {"llm_transport_error", "RemoteProtocolError", "ConnectError", "ReadTimeout"},
            "recorded_at": utc_now().isoformat(),
            "graph_write": False,
        }
        prepare_state_for_resume(state, reason=f"{phase}_failed")
        recovery = state.execution.metadata.setdefault("recovery", {})
        if isinstance(recovery, dict):
            recovery["lifecycle_state"] = "recoverable"
            recovery["last_error"] = dict(error_record)
            recovery["active_cycle_id"] = None
            recovery["active_stage_id"] = None
            recovery["runtime_lock"] = "released"
        archived_errors = state.execution.metadata.setdefault("archived_errors", [])
        if isinstance(archived_errors, list):
            archived_errors.append(dict(error_record))
        state.execution.metadata["lifecycle_state"] = "recoverable"
        state.execution.metadata["last_error"] = dict(error_record)
        state.operation_status = RuntimeStatus.PAUSED
        state.execution.status = RuntimeStatus.PAUSED
        state.execution.summary = f"control cycle {cycle_index} failed during {phase}: {error_type}"
        state.last_updated = utc_now()
        self._log_operation_event(
            state,
            event_type="cycle_failed",
            cycle_index=cycle_index,
            phase=phase,
            error_type=error_type,
            retryable=error_record["retryable"],
        )
        self._checkpoint_phase(
            state,
            cycle_index=cycle_index,
            phase="cycle_failed",
            status="failed",
            success=False,
            stopped=True,
            stop_reason=error_type,
            persist=False,
        )
        history = state.execution.metadata.setdefault("control_cycle_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "cycle_index": cycle_index,
                    "started_at": error_record["recorded_at"],
                    "architecture": "planner_stage_agent_graph_main_path",
                    "planning_success": planning_success,
                    "execution_success": False,
                    "selected_agent": None,
                    "selected_stage": None,
                    "applied_results": len(apply_results or []),
                    "stopped": True,
                    "stop_reason": error_type,
                    "cycle_status": "cycle_failed",
                }
            )
        if txt_logger is not None:
            txt_logger.write_block(
                "CYCLE",
                "cycle failed",
                {
                    "cycle": cycle_index,
                    "phase": phase,
                    "error_type": error_type,
                    "retryable": error_record["retryable"],
                    "operation_status": state.operation_status.value,
                },
            )
        self.runtime_store.save_state(state)
        self.graph_memory_store.save_kg(operation_id, kg)
        self.graph_memory_store.save_ag(operation_id, ag)
        self.graph_memory_store.save_runtime(operation_id, state)
        self.graph_memory_store.save_snapshot(operation_id, cycle_index)
        return OperationCycleResult(
            operation_id=operation_id,
            cycle_index=cycle_index,
            planner_decision=planner_decision,
            planning_success=planning_success,
            stage_result=None,
            execution_success=False,
            apply_results=list(apply_results or []),
            selected_task_ids=[],
            applied_task_ids=[],
            stopped=True,
            stop_reason=error_type,
            runtime_state=state.model_copy(deep=True),
        )

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

        return {
            "architecture": "llm_multi_agent_graph_driven_multihost",
            "planner_agent": {
                "implementation": "MissionPlannerAgent",
                "kind": "llm_agent",
                "core_decision_owner": "llm",
            },
            "stage_dispatcher": {
                "implementation": "StageDispatcher",
                "kind": "runtime_router",
            },
            "critic_agent": {
                "implementation": "CriticAgent",
                "kind": "llm_agent",
            },
            "non_agent_services": [
                "ResultApplier",
                "PolicyGate",
                "RuntimeStore",
                "GraphMemoryStore",
                "ExecutionAdapter",
                "Parser",
                "CandidateTaskService",
            ],
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
