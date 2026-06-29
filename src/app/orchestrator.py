"""Top-level orchestration entry point."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.app.settings import AppSettings
from src.core.agents.agent_protocol import GraphRef as AgentGraphRef
from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.configured_mcp_client import ConfiguredMCPClient
from src.core.execution.execution_agent import ExecutionAgent
from src.core.graph.graph_initializer import GraphInitializer
from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.graph.kg_store import KnowledgeGraph
from src.core.evaluation.profile_loader import profile_from_dict
from src.core.evaluation.success_condition_tracker import SuccessConditionTracker
from src.core.evaluation.success_contract_loader import contract_from_dict, load_contract
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState, RuntimeStatus, utc_now
from src.core.models.scope import Asset, Engagement
from src.core.planning.planner import Planner
from src.core.planning.graph_tools import PlannerGraphTools
from src.core.planning.models import PlannerOutcome
from src.core.runtime.audit_report import build_operation_audit_report
from src.core.runtime.observability import (
    append_operation_log,
    mark_clean_shutdown,
    mark_unclean_shutdown,
    prepare_state_for_resume,
    record_phase_checkpoint,
)
from src.core.runtime.report_generator import ReportFormat, ReportGenerator
from src.core.runtime.result_applier import PhaseTwoApplyResult, PhaseTwoResultApplier
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.runtime.store import FileRuntimeStore, InMemoryRuntimeStore, RuntimeStore
from src.integrations.mcp_lab.catalog import build_default_lab_tool_catalog

#把用户输入的目标转换成标准 Asset  用户输入目标 → 标准资产对象 → 写入 runtime policy 的 engagement scope
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
    planner_outcome: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
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

 #主循环
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
        # Tools are called over MCP only; the executor talks to the MCP client
        # directly. Pivot transport is resolved server-side by the mcp_lab tools
        # (route_id in tool arguments / runtime policy), not a client-side adapter.
        self.execution_agent = ExecutionAgent.from_clients(
            llm_client=stage_llm_client,
            mcp_client=self.mcp_client,
            default_timeout_seconds=self.settings.mcp_default_timeout_seconds,
        )
        self.planner = Planner(client=stage_llm_client)

#创建 operation 的初始 RuntimeState，状态为 CREATED
    def create_operation(self, operation_id: str, metadata: dict[str, Any] | None = None) -> RuntimeState:
        """Create a new operation with control-plane metadata attached."""

        runtime_policy = self.settings.load_runtime_policy()
        lab_profile = self.settings.load_lab_profile()
        public_lab_profile = self._public_lab_profile_metadata(lab_profile)
        operation_metadata = {        #某一次 operation 的附加运行上下文信息
            "control_plane": {            #运行控制参数
                # 日志治理参数跟随 operation 一起固化到 runtime metadata，
                # 这样后续 resume、export 和不同 store 后端都能复用同一份配置。
                "audit_max_entries": self.settings.audit_max_entries,
                "operation_log_max_entries": self.settings.operation_log_max_entries,  #operation 普通运行日志最多保留多少条
                "audit_redaction_enabled": self.settings.audit_redaction_enabled,      #是否开启审计脱敏
                "recovery_enabled": self.settings.recovery_enabled,
                "llm_advisors": self._llm_advisor_status(self.settings),
                "agent_architecture": self._agent_architecture_metadata(self.settings),            #当前 agent 架构信息
            },
          
            # operation metadata 中只落稳定 JSON 结构，避免把 Pydantic 对象直接塞进 state。
            "runtime_policy": runtime_policy.to_runtime_metadata(),              #保存运行策略和授权范围
            "lab_profile": public_lab_profile,
            "target_inventory": [],                       #用户导入的目标列表
            "target_count": 0,
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
            runtime_task_count=0,
            worker_count=0,
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
        and execution rounds may contribute evidence and hints, but they do not emit
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
        operation_trace_logger = TxtTraceLogger.operation_trace(operation_id)
        operation_trace_logger.write_block(
            "CYCLE_START",
            "operation cycle started",
            {
                "operation_id": operation_id,
                "cycle_index": cycle_index,
                "policy_gate": "audit_only_non_blocking",
                "graph_write": True,
                "result_applier": "enabled",
                "ag_write": "result_tier_single_step",
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
        operation_trace_logger.write("CYCLE", f"cycle={cycle_index} started")
        self._checkpoint_phase(state, cycle_index=cycle_index, phase="cycle_started", status="running")

        goal = self._mission_goal(planner_payload=planner_payload, context=context)
        policy_context = self._mapping(
            planner_payload.get("policy_context")
            or state.execution.metadata.get("runtime_policy")
        )
        blackbox_policy_context = self._blackbox_policy_context(policy_context, state)
        tool_catalog = self._load_tool_catalog()
        state.execution.metadata["tool_catalog"] = tool_catalog
        operation_trace_logger.write_block(
            "MISSION",
            "mission context",
            {
                "mission_goal": goal,
                "targets": state.execution.metadata.get("target_inventory", []),
                "scope": blackbox_policy_context,
                # The tool catalog (names + schemas) is static and lives in
                # state.execution.metadata["tool_catalog"] for the planner; it is
                # intentionally NOT written to the per-cycle trace to keep it readable.
            },
        )
        #更新成功条件进度
        self._update_success_condition_progress(
            state=state,
            kg=kg,
            ag=ag,
            cycle_index=cycle_index,
        )
        #图查询/摘要工具，构造 Planner 所需的轻量图摘要
        planner_tools = PlannerGraphTools(
            operation_id=operation_id,
            cycle_index=cycle_index,
            kg=kg,
            ag=ag,
            runtime_state=state,
        )
        min_summary = planner_tools.build_min_summary()
        success_progress = self._mapping(state.execution.metadata.get("success_condition_progress"))
        planner_context = {
            "operation_id": operation_id,
            "cycle_index": cycle_index,
            "current_goal": goal,
            "min_summary": min_summary,
            "success_condition_progress": success_progress,
            "graph_tools": PlannerGraphTools.tool_manifest(),
            "agent_capabilities": self.execution_agent.capability_summary(),
            "mcp_tool_catalog": tool_catalog,
        }
        #调用 Planner 做决策
        try:
            outcome = self._planner_decide(
                goal=goal,
                graph_context=planner_context,
                policy_context=blackbox_policy_context,
                recent_execution_results=self._planner_recent_execution_results(state),          #最近几轮 Stage 执行结果
                graph_tools=planner_tools,
            )
            operation_trace_logger.write_block(
                "PLANNER_OUTCOME",
                "planner outcome",
                {
                    "cycle_index": cycle_index,
                    "action": outcome.action,
                    "capability": outcome.directive.capability if outcome.directive else None,
                    "objective": outcome.directive.objective if outcome.directive else goal,
                    "max_tools": outcome.directive.max_tools if outcome.directive else None,
                    "risk_level": outcome.directive.risk_level if outcome.directive else "medium",
                    "confidence": outcome.confidence,
                    "reason": outcome.reason,
                    "target_refs": list(outcome.directive.target_refs if outcome.directive else []),
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
                trace_logger=operation_trace_logger,
            )
        if outcome.operation_id != operation_id:
            original_operation_id = outcome.operation_id
            outcome.operation_id = operation_id
            self._log_operation_event(
                state,
                event_type="planner_outcome_operation_id_normalized",
                cycle_index=cycle_index,
                original_operation_id=original_operation_id,
                normalized_operation_id=operation_id,
            )
            if outcome.directive is not None:
                outcome.directive.operation_id = operation_id
        outcome.cycle_index = cycle_index
        if outcome.directive is not None:
            outcome.directive.cycle_index = cycle_index
        planning_apply = self.result_applier.apply_planner_outcome(outcome, state, kg, ag)

        planner_outcome_dump = outcome.model_dump(mode="json")
        planning_success = outcome.action not in {"stop_failed"}
        apply_results: list[PhaseTwoApplyResult] = [planning_apply]
        execution_result_dump: dict[str, Any] | None = None
        execution_success = True
        applied_ids: list[str] = []
        stopped = outcome.action in {"stop_success", "stop_failed", "pause_for_review"}
        stop_reason = outcome.stop_condition or (outcome.action if stopped else None)

        if outcome.action == "execute" and outcome.directive is not None:
            try:
                execution_result = self.execution_agent.run(
                    outcome.directive,
                    graph_summary={
                        "operation_id": operation_id,
                        "cycle_index": cycle_index,
                        "current_goal": goal,
                        "min_summary": min_summary,
                        "success_condition_progress": success_progress,
                    },
                    graph_history={},
                    runtime_context=self._blackbox_runtime_context(state),
                    policy_context=blackbox_policy_context,
                    mcp_tool_catalog=tool_catalog,
                    pivot_routes=[route.model_dump(mode="json") for route in state.pivot_routes.values()],
                    sessions=[session.model_dump(mode="json") for session in state.sessions.values()],
                )
                if execution_result.operation_id != operation_id:
                    original_operation_id = execution_result.operation_id
                    execution_result.operation_id = operation_id
                    execution_result.runtime_hints = {
                        **dict(execution_result.runtime_hints),
                        "normalized_operation_id_from": original_operation_id,
                    }
                    self._log_operation_event(
                        state,
                        event_type="execution_result_operation_id_normalized",
                        cycle_index=cycle_index,
                        execution_id=execution_result.execution_id,
                        original_operation_id=original_operation_id,
                        normalized_operation_id=operation_id,
                    )
                stage_apply = self.result_applier.apply_execution_result(execution_result, state, kg, ag)
                self._update_success_condition_progress(
                    state=state,
                    kg=kg,
                    ag=ag,
                    cycle_index=cycle_index,
                )
            except Exception as exc:
                return self._fail_operation_cycle(
                    operation_id=operation_id,
                    cycle_index=cycle_index,
                    state=state,
                    kg=kg,
                    ag=ag,
                    phase="execution_round",
                    exc=exc,
                    trace_logger=operation_trace_logger,
                    planning_success=planning_success,
                    apply_results=apply_results,
                )
            stage_graph_write_payload = {
                "cycle": cycle_index,
                "capability": execution_result.capability,
                "agent": execution_result.agent_name,
                "execution_status": execution_result.status,
                "kg_delta_count": len(stage_apply.kg_state_deltas or []),
                "runtime_event_count": len(stage_apply.runtime_event_refs or []),
                "created_ag_nodes": len((stage_apply.ag_graph or {}).get("nodes", [])) if isinstance(stage_apply.ag_graph, dict) else 0,
                "created_ag_edges": len((stage_apply.ag_graph or {}).get("edges", [])) if isinstance(stage_apply.ag_graph, dict) else 0,
                "evidence_refs": list(execution_result.evidence_refs),
            }
            operation_trace_logger.write_block("GRAPH_WRITE", "ResultApplier graph write completed", stage_graph_write_payload)
            apply_results.append(stage_apply)
            applied_ids.append(execution_result.execution_id)
            execution_result_dump = execution_result.model_dump(mode="json")
            execution_success = execution_result.status not in {"failed", "blocked"}
            if self._goal_satisfied(state):
                state.execution.metadata["goal_satisfied"] = True
            if execution_result.status in {"failed", "blocked"}:
                stop_reason = execution_result.summary
                state.execution.metadata["needs_replan"] = True
                state.execution.metadata["last_stage_stop_reason"] = {
                    "status": execution_result.status,
                    "summary": execution_result.summary,
                    "execution_id": execution_result.execution_id,
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
            "feedback_success": True,
            "planner_action": outcome.action,
            "capability": outcome.directive.capability if outcome.directive else None,
            "applied_results": len(apply_results),
            "stopped": stopped,
            "stop_reason": stop_reason,
        }
        history = state.execution.metadata.setdefault("control_cycle_history", [])
        history.append(summary)
        state.execution.metadata["last_control_cycle"] = summary
        state.execution.summary = f"control cycle {cycle_index} completed"
        if outcome.action == "stop_success":
            state.operation_status = RuntimeStatus.COMPLETED
        elif outcome.action == "stop_failed":
            state.operation_status = RuntimeStatus.FAILED
        elif outcome.action == "pause_for_review":
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
        operation_trace_logger.write_block(
            "CYCLE_END",
            "cycle completed",
            {
                "cycle": cycle_index,
                "planner_action": outcome.action,
                "capability": outcome.directive.capability if outcome.directive else None,
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
            planner_outcome=planner_outcome_dump,
            execution_result=execution_result_dump,
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
    
    #多轮运行
    def run_until_quiescent(
        self,
        operation_id: str,
        *,
        graph_refs: list[AgentGraphRef],
        planner_payload: dict[str, Any],
        feedback_payload: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        max_cycles: int = 12,
        max_replans: int = 3,
        consecutive_llm_rejections: int = 3,
        stop_when_quiescent: bool = True,
    ) -> list[OperationCycleResult]:
        """持续运行主循环，直到静止或达到上限。"""

        results: list[OperationCycleResult] = []
        llm_rejection_count = 0
        supervisor_replan_count = 0
        planner_replan_count = 0
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
            elif planner_decision == "execute":
                planner_replan_count = 0
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
            # The initial goal node IS the imported-target goal; tag it so the
            # success contract's target_imported predicate (Goal with
            # category=imported_target) can match it.
            goal_category="imported_target",
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

    def _planner_decide(
        self,
        *,
        goal: str,
        graph_context: dict[str, Any],
        policy_context: dict[str, Any],
        recent_execution_results: list[dict[str, Any]],
        graph_tools: PlannerGraphTools,
    ) -> PlannerOutcome:
        return self.planner.decide(
            goal=goal,
            graph_context=graph_context,
            policy_context=policy_context,
            recent_execution_results=recent_execution_results,
            graph_tools=graph_tools,
        )

    @staticmethod
    def _planner_recent_execution_results(state: RuntimeState, *, limit: int = 8) -> list[dict[str, Any]]:
        """Return compact ExecutionResult records for Planner Agent replanning context."""

        results: list[dict[str, Any]] = []
        for outcome in state.recent_outcomes[-limit:]:
            payload = outcome.metadata.get("outcome_payload")
            if not isinstance(payload, dict):
                continue
            execution_result = payload.get("execution_result")
            if not isinstance(execution_result, dict):
                continue
            results.append(
                {
                    "task_id": outcome.task_id,
                    "outcome_id": outcome.outcome_id,
                    "outcome_type": outcome.outcome_type,
                    "summary": outcome.summary,
                    "payload_ref": outcome.payload_ref,
                    "status": outcome.metadata.get("status"),
                    "execution_result": AppOrchestrator._compact_execution_result(execution_result),
                    "runtime_hints": payload.get("runtime_hints") if isinstance(payload.get("runtime_hints"), dict) else {},
                    "writeback_hints": payload.get("writeback_hints") if isinstance(payload.get("writeback_hints"), dict) else {},
                    "task_candidates": payload.get("task_candidates") if isinstance(payload.get("task_candidates"), list) else [],
                }
            )
        return results

    #raw-fact lists already written to the KG (surfaced to the planner via
    #min_summary); the planner never re-reads them, so collapse to counts.
    _COMPACTED_RESULT_LISTS = frozenset({
        "observations", "evidence", "findings", "discovered_entities",
        "discovered_relations", "capabilities_gained", "credentials", "sessions",
        "pivot_routes",
        "evidence_refs", "tool_trace",
    })

    @staticmethod
    def _compact_execution_result(execution_result: dict[str, Any]) -> dict[str, Any]:
        """Project an ExecutionResult to planner-relevant fields only.

        The planner decides from summaries + control signals (replan/retry
        recommendation, runtime/writeback hints), not raw tool traces. Those
        raw-fact lists are already in the KG and surface via min_summary, so
        collapse only those to counts; keep every scalar, dict, and small
        decision list (e.g. failed_hypotheses).
        """

        compact: dict[str, Any] = {}
        for key, value in execution_result.items():
            if isinstance(value, list) and key in AppOrchestrator._COMPACTED_RESULT_LISTS:
                compact[f"{key}_count"] = len(value)
            else:
                compact[key] = value
        return compact

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
            "blocked_hosts": list(policy_context.get("blocked_hosts") or []),
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
        outcome = cycle_result.planner_outcome
        if not isinstance(outcome, dict):
            return None
        action = outcome.get("action")
        return str(action) if action is not None else None

    def _update_success_condition_progress(
        self,
        *,
        state: RuntimeState,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        cycle_index: int = 0,
    ) -> dict[str, Any] | None:
        """Refresh the deterministic success gate for the next planner turn."""

        lab_profile = self.settings.load_lab_profile()
        success_config = self._mapping(lab_profile.get("success_conditions"))
        contract_ref = lab_profile.get("success_contract_ref")
        if not success_config and not contract_ref:
            return None
        if contract_ref:
            contract = load_contract(str(contract_ref))
            contract_payload = contract.model_dump(mode="json")
        else:
            contract_payload = {
                "contract_id": str(success_config.get("contract_id") or f"{lab_profile.get('profile_id', 'operation')}_inline"),
                "mode": str(success_config.get("mode") or lab_profile.get("mode") or "generic"),
                "require_all": list(success_config.get("require_all") or []),
                "require_chain": list(success_config.get("require_chain") or []),
                "levels": self._mapping(success_config.get("levels")),
                "target_level": success_config.get("target_level"),
                "condition_bindings": self._mapping(success_config.get("condition_bindings")),
            }
        if not contract_payload["condition_bindings"]:
            raise ValueError(
                f"Success contract '{contract_payload.get('contract_id')}' has no "
                "condition_bindings. Success is evaluated solely by reading the graph "
                "through SuccessConditionTracker; a binding-less contract has no "
                "evaluable success definition. Add condition_bindings to the contract."
            )
        profile = profile_from_dict(dict(lab_profile))
        progress = SuccessConditionTracker().evaluate(
            contract=contract_from_dict(dict(contract_payload)),
            profile=profile,
            kg_nodes=list(kg.to_dict().get("nodes") or []),
            kg_edges=list(kg.to_dict().get("edges") or []),
            ag_nodes=list(ag.to_dict().get("nodes") or []),
            ag_edges=list(ag.to_dict().get("edges") or []),
            runtime_state=state.model_dump(mode="json"),
            cycle_index=cycle_index,
        ).model_dump(mode="json")
        progress = self._normalize_success_progress(progress)
        state.execution.metadata["success_condition_progress"] = progress
        return progress

    @staticmethod
    def _normalize_success_progress(progress: dict[str, Any]) -> dict[str, Any]:
        conditions: dict[str, Any] = {}
        for name, result in (progress.get("condition_results") or {}).items():
            if not isinstance(result, dict):
                continue
            refs = list(result.get("evidence_refs") or result.get("evidence_ids") or [])
            conditions[str(name)] = {
                **result,
                "evidence_ids": [str(ref) for ref in refs if str(ref).strip()],
                "evidence_refs": [str(ref) for ref in refs if str(ref).strip()],
            }
        normalized = {
            **progress,
            "conditions": conditions,
            "recommended_planner_action": "stop_success"
            if bool(progress.get("eligible_for_stop"))
            else "continue",
        }
        return normalized

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
        return {
            "full_pentest_active": profile_id == "full-vulhub-multihost-pentest",
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
        trace_logger: TxtTraceLogger | None = None,
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
                    "feedback_success": False,
                    "selected_agent": None,
                    "selected_stage": None,
                    "applied_results": len(apply_results or []),
                    "stopped": True,
                    "stop_reason": error_type,
                    "cycle_status": "cycle_failed",
                }
            )
        if trace_logger is not None:
            trace_logger.write_block(
                "CYCLE_FAILED",
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
            planner_outcome=None,
            execution_result=None,
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
        # The planner advisor is active whenever an LLM client is configured;
        # there are no separate planner/critic/supervisor enable toggles in v3.
        config = settings.to_packy_llm_config()
        return {
            "planner_enabled": config is not None,
            "configured": config is not None,
            "model": config.model if config is not None else None,
            "base_url": config.base_url if config is not None else None,
        }

    @staticmethod
    def _agent_architecture_metadata(settings: AppSettings) -> dict[str, Any]:
        """Stable description of the graph-driven multi-agent runtime."""

        return {
            "architecture": "agentic_planner_single_executor_pev",
            "planner_agent": {
                "implementation": "Planner",
                "kind": "agentic_graph_planner",
                "core_decision_owner": "llm_with_typed_graph_tools",
            },
            "execution_agent": {
                "implementation": "ExecutionAgent",
                "kind": "single_capability_executor",
            },
            "verify_write": {
                "implementation": "SuccessConditionTracker + PhaseTwoResultApplier",
                "kind": "deterministic_verify_and_graph_write",
            },
            "non_agent_services": [
                "ResultApplier",
                "SuccessConditionTracker",
                "RuntimeStore",
                "GraphMemoryStore",
            ],
        }

    @staticmethod
    def _build_runtime_store(settings: AppSettings) -> RuntimeStore:
        if settings.runtime_store_backend == "memory":
            return InMemoryRuntimeStore()
        return FileRuntimeStore(settings.runtime_store_dir)


__all__ = ["AppOrchestrator", "OperationCycleResult", "OperationSummary", "TargetHost"]
