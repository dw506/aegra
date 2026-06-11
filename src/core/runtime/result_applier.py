"""Phase-two result applier for worker-originated execution results.

This module consumes compatibility-layer ``AgentTaskResult`` objects and routes
their side effects through the existing ownership boundaries:

- observations / evidence -> State Writer -> KG state deltas
- fact write requests -> KG structural state deltas
- projection requests -> Graph Projection
- runtime requests / hints -> Runtime managers and runtime event queue

Workers remain unable to write KG / AG stores directly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import EvidenceRecord, ObservationRecord, StateDeltaRecord
from src.core.agents.agent_protocol import (
    AgentContext,
    AgentExecutionResult,
    AgentInput,
    GraphRef,
    GraphScope,
)
from src.core.agents.kg_events import KGDeltaEvent, KGDeltaEventType, KGEventBatch
from src.core.agents.state_writer import KGEntityPatch, KGRelationPatch, StateWriterAgent
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.attack_process import (
    AgentExecutionNode,
    AttackProcessEdge,
    AttackProcessEdgeType,
    AttackProcessNode,
    AttackProcessNodeType,
    PlannerDecisionNode,
    StageResultNode,
    ToolCallNode,
    stable_node_id,
)
from src.core.models.events import (
    AgentResultStatus,
    AgentTaskResult,
    FactWriteKind,
    FactWriteRequest,
    RuntimeControlRequest,
    RuntimeControlType,
)
from src.core.models.finding import EvidenceArtifactRecord, Finding
from src.core.models.kg_enums import EdgeType, NodeType
from src.core.models.runtime import OutcomeCacheEntry, ReplanRequest, RuntimeEventRef, RuntimeState, TaskRuntimeStatus, WorkerStatus
from src.core.models.runtime import utc_now
from src.core.planning.models import PlannerDecision
from src.core.runtime.attack_log_models import AttackLogExtraction
from src.core.runtime.budgets import RuntimeBudgetManager
from src.core.runtime.checkpoint_store import RuntimeCheckpointManager
from src.core.runtime.credential_manager import RuntimeCredentialManager
from src.core.runtime.events import (
    BudgetConsumedEvent,
    CheckpointCreatedEvent,
    LockAcquiredEvent,
    LockReleasedEvent,
    ReplanRequestedEvent,
    SessionExpiredEvent,
    SessionHeartbeatEvent,
    SessionOpenedEvent,
    event_to_ref,
)
from src.core.runtime.lease_manager import RuntimeLeaseManager
from src.core.runtime.locks import RuntimeLockManager
from src.core.runtime.observability import append_audit_log
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager
from src.core.runtime.reachability import ReachabilityPropagator
from src.core.runtime.risk_scoring import RiskScorer
from src.core.runtime.session_manager import RuntimeSessionManager
from src.core.stage.adapters import StageResultAdapter
from src.core.stage.models import StageResult, ToolTrace
from src.core.visualization.graph_event import VisualGraphDelta
from src.core.visualization.graph_serializer import graph_payload_to_delta, runtime_to_delta


class PhaseTwoApplyResult(BaseModel):
    """Structured summary of one phase-two result application pass."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    runtime_event_refs: list[RuntimeEventRef] = Field(default_factory=list)
    kg_state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    ag_state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    kg_apply_result: dict[str, Any] | None = None
    kg_write_diagnostics: dict[str, Any] = Field(default_factory=dict)
    ag_graph: dict[str, Any] | None = None
    kg_event_batch: KGEventBatch | None = None
    state_writer_result: AgentExecutionResult | None = None
    visual_graph_deltas: list[VisualGraphDelta] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)


class PhaseTwoResultApplier:
    """Apply one worker result using existing phase-two owners and managers."""

    def __init__(
        self,
        *,
        state_writer: StateWriterAgent | None = None,
        session_manager: RuntimeSessionManager | None = None,
        credential_manager: RuntimeCredentialManager | None = None,
        lease_manager: RuntimeLeaseManager | None = None,
        pivot_route_manager: RuntimePivotRouteManager | None = None,
        budget_manager: RuntimeBudgetManager | None = None,
        checkpoint_manager: RuntimeCheckpointManager | None = None,
        lock_manager: RuntimeLockManager | None = None,
        reachability_propagator: ReachabilityPropagator | None = None,
    ) -> None:
        self._state_writer = state_writer or StateWriterAgent()
        self._session_manager = session_manager or RuntimeSessionManager()
        self._credential_manager = credential_manager or RuntimeCredentialManager()
        self._lease_manager = lease_manager or RuntimeLeaseManager()
        self._pivot_route_manager = pivot_route_manager or RuntimePivotRouteManager()
        self._reachability_propagator = reachability_propagator or ReachabilityPropagator(self._pivot_route_manager)
        self._budget_manager = budget_manager or RuntimeBudgetManager()
        self._checkpoint_manager = checkpoint_manager or RuntimeCheckpointManager()
        self._lock_manager = lock_manager or RuntimeLockManager()

    def apply(
        self,
        result: AgentTaskResult,
        state: RuntimeState,
        kg_store: KnowledgeGraph | None = None,
        attack_graph: AttackGraph | None = None,
    ) -> PhaseTwoApplyResult:
        """Compatibility entry point for applying canonical AgentTaskResult output."""

        if result.operation_id != state.operation_id:
            raise ValueError("result.operation_id must match RuntimeState.operation_id")

        apply_result = PhaseTwoApplyResult()
        runtime_event_refs = self._apply_runtime_effects(result=result, state=state)
        apply_result.runtime_event_refs.extend(runtime_event_refs)
        self._sync_runtime_views_from_result(state=state, result=result)
        self._apply_stage_runtime_hints(state=state, result=result)
        self._apply_capability_hints(state=state, result=result)
        self._apply_failed_hypotheses(state=state, result=result)
        self._record_recent_outcome(state=state, result=result)
        self._audit_stage_result(state=state, result=result)
        self._audit_tool_invocations(state=state, result=result)
        self._audit_tool_execution_from_result(state=state, result=result)
        self._audit_stage_tool_trace(state=state, result=result)
        self._record_evidence_and_findings(state=state, result=result)
        if kg_store is not None:
            kg_ref = self._resolve_kg_ref(result)
            state_writer_result, _ = self._run_state_writer(
                result=result,
                state=state,
                kg_ref=kg_ref,
            )
            if state_writer_result is not None:
                apply_result.state_writer_result = state_writer_result
                apply_result.kg_state_deltas.extend(list(state_writer_result.state_deltas))
                kg_event_batch = self._kg_events_from_state_deltas(state_writer_result.state_deltas)
                if kg_event_batch.events:
                    apply_result.kg_event_batch = kg_event_batch
                    apply_result.kg_apply_result = kg_store.apply_events(kg_event_batch).model_dump(mode="json")
        if attack_graph is not None:
            apply_result.ag_graph = attack_graph.to_dict()
        apply_result.visual_graph_deltas.extend(
            self._visual_graph_deltas(
                operation_id=state.operation_id,
                state=state,
                kg_store=kg_store,
                ag_graph=apply_result.ag_graph,
                include_kg=kg_store is not None,
                include_runtime=True,
            )
        )
        return apply_result

    def apply_planner_decision(
        self,
        decision: PlannerDecision,
        state: RuntimeState,
        kg_store: KnowledgeGraph,
        attack_graph: AttackGraph,
    ) -> PhaseTwoApplyResult:
        """Record a PlannerDecision in AG and Runtime."""

        del kg_store
        apply_result = PhaseTwoApplyResult()
        cycle_id = self._attack_cycle_id(decision.operation_id, decision.cycle_index)
        decision_id = self._planner_decision_id(decision)
        self._add_ag_node(
            attack_graph,
            AttackProcessNode(
                id=cycle_id,
                node_type=AttackProcessNodeType.ATTACK_CYCLE,
                label=f"Attack cycle {decision.cycle_index}",
                operation_id=decision.operation_id,
                cycle_index=decision.cycle_index,
                status="running",
                summary=f"cycle {decision.cycle_index}",
                properties={
                    "operation_id": decision.operation_id,
                    "node_role": "ATTACK_CYCLE",
                    "display_name": f"Cycle {decision.cycle_index}",
                    "cycle_index": decision.cycle_index,
                    "step_order": 1,
                    "status": "running",
                },
            ),
        )
        self._add_ag_node(
            attack_graph,
            PlannerDecisionNode(
                id=decision_id,
                label=f"Planner decision: {decision.decision}",
                operation_id=decision.operation_id,
                cycle_index=decision.cycle_index,
                agent_name="planner_agent",
                stage_type=decision.selected_stage,
                status=decision.decision,
                summary=decision.reasoning_summary or decision.objective,
                refs=list(decision.target_refs),
                properties={
                    **decision.model_dump(mode="json"),
                    "node_role": "PLANNER_DECISION",
                    "display_name": f"规划决策：{decision.selected_stage or decision.decision}",
                    "cycle_index": decision.cycle_index,
                    "step_order": 2,
                    "selected_stage": decision.selected_stage,
                    "selected_agent": decision.selected_agent,
                    "objective": decision.objective,
                    "reasoning_summary": decision.reasoning_summary,
                    "confidence": decision.confidence,
                },
            ),
        )
        self._add_ag_edge(
            attack_graph,
            AttackProcessEdge(
                id=stable_node_id("edge", {"type": "planned", "source": cycle_id, "target": decision_id}),
                edge_type=AttackProcessEdgeType.PLANNED,
                source=cycle_id,
                target=decision_id,
                label="planned",
            ),
        )
        if decision.decision == "dispatch_agent" and decision.selected_agent:
            execution_id = self._agent_execution_id(
                decision.operation_id,
                decision.cycle_index,
                decision.selected_agent,
            )
            self._add_ag_node(
                attack_graph,
                AgentExecutionNode(
                    id=execution_id,
                    label=f"{decision.selected_agent} execution",
                    operation_id=decision.operation_id,
                    cycle_index=decision.cycle_index,
                    agent_name=decision.selected_agent,
                    stage_type=decision.selected_stage,
                    status="planned",
                    summary=decision.objective,
                    refs=list(decision.target_refs),
                    properties={
                        "planner_decision_id": decision_id,
                        "node_role": "AGENT_EXECUTION",
                        "display_name": f"执行 Agent：{decision.selected_agent}",
                        "cycle_index": decision.cycle_index,
                        "step_order": 3,
                        "selected_stage": decision.selected_stage,
                        "selected_agent": decision.selected_agent,
                    },
                ),
            )
            self._add_ag_edge(
                attack_graph,
                AttackProcessEdge(
                    id=stable_node_id("edge", {"type": "dispatch", "source": decision_id, "target": execution_id}),
                    edge_type=AttackProcessEdgeType.DISPATCHED_TO,
                    source=decision_id,
                    target=execution_id,
                    label="dispatched to",
                ),
            )

        state.execution.metadata["last_planner_decision"] = decision.model_dump(mode="json")
        self._append_audit_log(
            state,
            {
                "event_type": "planner_decision_applied",
                "planner_decision": decision.model_dump(mode="json"),
            },
        )
        apply_result.ag_graph = attack_graph.to_dict()
        apply_result.visual_graph_deltas.extend(
            self._visual_graph_deltas(
                operation_id=state.operation_id,
                state=state,
                kg_store=None,
                ag_graph=apply_result.ag_graph,
                include_kg=False,
                include_runtime=True,
            )
        )
        apply_result.logs.append(f"recorded planner decision {decision.decision}")
        return apply_result

    def apply_stage_result(
        self,
        stage_result: StageResult,
        state: RuntimeState,
        kg_store: KnowledgeGraph,
        attack_graph: AttackGraph,
    ) -> PhaseTwoApplyResult:
        """Apply StageResult effects to KG, AG and Runtime."""

        canonical_result = StageResultAdapter.to_task_result(stage_result)
        if canonical_result.operation_id != state.operation_id:
            raise ValueError("stage_result.operation_id must match RuntimeState.operation_id")

        apply_result = PhaseTwoApplyResult()
        runtime_event_refs = self._apply_runtime_effects(result=canonical_result, state=state)
        apply_result.runtime_event_refs.extend(runtime_event_refs)
        self._sync_runtime_views_from_result(state=state, result=canonical_result)
        self._apply_stage_runtime_hints(state=state, result=canonical_result)
        self._apply_capability_hints(state=state, result=canonical_result)
        self._apply_failed_hypotheses(state=state, result=canonical_result)
        self._record_recent_outcome(state=state, result=canonical_result)
        self._audit_stage_result(state=state, result=canonical_result)
        self._audit_tool_invocations(state=state, result=canonical_result)
        self._audit_tool_execution_from_result(state=state, result=canonical_result)
        self._audit_stage_tool_trace(state=state, result=canonical_result)
        self._record_evidence_and_findings(state=state, result=canonical_result)
        self._record_direct_stage_findings(state=state, stage_result=stage_result)

        kg_ref = self._resolve_kg_ref(canonical_result)
        state_writer_result, state_writer_input = self._run_state_writer(
            result=canonical_result,
            state=state,
            kg_ref=kg_ref,
        )
        if state_writer_result is not None:
            apply_result.state_writer_result = state_writer_result
            apply_result.kg_state_deltas.extend(state_writer_result.output.state_deltas)
            apply_result.logs.extend(state_writer_result.output.logs)

        apply_result.kg_state_deltas.extend(
            self._structured_stage_state_deltas(stage_result=stage_result, result=canonical_result)
        )
        apply_result.kg_state_deltas.extend(self._fact_state_deltas(result=canonical_result))
        if apply_result.kg_state_deltas:
            apply_result.kg_state_deltas = self._order_kg_state_deltas(apply_result.kg_state_deltas)
            # 写图失败与 cycle 失败解耦：apply_patch_batch 已逐条容错，这里再兜一层异常，
            # 把整批写图失败降级为「部分写入 + 诊断」，不让它把整个 operation cycle 拖垮。
            try:
                apply_result.kg_apply_result = self._apply_kg_deltas_to_store(
                    kg_store=kg_store,
                    kg_ref=kg_ref,
                    state=state,
                    state_deltas=apply_result.kg_state_deltas,
                    state_writer_input=state_writer_input,
                )
                apply_result.kg_event_batch = self._kg_batch(state_deltas=apply_result.kg_state_deltas)
            except Exception as exc:  # noqa: BLE001 - 写图绝不阻断推进，失败仅降级记录
                apply_result.kg_write_diagnostics = {
                    "status": "write_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "delta_count": len(apply_result.kg_state_deltas),
                }
                apply_result.logs.append(f"KG write failed (degraded, cycle continues): {exc}")
            else:
                apply_result.kg_write_diagnostics = self._summarize_kg_write(
                    delta_count=len(apply_result.kg_state_deltas),
                    apply_result=apply_result.kg_apply_result,
                )
        else:
            apply_result.kg_write_diagnostics = {
                "status": "no_deltas",
                "reason": self._diagnose_empty_kg_deltas(stage_result=stage_result, result=canonical_result),
                "delta_count": 0,
            }
            apply_result.logs.append(
                f"KG write produced 0 deltas: {apply_result.kg_write_diagnostics['reason']}"
            )

        self._record_stage_result_in_ag(stage_result=stage_result, attack_graph=attack_graph)
        apply_result.ag_graph = attack_graph.to_dict()
        apply_result.visual_graph_deltas.extend(
            self._visual_graph_deltas(
                operation_id=state.operation_id,
                state=state,
                kg_store=kg_store,
                ag_graph=apply_result.ag_graph,
                include_kg=bool(apply_result.kg_apply_result),
                include_runtime=True,
            )
        )
        apply_result.logs.append(f"applied StageResult {stage_result.result_id}")
        return apply_result

    def apply_log_extraction(
        self,
        extraction: AttackLogExtraction,
        state: RuntimeState,
        attack_graph: AttackGraph,
    ) -> PhaseTwoApplyResult:
        """Apply AttackLogExtractor AG node/edge output."""

        apply_result = PhaseTwoApplyResult()
        for payload in extraction.ag_nodes:
            self._add_ag_node(attack_graph, payload)
        for payload in extraction.ag_edges:
            self._add_ag_edge(attack_graph, payload)
        self._append_audit_log(
            state,
            {
                "event_type": "attack_log_extraction_applied",
                "operation_id": extraction.operation_id,
                "cycle_index": extraction.cycle_index,
                "ag_node_count": len(extraction.ag_nodes),
                "ag_edge_count": len(extraction.ag_edges),
                "summary": extraction.summary,
                "evidence_refs": list(extraction.evidence_refs),
            },
        )
        apply_result.ag_graph = attack_graph.to_dict()
        apply_result.visual_graph_deltas.extend(
            self._visual_graph_deltas(
                operation_id=state.operation_id,
                state=state,
                kg_store=None,
                ag_graph=apply_result.ag_graph,
                include_kg=False,
                include_runtime=True,
            )
        )
        return apply_result

    @staticmethod
    def _visual_graph_deltas(
        *,
        operation_id: str,
        state: RuntimeState,
        kg_store: KnowledgeGraph | None,
        ag_graph: dict[str, Any] | None,
        include_kg: bool,
        include_runtime: bool,
    ) -> list[VisualGraphDelta]:
        deltas: list[VisualGraphDelta] = []
        if include_kg and kg_store is not None:
            deltas.append(graph_payload_to_delta(operation_id=operation_id, graph="kg", payload=kg_store.to_dict()))
        if ag_graph is not None:
            deltas.append(graph_payload_to_delta(operation_id=operation_id, graph="ag", payload=ag_graph))
        if include_runtime:
            deltas.append(runtime_to_delta(operation_id=operation_id, runtime_state=state))
        return deltas

    @staticmethod
    def _attack_cycle_id(operation_id: str, cycle_index: int) -> str:
        return f"attack-cycle::{operation_id}::{cycle_index}"

    @staticmethod
    def _planner_decision_id(decision: PlannerDecision) -> str:
        return stable_node_id(
            "planner-decision",
            {
                "operation_id": decision.operation_id,
                "cycle_index": decision.cycle_index,
                "decision": decision.decision,
                "selected_agent": decision.selected_agent,
                "selected_stage": decision.selected_stage,
            },
        )

    @staticmethod
    def _agent_execution_id(operation_id: str, cycle_index: int | None, agent_name: str) -> str:
        return f"agent-execution::{operation_id}::{cycle_index if cycle_index is not None else 'unknown'}::{agent_name}"

    @staticmethod
    def _add_ag_node(attack_graph: AttackGraph, node: Any) -> None:
        from src.core.models.ag import parse_ag_node

        parsed = parse_ag_node(node) if isinstance(node, dict) else node
        if parsed.id in attack_graph._nodes:
            return
        attack_graph.add_node(parsed)

    @staticmethod
    def _add_ag_edge(attack_graph: AttackGraph, edge: Any) -> None:
        from src.core.models.ag import parse_ag_edge

        parsed = parse_ag_edge(edge) if isinstance(edge, dict) else edge
        if parsed.id in attack_graph._edges:
            return
        if parsed.source not in attack_graph._nodes or parsed.target not in attack_graph._nodes:
            return
        attack_graph.add_edge(parsed)

    def _record_stage_result_in_ag(self, *, stage_result: StageResult, attack_graph: AttackGraph) -> None:
        cycle_index = self._stage_result_cycle_index(stage_result)
        result_node_id = f"stage-result::{stage_result.result_id}"
        execution_id = self._agent_execution_id(
            stage_result.operation_id,
            cycle_index,
            stage_result.agent_name,
        )
        self._add_ag_node(
            attack_graph,
            AgentExecutionNode(
                id=execution_id,
                label=f"{stage_result.agent_name} execution",
                operation_id=stage_result.operation_id,
                cycle_index=cycle_index,
                agent_name=stage_result.agent_name,
                stage_type=stage_result.stage_type,
                status=stage_result.status,
                summary=stage_result.summary,
                properties={"stage_task_id": stage_result.stage_task_id},
            ),
        )
        self._add_ag_node(
            attack_graph,
            StageResultNode(
                id=result_node_id,
                label=f"{stage_result.stage_type} result",
                operation_id=stage_result.operation_id,
                cycle_index=cycle_index,
                agent_name=stage_result.agent_name,
                stage_type=stage_result.stage_type,
                status=stage_result.status,
                summary=stage_result.summary,
                evidence_refs=list(stage_result.evidence_refs),
                properties=self._stage_result_process_properties(stage_result),
            ),
        )
        self._add_ag_edge(
            attack_graph,
            AttackProcessEdge(
                id=stable_node_id("edge", {"type": "produced-result", "source": execution_id, "target": result_node_id}),
                edge_type=AttackProcessEdgeType.PRODUCED_RESULT,
                source=execution_id,
                target=result_node_id,
                label="produced result",
            ),
        )
        for trace in stage_result.tool_trace:
            node_id = stable_node_id(
                "tool-call",
                {
                    "operation_id": stage_result.operation_id,
                    "cycle_index": cycle_index,
                    "stage_result_id": stage_result.result_id,
                    "stage_task_id": stage_result.stage_task_id,
                    "step": trace.step,
                    "server_id": trace.server_id,
                    "tool_name": trace.tool_name,
                    "raw_output_ref": trace.raw_output_ref,
                },
            )
            self._add_ag_node(
                attack_graph,
                ToolCallNode(
                    id=node_id,
                    label=trace.tool_name,
                    operation_id=stage_result.operation_id,
                    cycle_index=cycle_index,
                    agent_name=stage_result.agent_name,
                    stage_type=stage_result.stage_type,
                    status="succeeded" if trace.success else "failed",
                    summary=trace.summary or trace.input_summary or trace.tool_name,
                    evidence_refs=list(trace.evidence_refs),
                    properties=self._tool_trace_process_properties(trace),
                ),
            )
            self._add_ag_edge(
                attack_graph,
                AttackProcessEdge(
                    id=stable_node_id("edge", {"type": "called-tool", "source": execution_id, "target": node_id}),
                    edge_type=AttackProcessEdgeType.CALLED_TOOL,
                    source=execution_id,
                    target=node_id,
                    label="called tool",
                ),
            )
        if stage_result.handoff_suggestion is not None:
            handoff_payload = stage_result.handoff_suggestion.model_dump(mode="json")
            handoff_id = stable_node_id(
                "handoff",
                {
                    "stage_result_id": stage_result.result_id,
                    "suggested_agent": stage_result.handoff_suggestion.suggested_agent,
                    "suggested_stage": stage_result.handoff_suggestion.suggested_stage,
                },
            )
            self._add_ag_node(
                attack_graph,
                AttackProcessNode(
                    id=handoff_id,
                    node_type=AttackProcessNodeType.HANDOFF_SUGGESTION,
                    label=f"Handoff to {stage_result.handoff_suggestion.suggested_agent}",
                    operation_id=stage_result.operation_id,
                    cycle_index=cycle_index,
                    agent_name=stage_result.agent_name,
                    stage_type=stage_result.stage_type,
                    status="suggested",
                    summary=stage_result.handoff_suggestion.reason,
                    properties=handoff_payload,
                ),
            )
            self._add_ag_edge(
                attack_graph,
                AttackProcessEdge(
                    id=stable_node_id("edge", {"type": "suggested-handoff", "source": result_node_id, "target": handoff_id}),
                    edge_type=AttackProcessEdgeType.SUGGESTED_HANDOFF,
                    source=result_node_id,
                    target=handoff_id,
                    label="suggested handoff",
                ),
            )

    @staticmethod
    def _stage_result_cycle_index(stage_result: StageResult) -> int:
        raw = stage_result.runtime_hints.get("cycle_index") or stage_result.writeback_hints.get("cycle_index")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _stage_result_process_properties(stage_result: StageResult) -> dict[str, Any]:
        return {
            "result_id": stage_result.result_id,
            "stage_task_id": stage_result.stage_task_id,
            "status": stage_result.status,
            "observation_count": len(stage_result.observations),
            "evidence_count": len(stage_result.evidence),
            "finding_count": len(stage_result.findings),
            "discovered_entity_count": len(stage_result.discovered_entities),
            "discovered_relation_count": len(stage_result.discovered_relations),
            "tool_trace_count": len(stage_result.tool_trace),
            "confidence": stage_result.confidence,
            "risk_level": stage_result.risk_level,
            "policy_notes": list(stage_result.policy_notes),
            "retry_recommendation": stage_result.retry_recommendation,
            "replan_recommendation": stage_result.replan_recommendation,
            "created_at": stage_result.created_at,
        }

    @staticmethod
    def _tool_trace_process_properties(trace: ToolTrace) -> dict[str, Any]:
        return {
            "trace_id": trace.trace_id,
            "step": trace.step,
            "server_id": trace.server_id,
            "tool_name": trace.tool_name,
            "tool_category": trace.tool_category,
            "input_summary": trace.input_summary[:240],
            "raw_output_ref": trace.raw_output_ref,
            "output_summary": (trace.summary or f"stdout {len(trace.stdout or '')} chars; stderr {len(trace.stderr or '')} chars")[:240],
            "stdout_chars": len(trace.stdout or ""),
            "stderr_chars": len(trace.stderr or ""),
            "argument_keys": sorted(str(key) for key in trace.arguments.keys()),
            "success": trace.success,
            "exit_code": trace.exit_code,
            "started_at": trace.started_at,
            "ended_at": trace.ended_at,
            "policy_check": dict(trace.policy_check),
            "metadata": dict(trace.metadata),
        }

    def _record_direct_stage_findings(self, *, state: RuntimeState, stage_result: StageResult) -> None:
        if not stage_result.findings:
            return
        bucket = state.execution.metadata.setdefault("findings", [])
        for index, finding in enumerate(stage_result.findings):
            if not isinstance(finding, dict):
                continue
            finding_id = str(
                finding.get("finding_id")
                or finding.get("id")
                or stable_node_id("finding", {"stage_result_id": stage_result.result_id, "index": index})
            )
            payload = {
                **dict(finding),
                "finding_id": finding_id,
                "operation_id": stage_result.operation_id,
                "stage_result_id": stage_result.result_id,
                "stage_task_id": stage_result.stage_task_id,
                "source_agent": stage_result.agent_name,
                "recorded_at": utc_now().isoformat(),
            }
            existing_index = next(
                (
                    existing_index
                    for existing_index, item in enumerate(bucket)
                    if isinstance(item, dict) and item.get("finding_id") == finding_id
                ),
                None,
            )
            if existing_index is None:
                bucket.append(payload)
            else:
                bucket[existing_index] = payload

    @staticmethod
    def _order_kg_state_deltas(state_deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply KG entity upserts before relations that may reference them."""

        def rank(delta: dict[str, Any]) -> tuple[int, int]:
            delta_type = str(delta.get("delta_type") or "")
            patch_kind = str((delta.get("payload") or {}).get("patch_kind") or "")
            if delta_type == "upsert_entity" or patch_kind == "entity":
                return (0, 0)
            if delta_type == "upsert_relation" or patch_kind == "relation":
                return (2, 0)
            return (1, 0)

        return sorted(list(state_deltas), key=rank)

    @staticmethod
    def _summarize_kg_write(*, delta_count: int, apply_result: dict[str, Any] | None) -> dict[str, Any]:
        """Summarize a KG write outcome, surfacing per-delta failures if any."""

        apply_result = apply_result or {}
        failed = list(apply_result.get("failed_delta_ids") or [])
        applied_entities = list(apply_result.get("applied_entity_ids") or [])
        applied_relations = list(apply_result.get("applied_relation_ids") or [])
        summary: dict[str, Any] = {
            "status": "partial_write" if failed else "ok",
            "delta_count": delta_count,
            "applied_entity_count": len(applied_entities),
            "applied_relation_count": len(applied_relations),
            "failed_delta_count": len(failed),
        }
        if failed:
            summary["failed_delta_ids"] = failed
            summary["errors"] = list(apply_result.get("errors") or [])
        return summary

    def _diagnose_empty_kg_deltas(
        self,
        *,
        stage_result: StageResult,
        result: AgentTaskResult,
    ) -> str:
        """Explain why a stage produced no KG deltas (instead of failing silently)."""

        if not stage_result.tool_trace:
            if result.observations or result.evidence:
                return "observations/evidence present but yielded no writable KG facts"
            return "stage produced no tool calls, observations or evidence"
        successful = [trace for trace in stage_result.tool_trace if getattr(trace, "success", False)]
        if not successful:
            return f"{len(stage_result.tool_trace)} tool call(s) ran but none succeeded"
        with_parsed = [trace for trace in successful if getattr(trace, "parsed_output", None)]
        if not with_parsed:
            return (
                f"{len(successful)} successful tool call(s) but none returned structured "
                "parsed_output to extract KG facts from"
            )
        return "tool calls returned parsed_output but no host/service/finding/evidence shapes matched"

    def _run_state_writer(
        self,
        *,
        result: AgentTaskResult,
        state: RuntimeState,
        kg_ref: GraphRef,
    ) -> tuple[AgentExecutionResult | None, AgentInput | None]:
        observations = [
            ObservationRecord(
                id=record.observation_id,
                source_agent=result.agent_role.value,
                confidence=record.confidence,
                summary=record.summary,
                refs=[self._to_protocol_ref(ref) for ref in record.refs],
                payload=dict(record.payload),
            )
            for record in result.observations
        ]
        evidence = [
            EvidenceRecord(
                id=record.evidence_id,
                source_agent=result.agent_role.value,
                confidence=1.0,
                summary=record.summary,
                refs=[self._to_protocol_ref(ref) for ref in record.refs],
                payload=dict(record.metadata),
                payload_ref=record.payload_ref,
            )
            for record in result.evidence
        ]
        if not observations and not evidence:
            return None, None
        graph_refs = [] if kg_ref.ref_id == "kg-root" else [kg_ref]
        agent_input = AgentInput(
            graph_refs=graph_refs,
            task_ref=result.task_id,
            context=AgentContext(
                operation_id=state.operation_id,
                runtime_state_ref=state.operation_id,
            ),
            raw_payload={
                "observations": [item.model_dump(mode="json") for item in observations],
                "evidences": [item.model_dump(mode="json") for item in evidence],
            },
        )
        return self._state_writer.run(agent_input), agent_input

    def _apply_kg_deltas_to_store(
        self,
        *,
        kg_store: KnowledgeGraph,
        kg_ref: GraphRef,
        state: RuntimeState,
        state_deltas: list[dict[str, Any]],
        state_writer_input: AgentInput | None,
    ) -> dict[str, Any]:
        agent_input = state_writer_input or AgentInput(
            graph_refs=[kg_ref],
            context=AgentContext(
                operation_id=state.operation_id,
                runtime_state_ref=state.operation_id,
            ),
            raw_payload={"kg_version": kg_store.version},
        )
        apply_request = self._state_writer.build_store_apply_request(
            kg_ref=kg_ref,
            state_deltas=state_deltas,
            agent_input=agent_input,
            base_kg_version=kg_store.version,
        )
        return self._state_writer.apply_to_store(store=kg_store, apply_request=apply_request)

    def _apply_runtime_effects(
        self,
        *,
        result: AgentTaskResult,
        state: RuntimeState,
    ) -> list[RuntimeEventRef]:
        refs: list[RuntimeEventRef] = []
        for request in result.runtime_requests:
            refs.extend(self._apply_runtime_request(state=state, request=request))
        for hint in result.checkpoint_hints:
            checkpoint = self._checkpoint_manager.create_checkpoint(
                state=state,
                checkpoint_id=hint.checkpoint_id,
                created_after_tasks=list(hint.created_after_tasks or [result.task_id]),
                summary=hint.summary,
            )
            refs.append(
                self._push_runtime_event(
                    state,
                    CheckpointCreatedEvent(
                        operation_id=state.operation_id,
                        checkpoint_id=checkpoint.checkpoint_id,
                        created_after_tasks=list(checkpoint.created_after_tasks),
                        summary=checkpoint.summary,
                    ),
                )
            )
        for hint in result.replan_hints:
            self._checkpoint_manager.add_replan_marker(
                state,
                {
                    "source_task_id": hint.source_task_id,
                    "scope": hint.scope.value,
                    "reason": hint.reason,
                    "task_ids": list(hint.task_ids),
                    "invalidated_ref_keys": list(hint.invalidated_ref_keys),
                    "metadata": dict(hint.metadata),
                },
            )
            request = ReplanRequest(
                request_id=f"replan-hint-{hint.hint_id}",
                reason=hint.reason,
                task_ids=list(hint.task_ids or [hint.source_task_id]),
                scope=hint.scope.value,
                metadata={
                    "hint_id": hint.hint_id,
                    "invalidated_ref_keys": list(hint.invalidated_ref_keys),
                    **dict(hint.metadata),
                },
            )
            state.request_replan(request)
            refs.append(
                self._push_runtime_event(
                    state,
                    ReplanRequestedEvent(
                        operation_id=state.operation_id,
                        request_id=request.request_id,
                        reason=request.reason,
                        task_ids=list(request.task_ids),
                        scope=request.scope,
                        payload={"metadata": dict(request.metadata)},
                    ),
                )
            )
        if result.critic_signals:
            bucket = state.execution.metadata.setdefault("critic_signals", [])
            bucket.extend(signal.model_dump(mode="json") for signal in result.critic_signals)
        return refs

    def _apply_runtime_request(
        self,
        *,
        state: RuntimeState,
        request: RuntimeControlRequest,
    ) -> list[RuntimeEventRef]:
        if request.request_type == RuntimeControlType.OPEN_SESSION:
            session_id = request.session_id or f"session-{request.request_id}"
            session = self._session_manager.open_session(
                state=state,
                session_id=session_id,
                bound_identity=self._string(request.metadata.get("bound_identity")),
                bound_target=self._string(request.metadata.get("bound_target")),
                lease_seconds=int(request.lease_seconds or 300),
                reusability=str(request.reuse_policy or "exclusive"),
            )
            self._session_manager.bind_task_to_session(state, request.source_task_id, session.session_id)
            lease_id = self._lease_id(session.session_id, request.source_task_id)
            lease = self._lease_manager.create_lease(
                state=state,
                lease_id=lease_id,
                session_id=session.session_id,
                owner_task_id=request.source_task_id,
                lease_seconds=int(request.lease_seconds or 300),
                reuse_policy=str(request.reuse_policy or "exclusive"),
                metadata={"created_by": "result_applier", "request_id": request.request_id},
            )
            self._lease_manager.bind_lease_to_task_or_session(
                state,
                lease.lease_id,
                task_id=request.source_task_id,
                session_id=session.session_id,
            )
            execution_endpoint = self._mapping(request.metadata.get("execution_endpoint"))
            if execution_endpoint:
                session.metadata["execution_endpoint"] = execution_endpoint
            capabilities = self._mapping(request.metadata.get("capabilities"))
            if capabilities:
                session.metadata["capabilities"] = capabilities
            return [
                self._push_runtime_event(
                    state,
                    SessionOpenedEvent(
                        operation_id=state.operation_id,
                        session_id=session.session_id,
                        bound_identity=session.bound_identity,
                        bound_target=session.bound_target,
                        lease_expiry=session.lease_expiry,
                        reusability=session.reusability,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.REGISTER_PIVOT_ROUTE:
            route_id = self._string(request.metadata.get("route_id")) or f"route-{request.request_id}"
            destination_host = self._string(request.metadata.get("destination_host") or request.metadata.get("target_host"))
            if destination_host is None:
                raise ValueError("register_pivot_route runtime request requires metadata.destination_host")
            route = self._pivot_route_manager.register_candidate(
                state,
                route_id,
                destination_host,
                source_host=self._string(request.metadata.get("source_host")),
                via_host=self._string(request.metadata.get("via_host")),
                session_id=request.session_id or self._string(request.metadata.get("session_id")),
                protocol=self._string(request.metadata.get("protocol")),
                destination_zone=self._string(request.metadata.get("destination_zone")),
                destination_cidr=self._string(request.metadata.get("destination_cidr")),
                allowed_ports=self._list_or_set(request.metadata.get("allowed_ports") or request.metadata.get("port")),
                protocols=self._list_or_set(request.metadata.get("protocols")),
                hop_count=self._int(request.metadata.get("hop_count")),
                confidence=self._float(request.metadata.get("confidence")),
                metadata={
                    "created_by": "result_applier",
                    "request_id": request.request_id,
                    "transport": self._mapping(request.metadata.get("transport")),
                },
            )
            if bool(request.metadata.get("active", False)):
                route = self._pivot_route_manager.activate_route(state, route.route_id)
            return [self._runtime_control_event(state, request, summary=f"registered pivot route {route.route_id}")]
        if request.request_type == RuntimeControlType.VERIFY_PIVOT_ROUTE:
            route_id = self._string(request.metadata.get("route_id")) or self._string(request.session_id)
            if route_id is None:
                raise ValueError("verify_pivot_route runtime request requires metadata.route_id")
            reachable = bool(request.metadata.get("reachable", True))
            if reachable:
                self._pivot_route_manager.activate_route(state, route_id)
            else:
                self._pivot_route_manager.fail_route(state, route_id, reason=request.reason or "pivot_verification_failed")
            return [self._runtime_control_event(state, request, summary=f"verified pivot route {route_id}")]
        if request.request_type == RuntimeControlType.OPEN_TUNNEL:
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                route = state.pivot_routes[route_id]
                transport = dict(route.metadata.get("transport", {})) if isinstance(route.metadata.get("transport"), dict) else {}
                transport.update(
                    {
                        "kind": "tcp_tunnel",
                        "tunnel_endpoint": self._string(request.metadata.get("tunnel_endpoint") or request.metadata.get("endpoint")),
                        "health": "ready",
                    }
                )
                route.metadata["transport"] = {key: value for key, value in transport.items() if value is not None}
            session_id = request.session_id or self._string(request.metadata.get("session_id"))
            if session_id is not None and session_id in state.sessions:
                endpoint = dict(state.sessions[session_id].metadata.get("execution_endpoint", {})) if isinstance(state.sessions[session_id].metadata.get("execution_endpoint"), dict) else {}
                endpoint.update({"kind": "tunnel", "adapter": "tcp_tunnel", "tunnel_endpoint": self._string(request.metadata.get("tunnel_endpoint") or request.metadata.get("endpoint"))})
                state.sessions[session_id].metadata["execution_endpoint"] = {key: value for key, value in endpoint.items() if value is not None}
            return [self._runtime_control_event(state, request, summary="opened tunnel")]
        if request.request_type == RuntimeControlType.CLOSE_TUNNEL:
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                route = state.pivot_routes[route_id]
                transport = dict(route.metadata.get("transport", {})) if isinstance(route.metadata.get("transport"), dict) else {}
                transport["health"] = "closed"
                route.metadata["transport"] = transport
            return [self._runtime_control_event(state, request, summary="closed tunnel")]
        if request.request_type == RuntimeControlType.ATTACH_NETWORK_NAMESPACE:
            namespace = self._string(request.metadata.get("network_namespace") or request.metadata.get("namespace"))
            if namespace is None:
                raise ValueError("attach_network_namespace runtime request requires metadata.network_namespace")
            for session_id in [request.session_id, self._string(request.metadata.get("session_id"))]:
                if session_id is not None and session_id in state.sessions:
                    endpoint = dict(state.sessions[session_id].metadata.get("execution_endpoint", {})) if isinstance(state.sessions[session_id].metadata.get("execution_endpoint"), dict) else {}
                    endpoint.update({"adapter": "netns_shell", "namespace": namespace})
                    state.sessions[session_id].metadata["execution_endpoint"] = endpoint
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                transport = dict(state.pivot_routes[route_id].metadata.get("transport", {})) if isinstance(state.pivot_routes[route_id].metadata.get("transport"), dict) else {}
                transport.update({"kind": "netns", "namespace": namespace, "health": "ready"})
                state.pivot_routes[route_id].metadata["transport"] = transport
            return [self._runtime_control_event(state, request, summary=f"attached network namespace {namespace}")]
        if request.request_type == RuntimeControlType.DETACH_NETWORK_NAMESPACE:
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                transport = dict(state.pivot_routes[route_id].metadata.get("transport", {})) if isinstance(state.pivot_routes[route_id].metadata.get("transport"), dict) else {}
                transport.pop("namespace", None)
                if transport.get("kind") == "netns":
                    transport["health"] = "closed"
                state.pivot_routes[route_id].metadata["transport"] = transport
            return [self._runtime_control_event(state, request, summary="detached network namespace")]
        if request.request_type == RuntimeControlType.EXTEND_SESSION:
            if not request.session_id:
                raise ValueError("extend_session runtime request requires session_id")
            session = self._session_manager.extend_lease(
                state=state,
                session_id=request.session_id,
                extra_seconds=int(request.lease_seconds or 60),
            )
            for lease in self._lease_manager.list_leases_for_session(state, request.session_id, active_only=True):
                self._lease_manager.extend_lease(
                    state,
                    lease.lease_id,
                    extra_seconds=int(request.lease_seconds or 60),
                )
            return [
                self._push_runtime_event(
                    state,
                    SessionHeartbeatEvent(
                        operation_id=state.operation_id,
                        session_id=session.session_id,
                        heartbeat_at=session.heartbeat_at,
                        lease_expiry=session.lease_expiry,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.EXPIRE_SESSION:
            if not request.session_id:
                raise ValueError("expire_session runtime request requires session_id")
            session = self._session_manager.expire_session(
                state=state,
                session_id=request.session_id,
                reason=request.reason,
            )
            self._cleanup_session_family(
                state,
                session_id=session.session_id,
                reason=request.reason or "runtime_request_expire_session",
                close_routes=False,
            )
            return [
                self._push_runtime_event(
                    state,
                    SessionExpiredEvent(
                        operation_id=state.operation_id,
                        session_id=session.session_id,
                        reason=request.reason,
                        failure_count=session.failure_count,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.ACQUIRE_LOCKS:
            owner_type = str(request.metadata.get("owner_type", "task"))
            owner_id = self._string(request.metadata.get("owner_id")) or request.source_task_id
            results = self._lock_manager.acquire_many(
                state=state,
                lock_keys=list(request.lock_keys),
                owner_type=owner_type,
                owner_id=owner_id,
                ttl_seconds=request.lease_seconds,
            )
            return [
                self._push_runtime_event(
                    state,
                    LockAcquiredEvent(
                        operation_id=state.operation_id,
                        lock_key=item.lock_key,
                        owner_type=item.owner_type,
                        owner_id=item.owner_id,
                        expires_at=item.expires_at,
                    ),
                )
                for item in results
            ]
        if request.request_type == RuntimeControlType.RELEASE_LOCKS:
            owner_id = self._string(request.metadata.get("owner_id"))
            refs: list[RuntimeEventRef] = []
            for lock_key in request.lock_keys:
                if self._lock_manager.release_lock(state=state, lock_key=lock_key, owner_id=owner_id):
                    refs.append(
                        self._push_runtime_event(
                            state,
                            LockReleasedEvent(
                                operation_id=state.operation_id,
                                lock_key=lock_key,
                                owner_id=owner_id or request.source_task_id,
                                reason=request.reason,
                            ),
                        )
                    )
            return refs
        if request.request_type == RuntimeControlType.CONSUME_BUDGET:
            delta = request.budget_delta
            if delta.time_sec:
                self._budget_manager.consume_time(state, delta.time_sec)
            if delta.tokens:
                self._budget_manager.consume_tokens(state, delta.tokens)
            if delta.operations:
                self._budget_manager.consume_operations(state, delta.operations)
            if delta.noise:
                self._budget_manager.consume_noise(state, delta.noise)
            if delta.risk:
                self._budget_manager.consume_risk(state, delta.risk)
            approval_updates = {
                str(key): bool(value)
                for key, value in dict(request.metadata.get("approval_updates", {})).items()
            }
            for key, value in approval_updates.items():
                self._budget_manager.set_approval(state, key, value)
            policy_updates = {
                str(key): value
                for key, value in dict(request.metadata.get("policy_flag_updates", {})).items()
            }
            for key, value in policy_updates.items():
                self._budget_manager.set_policy_flag(state, key, value)
            return [
                self._push_runtime_event(
                    state,
                    BudgetConsumedEvent(
                        operation_id=state.operation_id,
                        time_budget_used_sec_delta=delta.time_sec,
                        token_budget_used_delta=delta.tokens,
                        operation_budget_used_delta=delta.operations,
                        noise_budget_used_delta=delta.noise,
                        risk_budget_used_delta=delta.risk,
                        approval_updates=approval_updates,
                        policy_flag_updates=policy_updates,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.CREATE_CHECKPOINT:
            checkpoint_id = request.checkpoint_id or f"checkpoint-{request.request_id}"
            checkpoint = self._checkpoint_manager.create_checkpoint(
                state=state,
                checkpoint_id=checkpoint_id,
                created_after_tasks=[request.source_task_id],
                summary=request.reason,
            )
            return [
                self._push_runtime_event(
                    state,
                    CheckpointCreatedEvent(
                        operation_id=state.operation_id,
                        checkpoint_id=checkpoint.checkpoint_id,
                        created_after_tasks=list(checkpoint.created_after_tasks),
                        summary=checkpoint.summary,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.REQUEST_REPLAN:
            replan_request = ReplanRequest(
                request_id=f"runtime-replan-{request.request_id}",
                reason=request.reason or "runtime control requested replanning",
                task_ids=[request.source_task_id],
                scope=str(request.metadata.get("scope", "local")),
                metadata=dict(request.metadata),
            )
            state.request_replan(replan_request)
            self._checkpoint_manager.add_replan_marker(
                state,
                {
                    "request_id": replan_request.request_id,
                    "reason": replan_request.reason,
                    "scope": replan_request.scope,
                    "task_ids": list(replan_request.task_ids),
                },
            )
            return [
                self._push_runtime_event(
                    state,
                    ReplanRequestedEvent(
                        operation_id=state.operation_id,
                        request_id=replan_request.request_id,
                        reason=replan_request.reason,
                        task_ids=list(replan_request.task_ids),
                        scope=replan_request.scope,
                        payload={"metadata": dict(replan_request.metadata)},
                    ),
                )
            ]
        raise ValueError(f"unsupported runtime request type: {request.request_type.value}")

    # 中文注释：
    # worker 只给出结果和线索，真正写回 RuntimeState 的生命周期转移统一在这里完成。
    def _apply_task_lifecycle(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        task = state.execution.tasks.get(result.task_id)
        if task is None:
            return
        now = utc_now()
        runtime_setup_block = self._is_runtime_setup_block(result)
        if not runtime_setup_block:
            task.attempt_count = min(task.attempt_count + 1, task.max_attempts)
        task.last_outcome_ref = f"runtime://results/{result.result_id}"
        task.metadata["last_result_status"] = result.status.value
        task.metadata["last_result_summary"] = result.summary
        task.metadata["last_result_at"] = now.isoformat()
        if result.error_message is not None:
            task.last_error = result.error_message
        elif result.status.value in {"failed", "blocked", "needs_replan"}:
            task.last_error = result.summary
        else:
            task.last_error = None

        if runtime_setup_block:
            task.status = TaskRuntimeStatus.PENDING
            task.finished_at = None
            task.started_at = None
            task.deadline = None
            task.metadata["runtime_blocked_reason"] = self._string(result.outcome_payload.get("blocked_on")) or result.summary
        elif result.status.value == "succeeded":
            task.status = TaskRuntimeStatus.SUCCEEDED
            task.finished_at = now
            task.metadata.pop("runtime_blocked_reason", None)
        elif result.status.value == "noop":
            task.status = TaskRuntimeStatus.SKIPPED
            task.finished_at = now
            task.metadata.pop("runtime_blocked_reason", None)
        elif result.status.value == "needs_replan":
            task.status = TaskRuntimeStatus.BLOCKED
            task.finished_at = now
            task.metadata["requires_replan"] = True
        elif result.status.value == "blocked":
            task.status = TaskRuntimeStatus.BLOCKED
            task.finished_at = now
            task.metadata["runtime_blocked_reason"] = self._string(result.outcome_payload.get("blocked_on")) or result.summary
        else:
            task.status = TaskRuntimeStatus.FAILED
            task.finished_at = now

        self._release_task_execution_resources(state=state, task_id=result.task_id)
        if runtime_setup_block:
            return

        lease_cleanup_reason = result.error_message or result.summary or f"task_{task.status.value}"
        self._lease_manager.release_leases_for_task(state, result.task_id, reason=lease_cleanup_reason)
        session_id = self._result_session_id(result=result, task=task)
        if task.status == TaskRuntimeStatus.SUCCEEDED:
            self._close_single_use_session(state=state, session_id=session_id, task_id=result.task_id)
            return
        if task.status in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.TIMED_OUT, TaskRuntimeStatus.BLOCKED}:
            if session_id is not None and (
                task.status == TaskRuntimeStatus.FAILED
                or result.status.value == "needs_replan"
                or self._string(result.outcome_payload.get("blocked_on")) == "reachability"
            ):
                self._session_manager.fail_session(
                    state,
                    session_id,
                    reason=result.error_message or result.summary,
                )
                self._cleanup_session_family(
                    state,
                    session_id=session_id,
                    reason=result.error_message or result.summary or f"task_{task.status.value}",
                    close_routes=False,
                )

    def _sync_runtime_views_from_result(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        self._sync_credential_view(state=state, result=result)
        self._apply_credential_hints(state=state, result=result)
        self._apply_session_hints(state=state, result=result)
        self._apply_pivot_route_hints(state=state, result=result)
        self._sync_pivot_route_view(state=state, result=result)

    def _apply_capability_hints(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        capabilities = self._stage_result_items(result, "capabilities_gained")
        if not capabilities:
            metadata_items = result.metadata.get("capabilities_gained")
            if isinstance(metadata_items, list):
                capabilities = [dict(item) for item in metadata_items if isinstance(item, dict)]
        if not capabilities:
            return
        bucket = state.execution.metadata.setdefault("capabilities", [])
        existing = {
            str(item.get("capability_id") or item.get("id")): index
            for index, item in enumerate(bucket)
            if isinstance(item, dict) and (item.get("capability_id") or item.get("id"))
        }
        for capability in capabilities:
            payload = {
                **dict(capability),
                "source_task_id": capability.get("source_task_id") or result.task_id,
                "worker_result_id": result.result_id,
            }
            key = self._string(payload.get("capability_id") or payload.get("id"))
            if key is None:
                key = f"capability::{result.task_id}::{len(bucket)}"
                payload["capability_id"] = key
            index = existing.get(key)
            if index is None:
                existing[key] = len(bucket)
                bucket.append(payload)
            else:
                bucket[index] = payload

    def _apply_stage_runtime_hints(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for hints in self._runtime_hints(result):
            if "goal_satisfied" in hints:
                goal_state = state.execution.metadata.setdefault("goal_state", {})
                goal_satisfied = bool(hints.get("goal_satisfied"))
                goal_summary = self._string(hints.get("goal_summary")) or result.summary
                goal_evidence_refs = [
                    str(item) for item in hints.get("goal_evidence_refs", []) if item is not None
                ] if isinstance(hints.get("goal_evidence_refs"), list) else []
                goal_state["goal_satisfied"] = goal_satisfied
                goal_state["goal_summary"] = goal_summary
                goal_state["goal_evidence_refs"] = goal_evidence_refs
                goal_state["source_task_id"] = result.task_id
                goal_state["updated_at"] = utc_now().isoformat()
                state.execution.metadata["goal_satisfied"] = goal_satisfied
                state.execution.metadata["goal_summary"] = goal_summary
                state.execution.metadata["goal_evidence_refs"] = goal_evidence_refs
            if hints.get("active_sessions") is not None:
                state.execution.metadata["active_sessions"] = hints.get("active_sessions")
            if hints.get("pivot_routes") is not None:
                state.execution.metadata["pivot_routes"] = hints.get("pivot_routes")

    def _apply_failed_hypotheses(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        failed = self._stage_result_items(result, "failed_hypotheses")
        if not failed:
            metadata_items = result.metadata.get("failed_hypotheses")
            if isinstance(metadata_items, list):
                failed = [dict(item) for item in metadata_items if isinstance(item, dict)]
        if not failed:
            return
        bucket = state.execution.metadata.setdefault("failed_hypotheses", [])
        for item in failed:
            bucket.append(
                {
                    **dict(item),
                    "source_task_id": item.get("source_task_id") or result.task_id,
                    "worker_result_id": result.result_id,
                    "recorded_at": utc_now().isoformat(),
                }
            )

    def _record_evidence_and_findings(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        normalized_evidence = self._normalize_evidence_records(result)
        if normalized_evidence:
            bucket = state.execution.metadata.setdefault("evidence_artifacts", [])
            existing_indexes = {
                str(item.get("evidence_id")): index
                for index, item in enumerate(bucket)
                if isinstance(item, dict) and item.get("evidence_id")
            }
            for evidence in normalized_evidence:
                payload = evidence.model_dump(mode="json")
                index = existing_indexes.get(evidence.evidence_id)
                if index is None:
                    existing_indexes[evidence.evidence_id] = len(bucket)
                    bucket.append(payload)
                else:
                    bucket[index] = payload

        validation = self._dict(result.outcome_payload.get("validation"))
        if not validation and result.outcome_payload.get("outcome_type") == "vulnerability_validation":
            validation = dict(result.outcome_payload)
        status = self._string(validation.get("status"))
        if status == "blocked":
            self._record_finding_audit(
                state=state,
                result=result,
                event_type="validation_blocked",
                validation=validation,
                reason=self._string(validation.get("failure_reason")) or result.error_message or result.summary,
            )
            return
        if status == "not_detected":
            return
        if status not in {"validated", "suspected"}:
            return

        finding = self._finding_from_validation(
            state=state,
            result=result,
            validation=validation,
            evidence_refs=[item.evidence_id for item in normalized_evidence],
        )
        bucket = state.execution.metadata.setdefault("findings", [])
        existing_index = next(
            (
                index
                for index, item in enumerate(bucket)
                if isinstance(item, dict) and item.get("finding_id") == finding.finding_id
            ),
            None,
        )
        payload = finding.model_dump(mode="json")
        if existing_index is None:
            bucket.append(payload)
            event_type = "finding_created"
        else:
            original_created_at = bucket[existing_index].get("created_at") if isinstance(bucket[existing_index], dict) else None
            if original_created_at:
                payload["created_at"] = original_created_at
            bucket[existing_index] = payload
            event_type = "finding_updated"
        self._record_finding_audit(
            state=state,
            result=result,
            event_type=event_type,
            validation=validation,
            finding_id=finding.finding_id,
            evidence_refs=list(finding.evidence_refs),
        )

    def _normalize_evidence_records(self, result: AgentTaskResult) -> list[EvidenceArtifactRecord]:
        records: list[EvidenceArtifactRecord] = []
        for evidence in result.evidence:
            records.append(
                EvidenceArtifactRecord(
                    evidence_id=evidence.evidence_id,
                    kind=evidence.kind,
                    summary=evidence.summary,
                    payload_ref=evidence.payload_ref,
                    task_ref=result.task_id,
                    tool_output_ref=evidence.tool_output_ref or evidence.payload_ref,
                    refs=[ref.model_dump(mode="json") for ref in evidence.refs],
                    metadata={
                        **dict(evidence.metadata),
                        "operation_id": result.operation_id,
                        "worker_result_id": result.result_id,
                        "source_agent": result.agent_role.value,
                        "execution_node_id": result.execution_node_id,
                        "timestamp": evidence.created_at.isoformat(),
                    },
                    created_at=evidence.created_at,
                )
            )
        return records

    def _finding_from_validation(
        self,
        *,
        state: RuntimeState,
        result: AgentTaskResult,
        validation: dict[str, Any],
        evidence_refs: list[str],
    ) -> Finding:
        service_ref = self._string(result.outcome_payload.get("service_id")) or self._ref_id_for_type(result, "Service") or "unknown-service"
        vulnerability_ref = (
            self._string(result.outcome_payload.get("vulnerability_id"))
            or self._string(validation.get("vulnerability_id"))
            or self._ref_id_for_type(result, "Vulnerability")
            or "unknown-vulnerability"
        )
        title = self._string(validation.get("vulnerability_name")) or self._string(validation.get("summary")) or vulnerability_ref
        cvss = self._coerce_float(validation.get("cvss"))
        epss = self._coerce_float(validation.get("epss"))
        kev = bool(validation.get("kev", False))
        confidence = self._coerce_float(result.outcome_payload.get("confidence")) or self._coerce_float(validation.get("confidence")) or 0.5
        requires_auth = bool(
            validation.get("requires_auth")
            or validation.get("requires_authentication")
            or result.outcome_payload.get("requires_auth")
        )
        public_exposed = self._is_publicly_exposed(state=state, service_ref=service_ref, validation=validation)
        critical_asset = self._is_critical_asset(state=state, service_ref=service_ref, validation=validation)
        validated = str(validation.get("status")) == "validated"
        risk_score = RiskScorer.score(
            cvss=cvss,
            epss=epss,
            kev=kev,
            public_exposed=public_exposed,
            validated=validated,
            requires_auth=requires_auth,
            critical_asset=critical_asset,
            confidence=confidence,
        )
        remediation = self._string(validation.get("remediation")) or self._default_remediation(validation)
        return Finding(
            finding_id=f"finding::{vulnerability_ref}::{service_ref}",
            title=title,
            affected_asset_refs=self._affected_asset_refs(state=state, service_ref=service_ref, result=result),
            service_ref=service_ref,
            vulnerability_ref=vulnerability_ref,
            evidence_refs=evidence_refs,
            validation_status=str(validation.get("status")),  # type: ignore[arg-type]
            severity=risk_score.severity,
            cvss=cvss,
            epss=epss,
            kev=kev,
            confidence=confidence,
            false_positive_risk=round(1.0 - confidence, 3),
            remediation=remediation,
            risk_score=risk_score,
            provenance={
                "operation_id": state.operation_id,
                "task_ref": result.task_id,
                "execution_node_id": result.execution_node_id,
                "worker_result_id": result.result_id,
                "tool_output_refs": [item.payload_ref for item in result.evidence],
                "timestamp": result.created_at.isoformat(),
            },
        )

    def _record_finding_audit(self, *, state: RuntimeState, result: AgentTaskResult, event_type: str, validation: dict[str, Any], **payload: Any) -> None:
        entry = {
            "event_type": event_type,
            "at": utc_now().isoformat(),
            "source_task_id": result.task_id,
            "worker_result_id": result.result_id,
            "validation_status": validation.get("status"),
            "vulnerability_id": validation.get("vulnerability_id") or result.outcome_payload.get("vulnerability_id"),
            "service_id": result.outcome_payload.get("service_id"),
            **payload,
        }
        finding_audit = state.execution.metadata.setdefault("finding_audit", [])
        finding_audit.append(dict(entry))
        self._append_audit_log(state, entry)

    def _sync_credential_view(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        raw = self._dict(result.outcome_payload.get("credential_validation"))
        credential_id = self._string(raw.get("credential_id"))
        if credential_id is None:
            return
        principal = self._string(raw.get("principal")) or self._string(raw.get("username")) or "unknown-principal"
        if credential_id not in state.credentials:
            self._credential_manager.upsert_credential(
                state,
                credential_id,
                principal,
                kind=str(raw.get("kind", "password")),
                secret_ref=self._string(raw.get("secret_ref")),
                source_session_id=self._result_session_id(result=result, task=state.execution.tasks.get(result.task_id)),
                metadata={"created_by": "result_applier"},
            )
        status = self._string(raw.get("status")) or ("valid" if bool(raw.get("validated")) else "unknown")
        target_id = self._string(raw.get("target_id")) or self._string(raw.get("bound_target"))
        self._credential_manager.record_validation(
            state,
            credential_id,
            status=status,
            target_id=target_id,
            metadata={"validator_output": raw, "source_task_id": result.task_id},
        )

    def _apply_credential_hints(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for hints in self._runtime_hints(result):
            credential_id = self._string(hints.get("credential_id"))
            if credential_id is None:
                continue
            principal = self._string(hints.get("principal") or hints.get("username")) or "unknown-principal"
            if credential_id not in state.credentials:
                self._credential_manager.upsert_credential(
                    state,
                    credential_id,
                    principal,
                    kind=self._string(hints.get("kind") or hints.get("credential_kind")) or "password",
                    secret_ref=self._string(hints.get("secret_ref")),
                    source_session_id=self._result_session_id(result=result, task=state.execution.tasks.get(result.task_id)),
                    metadata={"created_by": "runtime_hints", "source_task_id": result.task_id},
                )
            status = (
                self._string(hints.get("credential_status"))
                or self._string(hints.get("status"))
                or ("valid" if bool(hints.get("authenticated")) else None)
            )
            if status is None:
                continue
            target_id = (
                self._string(hints.get("bind_target"))
                or self._string(hints.get("target_service_id"))
                or self._string(hints.get("target_id"))
            )
            self._credential_manager.record_validation(
                state,
                credential_id,
                status=status,
                target_id=target_id if status == "valid" else None,
                metadata={"runtime_hints": hints, "source_task_id": result.task_id},
            )

    def _apply_session_hints(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for hints in self._runtime_hints(result):
            if not bool(hints.get("open_session")):
                continue
            session_id = self._string(hints.get("session_id")) or f"session::{result.task_id}"
            lease_seconds = self._int(hints.get("lease_seconds")) or 300
            reuse_policy = self._string(hints.get("reuse_policy")) or "exclusive"
            session = self._session_manager.open_session(
                state,
                session_id=session_id,
                bound_identity=self._string(hints.get("bound_identity") or hints.get("identity")),
                bound_target=self._string(hints.get("bound_target") or hints.get("target_id")),
                lease_seconds=lease_seconds,
                reusability=reuse_policy,
            )
            self._session_manager.bind_task_to_session(state, result.task_id, session.session_id)
            lease_id = self._lease_id(session.session_id, result.task_id)
            lease = self._lease_manager.create_lease(
                state,
                lease_id=lease_id,
                session_id=session.session_id,
                owner_task_id=result.task_id,
                lease_seconds=lease_seconds,
                reuse_policy=reuse_policy,
                metadata={"created_by": "runtime_hints", "source_task_id": result.task_id},
            )
            self._lease_manager.bind_lease_to_task_or_session(
                state,
                lease.lease_id,
                task_id=result.task_id,
                session_id=session.session_id,
            )

    def _apply_pivot_route_hints(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for hints in self._runtime_hints(result):
            if not bool(hints.get("register_pivot_route")):
                continue
            destination_host = self._string(hints.get("destination_host") or hints.get("target_host"))
            if destination_host is None:
                continue
            route = self._pivot_route_manager.refresh_from_reachability(
                state,
                destination_host=destination_host,
                reachable=bool(hints.get("reachable", result.status == AgentResultStatus.SUCCEEDED)),
                source_host=self._string(hints.get("source_host")),
                via_host=self._string(hints.get("via_host")),
                session_id=self._string(hints.get("session_id")) or self._result_session_id(result=result, task=state.execution.tasks.get(result.task_id)),
                protocol=self._string(hints.get("protocol")),
                route_id=self._string(hints.get("route_id")),
                destination_zone=self._string(hints.get("destination_zone")),
                destination_cidr=self._string(hints.get("destination_cidr")),
                allowed_ports=self._list_or_set(hints.get("allowed_ports") or hints.get("port")),
                protocols=self._list_or_set(hints.get("protocols")),
                hop_count=self._int(hints.get("hop_count")),
                confidence=self._float(hints.get("confidence")),
                metadata={"runtime_hints": hints, "source_task_id": result.task_id},
            )
            task = state.execution.tasks.get(result.task_id)
            if task is not None:
                task.metadata["selected_route_id"] = route.route_id

    def _sync_pivot_route_view(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        self._reachability_propagator.sync_from_task_result(state=state, result=result)

    def _audit_stage_result(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        stage_result = result.outcome_payload.get("stage_result")
        if not isinstance(stage_result, dict):
            return
        self._append_audit_log(
            state,
            {
                "event_type": "stage_result_applied",
                "at": utc_now().isoformat(),
                "source_task_id": result.task_id,
                "worker_result_id": result.result_id,
                "stage_type": stage_result.get("stage_type"),
                "stage_status": stage_result.get("status"),
                "summary": result.summary,
            },
        )

    def _audit_stage_tool_trace(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        traces = self._stage_result_items(result, "tool_trace")
        if not traces:
            raw = result.outcome_payload.get("tool_trace")
            if isinstance(raw, list):
                traces = [dict(item) for item in raw if isinstance(item, dict)]
        if not traces:
            return
        for trace in traces:
            self._append_audit_log(
                state,
                {
                    "event_type": "stage_tool_trace",
                    "at": utc_now().isoformat(),
                    "source_task_id": result.task_id,
                    "worker_result_id": result.result_id,
                    "stage_type": result.metadata.get("stage_type") or result.outcome_payload.get("outcome_type"),
                    "server_id": trace.get("server_id"),
                    "tool_name": trace.get("tool_name"),
                    "success": trace.get("success"),
                    "exit_code": trace.get("exit_code"),
                    "summary": trace.get("summary"),
                },
            )

    def _release_task_execution_resources(self, *, state: RuntimeState, task_id: str) -> None:
        task = state.execution.tasks.get(task_id)
        if task is None:
            return
        self._lock_manager.release_all_for_owner(state, task_id)
        worker_id = task.assigned_worker
        if worker_id is not None and worker_id in state.workers:
            worker = state.workers[worker_id]
            worker.current_task_id = None
            if worker.status not in {WorkerStatus.LOST, WorkerStatus.UNAVAILABLE}:
                worker.status = WorkerStatus.IDLE
        task.assigned_worker = None

    def _cleanup_session_family(
        self,
        state: RuntimeState,
        *,
        session_id: str,
        reason: str,
        close_routes: bool,
    ) -> None:
        self._lease_manager.release_leases_for_session(state, session_id, reason=reason)
        self._credential_manager.expire_credentials_for_session(state, session_id, reason=reason)
        if close_routes:
            self._pivot_route_manager.close_routes_for_session(state, session_id, reason=reason)
        else:
            self._pivot_route_manager.fail_routes_for_session(state, session_id, reason=reason)

    def _close_single_use_session(self, *, state: RuntimeState, session_id: str | None, task_id: str) -> None:
        if session_id is None or session_id not in state.sessions:
            return
        session = state.sessions[session_id]
        if session.reusability != "single_use":
            self._session_manager.unbind_task_from_session(state, task_id, session_id)
            return
        self._session_manager.close_session(state, session_id, reason="single_use_session_completed")
        self._cleanup_session_family(
            state,
            session_id=session_id,
            reason="single_use_session_completed",
            close_routes=True,
        )

    @staticmethod
    def _is_runtime_setup_block(result: AgentTaskResult) -> bool:
        if result.status.value != "blocked":
            return False
        return any(request.request_type == RuntimeControlType.OPEN_SESSION for request in result.runtime_requests)

    @staticmethod
    def _lease_id(session_id: str, task_id: str) -> str:
        return f"lease::{session_id}::{task_id}"

    @staticmethod
    def _ref_id_for_type(result: AgentTaskResult, ref_type: str) -> str | None:
        expected = ref_type.lower()
        for collection in (result.observations, result.evidence):
            for item in collection:
                for ref in getattr(item, "refs", []):
                    if str(getattr(ref, "ref_type", "")).lower() == expected:
                        return str(getattr(ref, "ref_id"))
        for request in result.fact_write_requests:
            for ref in (request.subject_ref, request.object_ref):
                if ref is not None and str(getattr(ref, "ref_type", "")).lower() == expected:
                    return str(getattr(ref, "ref_id"))
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _default_remediation(validation: dict[str, Any]) -> str:
        cve = validation.get("cve")
        if cve:
            return f"Review vendor guidance for {cve}, apply the relevant patch, and validate compensating controls."
        return "Review vendor guidance, apply the relevant patch, and validate compensating controls."

    def _affected_asset_refs(self, *, state: RuntimeState, service_ref: str, result: AgentTaskResult) -> list[str]:
        refs = [service_ref]
        for ref_type in ("Host", "DataAsset"):
            ref_id = self._ref_id_for_type(result, ref_type)
            if ref_id is not None:
                refs.append(ref_id)
        for target in self._target_inventory(state):
            target_id = self._string(target.get("asset_id")) or self._string(target.get("value")) or self._string(target.get("address"))
            if target_id and (target_id in service_ref or service_ref in target_id):
                refs.append(target_id)
        return list(dict.fromkeys(refs))

    def _is_publicly_exposed(self, *, state: RuntimeState, service_ref: str, validation: dict[str, Any]) -> bool:
        explicit = validation.get("public_exposed")
        if explicit is not None:
            return bool(explicit)
        for target in self._target_inventory(state):
            text = " ".join(str(target.get(key, "")) for key in ("value", "address", "hostname", "url"))
            if text and (text in service_ref or service_ref in text):
                tags = {str(item).lower() for item in target.get("tags", []) if item is not None}
                metadata = self._dict(target.get("metadata"))
                return bool(metadata.get("public_exposed") or metadata.get("internet_exposed") or tags & {"public", "internet", "external"})
        return False

    def _is_critical_asset(self, *, state: RuntimeState, service_ref: str, validation: dict[str, Any]) -> bool:
        explicit = validation.get("critical_asset")
        if explicit is not None:
            return bool(explicit)
        for target in self._target_inventory(state):
            text = " ".join(str(target.get(key, "")) for key in ("value", "address", "hostname", "url"))
            if text and (text in service_ref or service_ref in text):
                tags = {str(item).lower() for item in target.get("tags", []) if item is not None}
                metadata = self._dict(target.get("metadata"))
                return bool(metadata.get("critical_asset") or metadata.get("business_critical") or tags & {"critical", "crown-jewel"})
        return False

    @staticmethod
    def _target_inventory(state: RuntimeState) -> list[dict[str, Any]]:
        value = state.execution.metadata.get("target_inventory", [])
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @staticmethod
    def _dict(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _result_session_id(self, *, result: AgentTaskResult, task: Any) -> str | None:
        for request in result.runtime_requests:
            if request.request_type in {RuntimeControlType.OPEN_SESSION, RuntimeControlType.EXTEND_SESSION, RuntimeControlType.EXPIRE_SESSION}:
                if request.session_id:
                    return request.session_id
        if task is not None:
            session_id = self._string(task.metadata.get("session_id"))
            if session_id is not None:
                return session_id
        return self._string(result.outcome_payload.get("session_id"))

    @classmethod
    def _runtime_hints(cls, result: AgentTaskResult) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []

        def add(value: Any) -> None:
            if isinstance(value, dict):
                hints.append(dict(value))

        add(result.outcome_payload.get("runtime_hints"))
        parsed = result.outcome_payload.get("parsed")
        if isinstance(parsed, dict):
            add(parsed.get("runtime_hints"))
        payload = result.outcome_payload.get("payload")
        if isinstance(payload, dict):
            add(payload.get("runtime_hints"))
            nested_parsed = payload.get("parsed")
            if isinstance(nested_parsed, dict):
                add(nested_parsed.get("runtime_hints"))
            mcp_payload = payload.get("mcp_payload")
            if isinstance(mcp_payload, dict):
                mcp_parsed = mcp_payload.get("parsed")
                if isinstance(mcp_parsed, dict):
                    add(mcp_parsed.get("runtime_hints"))
        mcp_payload = result.outcome_payload.get("mcp_payload")
        if isinstance(mcp_payload, dict):
            mcp_parsed = mcp_payload.get("parsed")
            if isinstance(mcp_parsed, dict):
                add(mcp_parsed.get("runtime_hints"))
        for observation in result.observations:
            add(observation.payload.get("runtime_hints"))
            parsed = observation.payload.get("parsed")
            if isinstance(parsed, dict):
                add(parsed.get("runtime_hints"))
        for evidence in result.evidence:
            add(evidence.metadata.get("runtime_hints"))
            parsed = evidence.metadata.get("parsed")
            if isinstance(parsed, dict):
                add(parsed.get("runtime_hints"))
            mcp_payload = evidence.metadata.get("mcp_payload")
            if isinstance(mcp_payload, dict):
                mcp_parsed = mcp_payload.get("parsed")
                if isinstance(mcp_parsed, dict):
                    add(mcp_parsed.get("runtime_hints"))

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for item in hints:
            fingerprint = tuple(sorted((str(key), str(value)) for key, value in item.items()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(item)
        return deduped

    @staticmethod
    def _stage_result_items(result: AgentTaskResult, key: str) -> list[dict[str, Any]]:
        stage_result = result.outcome_payload.get("stage_result")
        if not isinstance(stage_result, dict):
            return []
        value = stage_result.get(key)
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    def _structured_stage_state_deltas(
        self,
        *,
        stage_result: StageResult,
        result: AgentTaskResult,
    ) -> list[dict[str, Any]]:
        """Extract common black-box discovery shapes into KG state deltas.

        This accepts schema-level facts only: hosts, services, findings and
        evidence records that the stage/tool output already contains. It does
        not infer target-specific topology, ports, credentials or vulnerabilities.
        """

        entities: dict[str, KGEntityPatch] = {}
        relations: dict[str, KGRelationPatch] = {}
        source_refs = self._structured_stage_source_refs(stage_result)
        provenance = {
            "worker_result_id": result.result_id,
            "source_task_id": stage_result.stage_task_id,
            "source_agent": stage_result.agent_name,
            "stage_result_id": stage_result.result_id,
        }

        def add_entity(patch: KGEntityPatch) -> None:
            entities[patch.entity_id] = patch

        def add_relation(patch: KGRelationPatch) -> None:
            relations[patch.relation_id] = patch

        for payload in self._iter_stage_structured_payloads(stage_result):
            for host in self._extract_host_records(payload):
                host_id = self._host_entity_id(host)
                if not host_id:
                    continue
                label = self._coalesce_string(host.get("label"), host.get("hostname"), host.get("address"), host_id)
                add_entity(
                    KGEntityPatch(
                        entity_id=host_id,
                        entity_type=NodeType.HOST.value,
                        label=label,
                        attributes={
                            **self._whitelist_attributes(host, self._HOST_ATTR_KEYS),
                            "address": self._coalesce_optional_string(host.get("address"), host.get("host"), host.get("ip")),
                            "confidence": self._bounded_confidence(host.get("confidence"), stage_result.confidence),
                            "source_task_id": stage_result.stage_task_id,
                        },
                        source_refs=source_refs,
                        provenance=provenance,
                    )
                )

            for service in self._extract_service_records(payload):
                host_id = self._host_entity_id(service)
                service_id = self._service_entity_id(service)
                if not host_id or not service_id:
                    continue
                # 只在 host 尚未被建过时补一个骨架节点；若 host 循环已写入富属性，
                # 不能用这个精简 patch 覆盖它（否则 host 的 hostname/属性会被抹掉）。
                if host_id not in entities:
                    add_entity(
                        KGEntityPatch(
                            entity_id=host_id,
                            entity_type=NodeType.HOST.value,
                            label=self._coalesce_string(service.get("host"), service.get("address"), service.get("ip"), host_id),
                            attributes={
                                "address": self._coalesce_optional_string(service.get("host"), service.get("address"), service.get("ip")),
                                "confidence": self._bounded_confidence(service.get("confidence"), stage_result.confidence),
                                "source_task_id": stage_result.stage_task_id,
                            },
                            source_refs=source_refs,
                            provenance=provenance,
                        )
                    )
                protocol = self._coalesce_string(service.get("protocol"), "tcp").lower()
                port = self._coerce_port(service.get("port"))
                label = self._service_label(service, service_id)
                add_entity(
                    KGEntityPatch(
                        entity_id=service_id,
                        entity_type=NodeType.SERVICE.value,
                        label=label,
                        attributes={
                            **self._whitelist_attributes(service, self._SERVICE_ATTR_KEYS),
                            "service_name": self._coalesce_optional_string(
                                service.get("service_name"),
                                service.get("service"),
                                service.get("name"),
                                service.get("product"),
                            ),
                            "port": port,
                            "protocol": protocol,
                            "confidence": self._bounded_confidence(service.get("confidence"), stage_result.confidence),
                            "source_task_id": stage_result.stage_task_id,
                        },
                        source_refs=source_refs,
                        provenance=provenance,
                    )
                )
                add_relation(
                    KGRelationPatch(
                        relation_id=f"{EdgeType.HOSTS.value.lower()}::{host_id}::{service_id}",
                        relation_type=EdgeType.HOSTS.value,
                        source=host_id,
                        target=service_id,
                        label="hosts",
                        attributes={
                            "confidence": self._bounded_confidence(service.get("confidence"), stage_result.confidence),
                            "source_task_id": stage_result.stage_task_id,
                        },
                        source_refs=source_refs,
                        provenance=provenance,
                    )
                )

            for evidence in self._extract_negative_evidence_records(payload, stage_result):
                add_entity(
                    KGEntityPatch(
                        entity_id=str(evidence["id"]),
                        entity_type=NodeType.EVIDENCE.value,
                        label=str(evidence["summary"]),
                        attributes={
                            "summary": evidence["summary"],
                            "evidence_kind": evidence["kind"],
                            "content_ref": evidence.get("payload_ref"),
                            "confidence": self._bounded_confidence(evidence.get("confidence"), stage_result.confidence),
                            "source_task_id": stage_result.stage_task_id,
                            "properties": dict(evidence.get("metadata") or {}),
                        },
                        source_refs=source_refs,
                        provenance=provenance,
                    )
                )

        for finding in stage_result.findings:
            if not isinstance(finding, dict):
                continue
            finding_id = self._coalesce_string(
                finding.get("finding_id"),
                finding.get("id"),
                f"finding::{stage_result.stage_task_id}::{len(entities)}",
            )
            summary = self._coalesce_string(finding.get("title"), finding.get("summary"), finding.get("description"), finding_id)
            add_entity(
                KGEntityPatch(
                    entity_id=finding_id,
                    entity_type=NodeType.FINDING.value,
                    label=summary,
                    attributes={
                        **self._whitelist_attributes(finding, self._FINDING_ATTR_KEYS),
                        "summary": summary,
                        "finding_kind": self._coalesce_optional_string(finding.get("kind"), finding.get("type")),
                        "severity": self._coalesce_optional_string(finding.get("severity")),
                        "confidence": self._bounded_confidence(finding.get("confidence"), stage_result.confidence),
                        "evidence_refs": list(finding.get("evidence_refs") or stage_result.evidence_refs),
                        "source_task_id": stage_result.stage_task_id,
                    },
                    source_refs=source_refs,
                    provenance=provenance,
                )
            )

        # 降级保障：若整段 stage 没抽出任何结构化实体，但确有成功的工具调用，
        # 至少把每次工具结果落成一个 Evidence 节点，避免「工具结果完全写不进图」。
        if not entities:
            for trace in stage_result.tool_trace:
                if not getattr(trace, "success", False):
                    continue
                evidence_id = f"evidence::tool::{stage_result.stage_task_id}::{trace.trace_id}"
                summary = self._coalesce_string(
                    trace.summary, trace.input_summary, f"tool {trace.tool_name} result"
                )
                add_entity(
                    KGEntityPatch(
                        entity_id=evidence_id,
                        entity_type=NodeType.EVIDENCE.value,
                        label=summary,
                        attributes={
                            "summary": summary,
                            "evidence_kind": "tool_result",
                            "content_ref": trace.raw_output_ref,
                            "confidence": self._bounded_confidence(None, stage_result.confidence),
                            "source_task_id": stage_result.stage_task_id,
                            "properties": {
                                "tool_name": trace.tool_name,
                                "tool_category": trace.tool_category,
                                "step": trace.step,
                            },
                        },
                        source_refs=source_refs,
                        provenance=provenance,
                    )
                )

        return self._state_writer.build_state_deltas(
            entity_patches=list(entities.values()),
            relation_patches=list(relations.values()),
            require_known_endpoints=True,
        )

    # 白名单：只把这些 schema 级字段写进节点属性，不再把原始工具输出整包 spread 进图。
    _HOST_ATTR_KEYS = frozenset(
        {"address", "hostname", "label", "os", "os_family", "mac", "status", "network_zone", "state"}
    )
    _SERVICE_ATTR_KEYS = frozenset(
        {"service_name", "service", "name", "product", "version", "port", "protocol", "banner", "cpe", "state", "tls"}
    )
    _FINDING_ATTR_KEYS = frozenset(
        {"title", "summary", "description", "kind", "type", "severity", "category", "cwe", "cve", "reference"}
    )

    @staticmethod
    def _whitelist_attributes(record: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
        """Keep only whitelisted schema-level keys from a raw tool/finding record."""

        return {key: value for key, value in record.items() if key in allowed and value is not None}

    @staticmethod
    def _structured_stage_source_refs(stage_result: StageResult) -> list[dict[str, Any]]:
        return [
            GraphRef(graph=GraphScope.RUNTIME, ref_id=stage_result.stage_task_id, ref_type="stage_task").model_dump(mode="json"),
            GraphRef(graph=GraphScope.RUNTIME, ref_id=stage_result.result_id, ref_type="stage_result").model_dump(mode="json"),
        ]

    @classmethod
    def _iter_stage_structured_payloads(cls, stage_result: StageResult) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for collection in (stage_result.observations, stage_result.evidence, stage_result.findings):
            for item in collection:
                if not isinstance(item, dict):
                    continue
                payloads.append(dict(item))
                for key in ("payload", "metadata", "properties", "parsed_output", "parsed"):
                    nested = item.get(key)
                    if isinstance(nested, dict):
                        payloads.append(dict(nested))
        for trace in stage_result.tool_trace:
            if trace.parsed_output:
                payloads.append(dict(trace.parsed_output))
            if isinstance(trace.metadata, dict):
                # parsed_output 为空时的降级路径：从 metadata 里回退读结构化载荷，
                # 覆盖 parsed/parsed_output/result/content/output 几种常见落点。
                for key in ("parsed", "parsed_output", "result", "content", "output"):
                    nested = trace.metadata.get(key)
                    if isinstance(nested, dict):
                        payloads.append(dict(nested))
                if any(k in trace.metadata for k in ("host", "address", "ip", "hostname", "port", "ports", "service", "service_name", "hosts", "services")):
                    payloads.append(dict(trace.metadata))
        return cls._dedupe_dicts(payloads)

    @classmethod
    def _extract_host_records(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for key in ("hosts_up", "live_hosts", "reachable_hosts", "discovered_hosts", "hosts"):
            for item in cls._as_list(payload.get(key)):
                record = cls._host_record(item)
                if record:
                    records.append(record)
        record = cls._host_record(payload)
        if record and any(key in payload for key in ("host", "address", "ip", "hostname")):
            records.append(record)
        return cls._dedupe_dicts(records)

    @classmethod
    def _extract_service_records(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for key in (
            "service_discovery",
            "open_services",
            "services",
            "discovered_services",
            "known_services",
            "service_fingerprints",
        ):
            for item in cls._as_list(payload.get(key)):
                records.extend(cls._service_records(item))
        analysis = payload.get("analysis")
        if isinstance(analysis, dict):
            for item in cls._as_list(analysis.get("service_fingerprints")):
                records.extend(cls._service_records(item))
        if any(key in payload for key in ("port", "ports", "service", "service_name")):
            records.extend(cls._service_records(payload))
        return cls._dedupe_dicts(records)

    @classmethod
    def _extract_negative_evidence_records(cls, payload: dict[str, Any], stage_result: StageResult) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for key in ("negative_evidence", "negative_observations"):
            for index, item in enumerate(cls._as_list(payload.get(key))):
                if isinstance(item, dict):
                    summary = cls._coalesce_string(item.get("summary"), item.get("description"), item.get("reason"), item)
                    metadata = dict(item)
                else:
                    summary = str(item)
                    metadata = {"value": item}
                if not summary:
                    continue
                records.append(
                    {
                        "id": cls._coalesce_string(
                            metadata.get("evidence_id"),
                            metadata.get("id"),
                            f"evidence::{stage_result.stage_task_id}::{key}::{index}",
                        ),
                        "kind": "negative_evidence",
                        "summary": summary,
                        "payload_ref": metadata.get("payload_ref") or f"runtime://stage-results/{stage_result.stage_task_id}",
                        "metadata": metadata,
                        "confidence": metadata.get("confidence"),
                    }
                )
        return records

    @classmethod
    def _host_record(cls, item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict):
            value = cls._coalesce_optional_string(item.get("host"), item.get("address"), item.get("ip"), item.get("hostname"), item.get("target"))
            if not value:
                return None
            return {**item, "host": value, "address": cls._coalesce_optional_string(item.get("address"), value)}
        if item is None:
            return None
        value = str(item).strip()
        if not value:
            return None
        host, _port, _protocol = cls._split_host_port(value)
        return {"host": host, "address": host, "value": value}

    @classmethod
    def _service_records(cls, item: Any) -> list[dict[str, Any]]:
        if isinstance(item, dict):
            host = cls._coalesce_optional_string(item.get("host"), item.get("address"), item.get("ip"), item.get("target"), item.get("hostname"))
            port = cls._coerce_port(item.get("port"))
            protocol = cls._coalesce_optional_string(item.get("protocol"), item.get("scheme")) or "tcp"
            if host and port is None:
                split_host, split_port, split_protocol = cls._split_host_port(host)
                host = split_host
                port = split_port
                protocol = split_protocol or protocol
            records: list[dict[str, Any]] = []
            ports = item.get("ports")
            if isinstance(ports, list):
                for raw_port in ports:
                    candidate = dict(item)
                    candidate.pop("ports", None)
                    candidate["host"] = host
                    candidate["port"] = cls._coerce_port(raw_port)
                    candidate["protocol"] = protocol
                    if candidate["host"] and candidate["port"]:
                        records.append(candidate)
            if host and port:
                fingerprint = item.get("improved_fingerprint") if isinstance(item.get("improved_fingerprint"), dict) else {}
                current = item.get("current_fingerprint") if isinstance(item.get("current_fingerprint"), dict) else {}
                records.append(
                    {
                        **item,
                        "host": host,
                        "port": port,
                        "protocol": protocol,
                        "service": cls._coalesce_optional_string(
                            item.get("service"),
                            item.get("service_name"),
                            fingerprint.get("application"),
                            current.get("service"),
                            current.get("product"),
                        ),
                        "product": cls._coalesce_optional_string(
                            item.get("product"),
                            fingerprint.get("application"),
                            current.get("product"),
                        ),
                        "version": cls._coalesce_optional_string(
                            item.get("version"),
                            fingerprint.get("application_version"),
                            current.get("version"),
                        ),
                    }
                )
            return records
        if item is None:
            return []
        host, port, protocol = cls._split_host_port(str(item))
        if not host or not port:
            return []
        return [{"host": host, "port": port, "protocol": protocol or "tcp", "value": str(item)}]

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        return [value]

    @classmethod
    def _host_entity_id(cls, record: dict[str, Any]) -> str | None:
        explicit = cls._coalesce_optional_string(record.get("host_id"), record.get("entity_id"), record.get("id"))
        if explicit and str(record.get("type") or record.get("entity_type") or "Host").lower() == "host":
            return explicit
        value = cls._coalesce_optional_string(record.get("host"), record.get("address"), record.get("ip"), record.get("hostname"))
        if not value:
            return None
        return value if value.startswith("host::") else f"host::{value}"

    @classmethod
    def _service_entity_id(cls, record: dict[str, Any]) -> str | None:
        explicit = cls._coalesce_optional_string(record.get("service_id"), record.get("entity_id"), record.get("id"))
        if explicit and str(record.get("type") or record.get("entity_type") or "Service").lower() == "service":
            return explicit
        host = cls._coalesce_optional_string(record.get("host"), record.get("address"), record.get("ip"), record.get("hostname"))
        port = cls._coerce_port(record.get("port"))
        if not host or not port:
            return None
        protocol = cls._coalesce_string(record.get("protocol"), "tcp").lower()
        return f"service::{host}:{port}/{protocol}"

    @classmethod
    def _service_label(cls, record: dict[str, Any], fallback: str) -> str:
        host = cls._coalesce_optional_string(record.get("host"), record.get("address"), record.get("ip"), record.get("hostname"))
        port = cls._coerce_port(record.get("port"))
        protocol = cls._coalesce_string(record.get("protocol"), "tcp").lower()
        name = cls._coalesce_optional_string(record.get("service_name"), record.get("service"), record.get("name"), record.get("product"))
        if host and port and name:
            return f"{host}:{port}/{protocol} {name}"
        if host and port:
            return f"{host}:{port}/{protocol}"
        return fallback

    @staticmethod
    def _split_host_port(value: str) -> tuple[str | None, int | None, str | None]:
        text = value.strip()
        if not text:
            return None, None, None
        protocol: str | None = None
        if "://" in text:
            scheme, remainder = text.split("://", 1)
            protocol = scheme.lower() or None
            text = remainder.split("/", 1)[0]
        elif "/" in text:
            text, protocol = text.rsplit("/", 1)
            protocol = protocol.lower() or None
        if ":" not in text:
            return text, None, protocol
        host, port_text = text.rsplit(":", 1)
        return host or None, PhaseTwoResultApplier._coerce_port(port_text), protocol

    @staticmethod
    def _coerce_port(value: Any) -> int | None:
        try:
            port = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        if 1 <= port <= 65535:
            return port
        return None

    @staticmethod
    def _bounded_confidence(value: Any, fallback: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = float(fallback)
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _coalesce_optional_string(*values: Any) -> str | None:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @classmethod
    def _coalesce_string(cls, *values: Any) -> str:
        return cls._coalesce_optional_string(*values) or "unknown"

    @staticmethod
    def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            fingerprint = repr(sorted(item.items(), key=lambda pair: str(pair[0])))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            result.append(item)
        return result

    @staticmethod
    def _result_target_host_id(result: AgentTaskResult) -> str | None:
        for collection in (result.observations, result.evidence):
            for item in collection:
                for ref in getattr(item, "refs", []):
                    if str(getattr(ref, "ref_type", "")).lower() == "host":
                        return str(getattr(ref, "ref_id"))
        for request in result.fact_write_requests:
            for ref in (request.subject_ref, request.object_ref):
                if ref is not None and str(getattr(ref, "ref_type", "")).lower() == "host":
                    return str(getattr(ref, "ref_id"))
        return None

    def _fact_state_deltas(self, *, result: AgentTaskResult) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for request in result.fact_write_requests:
            if request.kind == FactWriteKind.RELATION_UPSERT or (
                request.object_ref is not None and request.relation_type
            ):
                deltas.append(self._relation_delta(result=result, request=request))
            else:
                deltas.append(self._entity_delta(result=result, request=request))
        return deltas

    def _entity_delta(self, *, result: AgentTaskResult, request: FactWriteRequest) -> dict[str, Any]:
        if request.subject_ref is None:
            raise ValueError("entity/assertion fact writes require subject_ref")
        subject_ref = self._to_protocol_ref(request.subject_ref)
        evidence_chain = self._evidence_chain(result=result, request=request)
        patch = {
            "patch_id": request.proposal_id,
            "entity_id": subject_ref.ref_id,
            "operation": "upsert",
            "entity_kind": "node",
            "entity_type": subject_ref.ref_type or "entity",
            "label": request.summary,
            "attributes": {
                **dict(request.attributes),
                "confidence": request.confidence,
                "evidence_ids": list(request.evidence_ids),
                "evidence_chain": evidence_chain,
                "source_task_id": request.source_task_id,
                "fact_kind": request.kind.value,
            },
            "source_refs": [subject_ref.model_dump(mode="json")],
            "provenance": {
                "worker_result_id": result.result_id,
                "source_task_id": request.source_task_id,
                "source_agent": result.agent_role.value,
                "evidence_chain": evidence_chain,
            },
        }
        return (
            StateDeltaRecord(
                id=f"delta-{request.proposal_id}",
                source_agent=result.agent_role.value,
                summary=request.summary,
                confidence=request.confidence,
                refs=[subject_ref],
                payload={"patch_kind": "entity", "fact_kind": request.kind.value},
                graph_scope=GraphScope.KG,
                delta_type="upsert_entity",
                target_ref=subject_ref,
                patch=patch,
            ).to_agent_output_fragment()
            | {"write_type": "structural"}
        )

    def _relation_delta(self, *, result: AgentTaskResult, request: FactWriteRequest) -> dict[str, Any]:
        if request.subject_ref is None or request.object_ref is None or not request.relation_type:
            raise ValueError("relation fact writes require subject_ref, object_ref and relation_type")
        source_ref = self._to_protocol_ref(request.subject_ref)
        object_ref = self._to_protocol_ref(request.object_ref)
        relation_id = f"{request.relation_type.lower()}::{source_ref.ref_id}::{object_ref.ref_id}"
        evidence_chain = self._evidence_chain(result=result, request=request)
        patch = {
            "patch_id": request.proposal_id,
            "relation_id": relation_id,
            "operation": "upsert",
            "entity_kind": "edge",
            "relation_type": request.relation_type,
            "source": source_ref.ref_id,
            "target": object_ref.ref_id,
            "label": request.summary,
            "attributes": {
                **dict(request.attributes),
                "confidence": request.confidence,
                "evidence_ids": list(request.evidence_ids),
                "evidence_chain": evidence_chain,
                "source_task_id": request.source_task_id,
                "fact_kind": request.kind.value,
            },
            "source_refs": [
                source_ref.model_dump(mode="json"),
                object_ref.model_dump(mode="json"),
            ],
            "provenance": {
                "worker_result_id": result.result_id,
                "source_task_id": request.source_task_id,
                "source_agent": result.agent_role.value,
                "evidence_chain": evidence_chain,
            },
        }
        target_ref = GraphRef(graph=GraphScope.KG, ref_id=relation_id, ref_type=request.relation_type)
        return (
            StateDeltaRecord(
                id=f"delta-{request.proposal_id}",
                source_agent=result.agent_role.value,
                summary=request.summary,
                confidence=request.confidence,
                refs=[source_ref, object_ref],
                payload={"patch_kind": "relation", "fact_kind": request.kind.value},
                graph_scope=GraphScope.KG,
                delta_type="upsert_relation",
                target_ref=target_ref,
                patch=patch,
            ).to_agent_output_fragment()
            | {"write_type": "structural"}
        )

    def _kg_batch(self, *, state_deltas: list[dict[str, Any]]) -> KGEventBatch:
        events: list[KGDeltaEvent] = []
        for delta in state_deltas:
            target_ref = GraphRef.model_validate(delta["target_ref"])
            patch = dict(delta.get("patch", {}))
            delta_type = str(delta.get("delta_type", ""))
            events.append(
                KGDeltaEvent(
                    event_type=self._kg_event_type(delta_type=delta_type, patch=patch),
                    source_agent=str(delta.get("source_agent", "phase_two_result_applier")),
                    target_ref=target_ref,
                    patch=patch,
                    metadata={"state_delta_id": delta.get("id")},
                )
            )
        return KGEventBatch.from_events(events, metadata={"source": self.__class__.__name__})

    # 中文注释：
    # 所有结构化事实写入都带上证据链，便于后续审计、回放和人工复核。
    @staticmethod
    def _evidence_chain(*, result: AgentTaskResult, request: FactWriteRequest) -> dict[str, Any]:
        return {
            "evidence_ids": list(request.evidence_ids),
            "worker_result_id": result.result_id,
            "source_task_id": request.source_task_id,
            "source_agent": result.agent_role.value,
            "summary": request.summary,
        }

    # 中文注释：
    # 工具调用审计统一写入 runtime metadata，不让 worker 直接操作审计存储。
    def _audit_tool_invocations(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for evidence in result.evidence:
            metadata = dict(evidence.metadata)
            tool = metadata.get("tool")
            command = metadata.get("command") or metadata.get("tool_command")
            adapter = metadata.get("adapter")
            if tool is None and command is None and adapter is None:
                continue
            self._append_audit_log(
                state,
                {
                    "event_type": "tool_invocation",
                    "source_task_id": result.task_id,
                    "worker_result_id": result.result_id,
                    "evidence_id": evidence.evidence_id,
                    "tool": tool,
                    "command": command,
                    "adapter": adapter,
                    "payload_ref": evidence.payload_ref,
                },
            )

    def _audit_tool_execution_from_result(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        seen: set[tuple[tuple[str, str], ...]] = set()
        for tool_execution in self._tool_execution_candidates(result):
            normalized = self._normalize_tool_execution(tool_execution)
            if not normalized:
                continue
            fingerprint = tuple(sorted((key, str(value)) for key, value in normalized.items() if value is not None))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            success = bool(normalized.get("success", False))
            self._append_audit_log(
                state,
                {
                    "event_type": "tool_execution_recorded" if success else "tool_execution_failed",
                    "source_task_id": result.task_id,
                    "worker_result_id": result.result_id,
                    "adapter": normalized.get("adapter"),
                    "tool": normalized.get("tool"),
                    "success": success,
                    "exit_code": normalized.get("exit_code"),
                    "command_id": normalized.get("command_id"),
                    "payload_ref": normalized.get("payload_ref"),
                    "stdout_excerpt": normalized.get("stdout_excerpt"),
                    "stderr_excerpt": normalized.get("stderr_excerpt"),
                },
            )

    @staticmethod
    def _tool_execution_candidates(result: AgentTaskResult) -> list[Any]:
        candidates: list[Any] = [
            result.outcome_payload.get("tool_execution"),
            result.metadata.get("tool_execution"),
        ]
        for evidence in result.evidence:
            candidates.append(evidence.metadata.get("tool_execution"))
        return [candidate for candidate in candidates if candidate is not None]

    @classmethod
    def _normalize_tool_execution(cls, tool_execution: Any) -> dict[str, Any]:
        if isinstance(tool_execution, BaseModel):
            data = tool_execution.model_dump(mode="json")
        elif isinstance(tool_execution, dict):
            data = dict(tool_execution)
        else:
            return {}
        return {
            "adapter": cls._string(data.get("adapter")) or "unknown_adapter",
            "tool": cls._string(data.get("tool")) or "unknown_tool",
            "success": bool(data.get("success", False)),
            "exit_code": data.get("exit_code"),
            "command_id": cls._string(data.get("command_id")),
            "payload_ref": cls._string(data.get("payload_ref")),
            "stdout_excerpt": cls._excerpt(data.get("stdout")),
            "stderr_excerpt": cls._excerpt(data.get("stderr")),
        }

    @staticmethod
    def _excerpt(value: Any, limit: int = 500) -> str:
        if value is None:
            return ""
        return str(value)[:limit]

    def _audit_fact_writes(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for request in result.fact_write_requests:
            evidence_chain = self._evidence_chain(result=result, request=request)
            self._append_audit_log(
                state,
                {
                    "event_type": "fact_write",
                    "source_task_id": request.source_task_id,
                    "worker_result_id": result.result_id,
                    "proposal_id": request.proposal_id,
                    "fact_kind": request.kind.value,
                    "summary": request.summary,
                    "evidence_chain": evidence_chain,
                },
            )
            self._append_audit_log(
                state,
                {
                    "event_type": "evidence_chain",
                    "source_task_id": request.source_task_id,
                    "worker_result_id": result.result_id,
                    "proposal_id": request.proposal_id,
                    "subject_ref": request.subject_ref.model_dump(mode="json") if request.subject_ref is not None else None,
                    "object_ref": request.object_ref.model_dump(mode="json") if request.object_ref is not None else None,
                    "evidence_chain": evidence_chain,
                },
            )

    def _record_recent_outcome(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        payload_ref = self._result_payload_ref(result)
        state.record_outcome(
            OutcomeCacheEntry(
                outcome_id=self._string(result.outcome_payload.get("id")) or f"outcome::{result.task_id}",
                task_id=result.task_id,
                outcome_type=self._string(result.outcome_payload.get("outcome_type"))
                or self._string(result.outcome_payload.get("kind"))
                or "worker_outcome",
                summary=self._string(result.outcome_payload.get("summary")) or result.summary,
                payload_ref=payload_ref,
                metadata={
                    "status": result.status.value,
                    "source_agent": result.agent_role.value,
                    "worker_result_id": result.result_id,
                    "outcome_payload": result.outcome_payload,
                },
            )
        )

    @staticmethod
    def _kg_event_type(*, delta_type: str, patch: dict[str, Any]) -> KGDeltaEventType:
        operation = str(patch.get("operation") or "upsert").lower()
        if delta_type == "upsert_relation":
            return KGDeltaEventType.RELATION_UPDATED if operation == "update" else KGDeltaEventType.RELATION_ADDED
        if delta_type == "upsert_entity":
            return KGDeltaEventType.ENTITY_UPDATED if operation == "update" else KGDeltaEventType.ENTITY_ADDED
        if "confidence" in patch:
            return KGDeltaEventType.CONFIDENCE_CHANGED
        return KGDeltaEventType.ENTITY_UPDATED

    @staticmethod
    def _resolve_kg_ref(result: AgentTaskResult) -> GraphRef:
        for collection in (result.observations, result.evidence, result.fact_write_requests):
            for item in collection:
                refs = getattr(item, "refs", None)
                if refs is None and isinstance(item, FactWriteRequest):
                    refs = [ref for ref in (item.subject_ref, item.object_ref) if ref is not None]
                for ref in refs or []:
                    graph = getattr(ref, "graph", None)
                    graph_value = graph.value if hasattr(graph, "value") else str(graph or "").lower()
                    if graph_value == GraphScope.KG.value:
                        return PhaseTwoResultApplier._to_protocol_ref(ref)
        return GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")

    @staticmethod
    def _to_protocol_ref(ref: Any) -> GraphRef:
        graph = getattr(ref, "graph", None)
        graph_value = graph.value if hasattr(graph, "value") else str(graph or "").lower()
        metadata = dict(getattr(ref, "metadata", {}) or {})
        if graph_value == "query":
            metadata.setdefault("original_graph", "query")
            graph_value = GraphScope.AG.value
        label = getattr(ref, "label", None)
        if label is not None and "label" not in metadata:
            metadata["label"] = label
        return GraphRef(
            graph=GraphScope(graph_value),
            ref_id=str(getattr(ref, "ref_id")),
            ref_type=getattr(ref, "ref_type", None),
            metadata=metadata,
        )

    @staticmethod
    def _push_runtime_event(state: RuntimeState, event: Any) -> RuntimeEventRef:
        ref = event_to_ref(event, cursor=state.event_cursor + 1)
        state.push_event(ref)
        return ref

    @staticmethod
    def _append_audit_log(state: RuntimeState, entry: dict[str, Any]) -> None:
        append_audit_log(state, entry)

    def _runtime_control_event(
        self,
        state: RuntimeState,
        request: RuntimeControlRequest,
        *,
        summary: str,
    ) -> RuntimeEventRef:
        self._append_audit_log(
            state,
            {
                "event_type": request.request_type.value,
                "source_task_id": request.source_task_id,
                "request_id": request.request_id,
                "summary": summary,
                "metadata": dict(request.metadata),
            },
        )
        ref = RuntimeEventRef(
            event_id=request.request_id,
            event_type=request.request_type.value,
            cursor=state.event_cursor + 1,
            summary=summary,
            metadata={"source_task_id": request.source_task_id, **dict(request.metadata)},
        )
        state.push_event(ref)
        return ref

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _list_or_set(value: Any) -> list[Any] | set[Any] | None:
        if value is None:
            return None
        if isinstance(value, (list, set)):
            return value
        return [value]

    @staticmethod
    def _int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _result_payload_ref(result: AgentTaskResult) -> str:
        raw_result_ref = result.outcome_payload.get("raw_result_ref")
        if raw_result_ref is not None:
            return str(raw_result_ref)
        if result.evidence:
            return str(result.evidence[0].payload_ref)
        return f"runtime://worker-results/{result.task_id}"


__all__ = ["PhaseTwoApplyResult", "PhaseTwoResultApplier"]
