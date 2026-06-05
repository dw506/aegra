"""Extract AG attack-process records from structured runtime objects."""

from __future__ import annotations

from typing import Any

from src.core.models.attack_process import (
    AgentExecutionNode,
    AttackCycleNode,
    AttackProcessEdge,
    AttackProcessEdgeType,
    BlockedReasonNode,
    GraphRef,
    GoalCheckNode,
    HandoffSuggestionNode,
    PlannerDecisionNode,
    StageResultNode,
    StopDecisionNode,
    ToolCallNode,
    stable_node_id,
)
from src.core.planning.models import PlannerDecision
from src.core.runtime.attack_log_models import AttackLogExtraction
from src.core.stage.models import StageResult, ToolTrace


class AttackLogExtractor:
    """Convert planner/stage/tool records into AG process-history records.

    This extractor is intentionally mechanical: it never calls an LLM and never
    infers missing vulnerabilities, facts, or attack paths. It only copies
    bounded summaries and refs from already-structured objects.
    """

    def extract(
        self,
        planner_decision: PlannerDecision | StageResult,
        stage_result: StageResult | None = None,
        tool_traces: list[ToolTrace] | None = None,
        runtime_events: list[dict[str, Any]] | None = None,
        policy_events: list[dict[str, Any]] | None = None,
    ) -> AttackLogExtraction:
        """Extract process nodes and edges.

        `extract(stage_result)` is accepted for legacy callers; the preferred
        path is `extract(planner_decision, stage_result, tool_traces, ...)`.
        """

        if isinstance(planner_decision, StageResult):
            stage_result = planner_decision
            decision: PlannerDecision | None = None
            operation_id = stage_result.operation_id
            cycle_index = self._cycle_from_stage_result(stage_result)
        else:
            decision = planner_decision
            operation_id = decision.operation_id
            cycle_index = decision.cycle_index

        traces = list(tool_traces or [])
        if stage_result is not None:
            traces.extend(stage_result.tool_trace or stage_result.tool_traces)
        traces = self._dedupe_traces(traces)

        nodes: list[Any] = []
        edges: list[AttackProcessEdge] = []
        evidence_refs = self._collect_evidence_refs(stage_result, traces, runtime_events or [], policy_events or [])

        cycle_id = self._cycle_id(operation_id, cycle_index)
        nodes.append(
            AttackCycleNode(
                id=cycle_id,
                label=f"Attack cycle {cycle_index}",
                operation_id=operation_id,
                cycle_index=cycle_index,
                status="completed" if stage_result is not None else "planned",
                summary=f"cycle {cycle_index}",
                evidence_refs=list(evidence_refs),
                properties={
                "node_role": "ATTACK_CYCLE",
                "display_name": f"Cycle {cycle_index}",
                "visual_title": self._cycle_visual_title(cycle_index, decision, stage_result),
                "visual_summary": self._cycle_visual_summary(decision, stage_result, traces),
                "cycle_index": cycle_index,
                    "step_order": 1,
                    "status": "completed" if stage_result is not None else "planned",
                    "runtime_event_count": len(runtime_events or []),
                    "policy_event_count": len(policy_events or []),
                },
            )
        )

        planner_id: str | None = None
        execution_id: str | None = None
        if decision is not None:
            planner_id = self._planner_id(decision)
            nodes.append(self._planner_node(decision, planner_id))
            edges.append(
                self._edge(
                    edge_type=AttackProcessEdgeType.PLANNED,
                    source=cycle_id,
                    target=planner_id,
                    label="planned",
                )
            )
            if decision.selected_agent is not None:
                execution_id = self._execution_id(operation_id, cycle_index, decision.selected_agent)
                nodes.append(self._execution_node_from_decision(decision, execution_id, planner_id))
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.DISPATCHED_TO,
                        source=planner_id,
                        target=execution_id,
                        label="dispatched to",
                    )
                )
            if decision.decision in {"stop_success", "stop_failed"}:
                stop_id = self._stop_decision_id(decision)
                nodes.append(self._stop_decision_node(decision, stop_id))
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.PRODUCED_RESULT,
                        source=planner_id,
                        target=stop_id,
                        label="produced stop decision",
                    )
                )

        if stage_result is not None:
            if execution_id is None:
                execution_id = self._execution_id(operation_id, cycle_index, stage_result.agent_name)
                nodes.append(self._execution_node_from_stage(stage_result, cycle_index, execution_id))
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.DISPATCHED_TO,
                        source=planner_id or cycle_id,
                        target=execution_id,
                        label="dispatched to",
                    )
                )

            result_id = self._stage_result_id(stage_result)
            tool_ids: list[str] = []
            for trace in traces:
                tool_id = self._tool_id(operation_id, cycle_index, stage_result, trace)
                tool_ids.append(tool_id)
                nodes.append(self._tool_node(stage_result, cycle_index, trace, tool_id))
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.CALLED_TOOL,
                        source=execution_id,
                        target=tool_id,
                        label="called tool",
                        properties={"step": trace.step, "tool_name": trace.tool_name},
                    )
                )

            nodes.append(self._stage_result_node(stage_result, cycle_index, result_id, tool_ids))
            edges.append(
                self._edge(
                    edge_type=AttackProcessEdgeType.PRODUCED_RESULT,
                    source=execution_id,
                    target=result_id,
                    label="produced result",
                )
            )
            for tool_id in tool_ids:
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.PRODUCED_RESULT,
                        source=tool_id,
                        target=result_id,
                        label="produced result",
                    )
                )

            if stage_result.handoff_suggestion is not None:
                handoff_id = self._handoff_id(stage_result)
                nodes.append(self._handoff_node(stage_result, cycle_index, handoff_id))
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.SUGGESTED_HANDOFF,
                        source=result_id,
                        target=handoff_id,
                        label="suggested handoff",
                    )
                )

            goal_node = self._goal_check_node(stage_result, cycle_index)
            if goal_node is not None:
                nodes.append(goal_node)
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.SATISFIED_GOAL
                        if stage_result.runtime_hints.get("goal_satisfied") is True
                        else AttackProcessEdgeType.PRODUCED_RESULT,
                        source=result_id,
                        target=goal_node.id,
                        label="goal check",
                    )
                )

            if stage_result.status == "blocked":
                blocked_id = stable_node_id(
                    "blocked-reason",
                    {
                        "operation_id": operation_id,
                        "cycle_index": cycle_index,
                        "stage_result_id": stage_result.result_id,
                        "summary": stage_result.summary,
                    },
                )
                nodes.append(
                    BlockedReasonNode(
                        id=blocked_id,
                        label="Stage blocked",
                        operation_id=operation_id,
                        cycle_index=cycle_index,
                        agent_name=stage_result.agent_name,
                        stage_type=stage_result.stage_type,
                        status="blocked",
                        summary=stage_result.summary,
                        evidence_refs=list(stage_result.evidence_refs),
                        properties={"stage_result_id": stage_result.result_id},
                    )
                )
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.BLOCKED_BY_POLICY,
                        source=result_id,
                        target=blocked_id,
                        label="blocked by policy",
                    )
                )

        for index, event in enumerate(policy_events or []):
            if self._policy_event_blocked(event):
                blocked_id = stable_node_id(
                    "blocked-reason",
                    {
                        "operation_id": operation_id,
                        "cycle_index": cycle_index,
                        "policy_event_index": index,
                        "reason": event.get("reason") or event.get("summary") or event.get("event_type"),
                    },
                )
                nodes.append(
                    BlockedReasonNode(
                        id=blocked_id,
                        label="Policy blocked action",
                        operation_id=operation_id,
                        cycle_index=cycle_index,
                        status="blocked",
                        summary=self._truncate(str(event.get("reason") or event.get("summary") or "policy blocked action")),
                        evidence_refs=self._event_refs(event),
                        properties=self._sanitize_event(event),
                    )
                )
                edges.append(
                    self._edge(
                        edge_type=AttackProcessEdgeType.BLOCKED_BY_POLICY,
                        source=execution_id or planner_id or cycle_id,
                        target=blocked_id,
                        label="blocked by policy",
                    )
                )

        unique_nodes = self._unique_nodes(nodes)
        return AttackLogExtraction(
            operation_id=operation_id,
            cycle_index=cycle_index,
            ag_nodes=unique_nodes,
            ag_edges=self._unique_edges(edges, {node.id for node in unique_nodes}),
            summary=self._summary(decision, stage_result, traces),
            evidence_refs=list(evidence_refs),
            metadata={
                "planner_decision_included": decision is not None,
                "stage_result_included": stage_result is not None,
                "tool_trace_count": len(traces),
                "runtime_event_count": len(runtime_events or []),
                "policy_event_count": len(policy_events or []),
            },
        )

    @staticmethod
    def _cycle_id(operation_id: str, cycle_index: int) -> str:
        return f"attack-cycle::{operation_id}::{cycle_index}"

    @staticmethod
    def _planner_id(decision: PlannerDecision) -> str:
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
    def _execution_id(operation_id: str, cycle_index: int, agent_name: str) -> str:
        return f"agent-execution::{operation_id}::{cycle_index}::{agent_name}"

    @staticmethod
    def _stage_result_id(stage_result: StageResult) -> str:
        return f"stage-result::{stage_result.result_id}"

    @staticmethod
    def _handoff_id(stage_result: StageResult) -> str:
        handoff = stage_result.handoff_suggestion
        return stable_node_id(
            "handoff",
            {
                "stage_result_id": stage_result.result_id,
                "suggested_agent": handoff.suggested_agent if handoff is not None else None,
                "suggested_stage": handoff.suggested_stage if handoff is not None else None,
            },
        )

    @staticmethod
    def _stop_decision_id(decision: PlannerDecision) -> str:
        return stable_node_id(
            "stop-decision",
            {
                "operation_id": decision.operation_id,
                "cycle_index": decision.cycle_index,
                "decision": decision.decision,
                "stop_condition": decision.stop_condition,
            },
        )

    @staticmethod
    def _tool_id(operation_id: str, cycle_index: int, stage_result: StageResult, trace: ToolTrace) -> str:
        return stable_node_id(
            "tool-call",
            {
                "operation_id": operation_id,
                "cycle_index": cycle_index,
                "stage_result_id": stage_result.result_id,
                "stage_task_id": stage_result.stage_task_id,
                "step": trace.step,
                "server_id": trace.server_id,
                "tool_name": trace.tool_name,
                "raw_output_ref": trace.raw_output_ref,
            },
        )

    @staticmethod
    def _cycle_from_stage_result(stage_result: StageResult) -> int:
        raw = stage_result.runtime_hints.get("cycle_index") or stage_result.writeback_hints.get("cycle_index")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _planner_node(decision: PlannerDecision, node_id: str) -> PlannerDecisionNode:
        return PlannerDecisionNode(
            id=node_id,
            label=f"Planner decision: {decision.decision}",
            operation_id=decision.operation_id,
            cycle_index=decision.cycle_index,
            agent_name="planner_agent",
            stage_type=decision.selected_stage,
            status=decision.decision,
            summary=decision.reasoning_summary or decision.objective,
            refs=list(decision.target_refs),
            properties={
                "node_role": "PLANNER_DECISION",
                "display_name": f"规划决策：{decision.selected_stage or decision.decision}",
                "visual_title": AttackLogExtractor._planner_visual_title(decision),
                "visual_subtitle": AttackLogExtractor._target_summary(decision.target_refs),
                "visual_summary": decision.reasoning_summary or decision.objective,
                "visual_target": AttackLogExtractor._target_summary(decision.target_refs),
                "visual_outcome": decision.decision,
                "cycle_index": decision.cycle_index,
                "step_order": 2,
                "decision": decision.decision,
                "selected_agent": decision.selected_agent,
                "selected_stage": decision.selected_stage,
                "objective": decision.objective,
                "reasoning_summary": decision.reasoning_summary,
                "required_context": decision.required_context,
                "success_criteria": list(decision.success_criteria),
                "risk_level": decision.risk_level,
                "max_steps": decision.max_steps,
                "handoff_acceptance": decision.handoff_acceptance,
                "stop_condition": decision.stop_condition,
                "confidence": decision.confidence,
                "metadata": decision.metadata,
            },
        )

    @staticmethod
    def _stop_decision_node(decision: PlannerDecision, node_id: str) -> StopDecisionNode:
        return StopDecisionNode(
            id=node_id,
            label=f"Planner stop: {decision.decision}",
            operation_id=decision.operation_id,
            cycle_index=decision.cycle_index,
            agent_name="planner_agent",
            stage_type=decision.selected_stage,
            status=decision.decision,
            summary=decision.stop_condition or decision.reasoning_summary or decision.objective,
            evidence_refs=[str(ref) for ref in decision.metadata.get("evidence_refs", [])] if isinstance(decision.metadata.get("evidence_refs"), list) else [],
            properties={
                "decision": decision.decision,
                "stop_condition": decision.stop_condition,
                "confidence": decision.confidence,
                "metadata": decision.metadata,
            },
        )

    @staticmethod
    def _execution_node_from_decision(
        decision: PlannerDecision,
        node_id: str,
        planner_id: str,
    ) -> AgentExecutionNode:
        return AgentExecutionNode(
            id=node_id,
            label=f"{decision.selected_agent} execution",
            operation_id=decision.operation_id,
            cycle_index=decision.cycle_index,
            agent_name=decision.selected_agent,
            stage_type=decision.selected_stage,
            status="planned",
            summary=decision.objective,
            refs=list(decision.target_refs),
            properties={
                "node_role": "AGENT_EXECUTION",
                "display_name": f"执行 Agent：{decision.selected_agent}",
                "visual_title": f"{decision.selected_stage or 'Stage'}：{decision.objective}",
                "visual_subtitle": f"{decision.selected_agent} / {AttackLogExtractor._target_summary(decision.target_refs) or 'target pending'}",
                "visual_summary": decision.reasoning_summary or decision.objective,
                "visual_target": AttackLogExtractor._target_summary(decision.target_refs),
                "visual_outcome": "planned",
                "cycle_index": decision.cycle_index,
                "step_order": 3,
                "planner_decision_id": planner_id,
                "selected_agent": decision.selected_agent,
                "selected_stage": decision.selected_stage,
                "agent_name": decision.selected_agent,
                "stage_type": decision.selected_stage,
            },
        )

    @classmethod
    def _execution_node_from_stage(cls, stage_result: StageResult, cycle_index: int, node_id: str) -> AgentExecutionNode:
        return AgentExecutionNode(
            id=node_id,
            label=f"{stage_result.agent_name} execution",
            operation_id=stage_result.operation_id,
            cycle_index=cycle_index,
            agent_name=stage_result.agent_name,
            stage_type=stage_result.stage_type,
            status=stage_result.status,
            summary=stage_result.summary,
            properties={
                "node_role": "AGENT_EXECUTION",
                "display_name": f"执行 Agent：{stage_result.agent_name}",
                "visual_title": f"{stage_result.stage_type}：{stage_result.summary}",
                "visual_subtitle": stage_result.agent_name,
                "visual_summary": stage_result.summary,
                "visual_target": cls._stage_visual_target(stage_result),
                "visual_outcome": stage_result.status,
                "cycle_index": cycle_index,
                "step_order": 3,
                "stage_task_id": stage_result.stage_task_id,
                "stage_result_id": stage_result.result_id,
                "agent_name": stage_result.agent_name,
                "stage_type": stage_result.stage_type,
            },
        )

    @classmethod
    def _tool_node(
        cls,
        stage_result: StageResult,
        cycle_index: int,
        trace: ToolTrace,
        node_id: str,
    ) -> ToolCallNode:
        return ToolCallNode(
            id=node_id,
            label=trace.tool_name,
            operation_id=stage_result.operation_id,
            cycle_index=cycle_index,
            agent_name=stage_result.agent_name,
            stage_type=stage_result.stage_type,
            status="succeeded" if trace.success else "failed",
            summary=cls._tool_summary(trace),
            evidence_refs=list(trace.evidence_refs),
            properties={
                "node_role": "TOOL_CALL",
                "display_name": f"工具调用：{trace.tool_name}",
                "visual_title": cls._tool_visual_title(stage_result, trace),
                "visual_subtitle": cls._tool_visual_subtitle(stage_result, trace),
                "visual_summary": cls._tool_summary(trace),
                "visual_target": cls._guess_target_from_trace(trace),
                "visual_outcome": "succeeded" if trace.success else "failed",
                "cycle_index": cycle_index,
                "step_order": 4,
                "trace_id": trace.trace_id,
                "step": trace.step,
                "server_id": trace.server_id,
                "tool_name": trace.tool_name,
                "tool_category": trace.tool_category,
                "input_summary": cls._truncate(trace.input_summary),
                "raw_output_ref": trace.raw_output_ref,
                "output_summary": cls._tool_summary(trace),
                "stdout_chars": len(trace.stdout or ""),
                "stderr_chars": len(trace.stderr or ""),
                "parsed_output_summary": cls._parsed_output_summary(trace.parsed_output),
                "argument_keys": sorted(str(key) for key in trace.arguments.keys()),
                "success": trace.success,
                "exit_code": trace.exit_code,
                "policy_original_allowed": cls._policy_original_allowed(trace.policy_check),
                "policy_original_reason": cls._policy_original_reason(trace.policy_check),
                "started_at": trace.started_at,
                "ended_at": trace.ended_at,
                "policy_check": cls._sanitize_event(trace.policy_check),
                "metadata": cls._sanitize_event(trace.metadata),
            },
        )

    @classmethod
    def _stage_result_node(
        cls,
        stage_result: StageResult,
        cycle_index: int,
        node_id: str,
        tool_ids: list[str],
    ) -> StageResultNode:
        return StageResultNode(
            id=node_id,
            label=f"{stage_result.stage_type} result",
            operation_id=stage_result.operation_id,
            cycle_index=cycle_index,
            agent_name=stage_result.agent_name,
            stage_type=stage_result.stage_type,
            status=stage_result.status,
            summary=stage_result.summary,
            evidence_refs=list(stage_result.evidence_refs),
            properties={
                "node_role": "STAGE_RESULT",
                "display_name": f"阶段结果：{stage_result.status}",
                "visual_title": f"{stage_result.stage_type} {stage_result.status}",
                "visual_subtitle": cls._stage_visual_target(stage_result),
                "visual_summary": stage_result.summary,
                "visual_target": cls._stage_visual_target(stage_result),
                "visual_outcome": stage_result.status,
                "visual_counts": {
                    "observations": len(stage_result.observations),
                    "evidence": len(stage_result.evidence),
                    "findings": len(stage_result.findings),
                    "relations": len(stage_result.discovered_relations),
                },
                "cycle_index": cycle_index,
                "step_order": 5,
                "result_id": stage_result.result_id,
                "stage_task_id": stage_result.stage_task_id,
                "status": stage_result.status,
                "summary": stage_result.summary,
                "evidence_refs": list(stage_result.evidence_refs),
                "observation_count": len(stage_result.observations),
                "evidence_count": len(stage_result.evidence),
                "finding_count": len(stage_result.findings),
                "discovered_entity_count": len(stage_result.discovered_entities),
                "discovered_relation_count": len(stage_result.discovered_relations),
                "tool_call_node_ids": list(tool_ids),
                "confidence": stage_result.confidence,
                "risk_level": stage_result.risk_level,
                "policy_notes": list(stage_result.policy_notes),
                "retry_recommendation": stage_result.retry_recommendation,
                "replan_recommendation": stage_result.replan_recommendation,
                "runtime_hints": cls._sanitize_event(stage_result.runtime_hints),
                "writeback_hints": cls._sanitize_event(stage_result.writeback_hints),
                "created_at": stage_result.created_at,
            },
        )

    @staticmethod
    def _handoff_node(stage_result: StageResult, cycle_index: int, node_id: str) -> HandoffSuggestionNode:
        handoff = stage_result.handoff_suggestion
        assert handoff is not None
        return HandoffSuggestionNode(
            id=node_id,
            label=f"Handoff to {handoff.suggested_agent}",
            operation_id=stage_result.operation_id,
            cycle_index=cycle_index,
            agent_name=stage_result.agent_name,
            stage_type=stage_result.stage_type,
            status="suggested",
            summary=handoff.reason,
            evidence_refs=[str(ref) for ref in handoff.required_context_refs],
            properties={
                **handoff.model_dump(mode="json"),
                "node_role": "HANDOFF_SUGGESTION",
                "display_name": f"下一步建议：{handoff.suggested_stage}",
                "visual_title": f"下一步：{handoff.suggested_stage}",
                "visual_subtitle": handoff.suggested_agent,
                "visual_summary": handoff.reason,
                "visual_outcome": "suggested",
                "cycle_index": cycle_index,
                "step_order": 6,
                "suggested_stage": handoff.suggested_stage,
                "suggested_agent": handoff.suggested_agent,
                "reason": handoff.reason,
            },
        )

    @classmethod
    def _goal_check_node(cls, stage_result: StageResult, cycle_index: int) -> GoalCheckNode | None:
        goal_finding = None
        for finding in stage_result.findings:
            if isinstance(finding, dict) and str(finding.get("kind") or finding.get("type")) in {
                "GoalCheck",
                "GoalNotSatisfied",
                "GoalBlocked",
                "GoalNeedsMoreEvidence",
            }:
                goal_finding = finding
                break
        if stage_result.stage_type != "GOAL_STAGE" and goal_finding is None and "goal_satisfied" not in stage_result.runtime_hints:
            return None
        satisfied = bool(stage_result.runtime_hints.get("goal_satisfied")) if "goal_satisfied" in stage_result.runtime_hints else bool((goal_finding or {}).get("goal_satisfied"))
        node_id = stable_node_id(
            "goal-check",
            {
                "operation_id": stage_result.operation_id,
                "cycle_index": cycle_index,
                "stage_result_id": stage_result.result_id,
                "goal_satisfied": satisfied,
            },
        )
        evidence_refs = list(stage_result.evidence_refs)
        if isinstance(stage_result.runtime_hints.get("goal_evidence_refs"), list):
            evidence_refs.extend(str(item) for item in stage_result.runtime_hints["goal_evidence_refs"] if item)
        return GoalCheckNode(
            id=node_id,
            label="Goal check",
            operation_id=stage_result.operation_id,
            cycle_index=cycle_index,
            agent_name=stage_result.agent_name,
            stage_type=stage_result.stage_type,
            status="satisfied" if satisfied else "not_satisfied",
            summary=str(stage_result.runtime_hints.get("goal_summary") or stage_result.summary),
            evidence_refs=sorted(set(evidence_refs)),
            properties={
                "goal_satisfied": satisfied,
                "finding": cls._sanitize_event(goal_finding or {}),
                "runtime_hints": cls._sanitize_event(stage_result.runtime_hints),
            },
        )

    @staticmethod
    def _edge(
        *,
        edge_type: AttackProcessEdgeType,
        source: str,
        target: str,
        label: str,
        properties: dict[str, Any] | None = None,
    ) -> AttackProcessEdge:
        return AttackProcessEdge(
            id=stable_node_id(
                "edge",
                {"edge_type": edge_type.value, "source": source, "target": target, "label": label},
            ),
            edge_type=edge_type,
            source=source,
            target=target,
            label=label,
            properties=properties or {},
        )

    @classmethod
    def _tool_summary(cls, trace: ToolTrace) -> str:
        if trace.summary:
            return cls._truncate(trace.summary)
        parts = []
        if trace.stdout:
            parts.append(f"stdout {len(trace.stdout)} chars")
        if trace.stderr:
            parts.append(f"stderr {len(trace.stderr)} chars")
        if parts:
            return "; ".join(parts)
        return trace.input_summary or trace.tool_name

    @staticmethod
    def _parsed_output_summary(parsed_output: dict[str, Any]) -> dict[str, Any]:
        if not parsed_output:
            return {}
        return {
            "keys": sorted(str(key) for key in parsed_output.keys())[:20],
            "item_count": len(parsed_output),
        }

    @staticmethod
    def _policy_original_allowed(policy_check: dict[str, Any]) -> bool | None:
        metadata = policy_check.get("metadata") if isinstance(policy_check, dict) else None
        if isinstance(metadata, dict) and "original_allowed" in metadata:
            return bool(metadata.get("original_allowed"))
        if isinstance(policy_check, dict) and "original_allowed" in policy_check:
            return bool(policy_check.get("original_allowed"))
        if isinstance(policy_check, dict) and "allowed" in policy_check:
            return bool(policy_check.get("allowed"))
        return None

    @staticmethod
    def _policy_original_reason(policy_check: dict[str, Any]) -> str | None:
        metadata = policy_check.get("metadata") if isinstance(policy_check, dict) else None
        if isinstance(metadata, dict) and metadata.get("original_reason") is not None:
            return str(metadata.get("original_reason"))
        if isinstance(policy_check, dict) and policy_check.get("original_reason") is not None:
            return str(policy_check.get("original_reason"))
        if isinstance(policy_check, dict) and policy_check.get("reason") is not None:
            return str(policy_check.get("reason"))
        return None

    @classmethod
    def _collect_evidence_refs(
        cls,
        stage_result: StageResult | None,
        traces: list[ToolTrace],
        runtime_events: list[dict[str, Any]],
        policy_events: list[dict[str, Any]],
    ) -> list[str]:
        refs: list[str] = []
        if stage_result is not None:
            refs.extend(str(ref) for ref in stage_result.evidence_refs)
            for record in stage_result.evidence:
                refs.extend(cls._event_refs(record))
        for trace in traces:
            refs.extend(str(ref) for ref in trace.evidence_refs)
            if trace.raw_output_ref:
                refs.append(trace.raw_output_ref)
        for event in [*runtime_events, *policy_events]:
            refs.extend(cls._event_refs(event))
        return sorted({ref for ref in refs if ref})

    @staticmethod
    def _event_refs(event: dict[str, Any]) -> list[str]:
        refs: list[str] = []
        for key in ("evidence_ref", "evidence_refs", "artifact_ref", "artifact_refs", "payload_ref", "raw_output_ref"):
            value = event.get(key)
            if isinstance(value, list):
                refs.extend(str(item) for item in value if item)
            elif value:
                refs.append(str(value))
        return refs

    @classmethod
    def _sanitize_event(cls, event: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in event.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(part in lowered for part in ("stdout", "stderr", "raw", "body", "content", "secret", "token", "password")):
                if lowered.endswith("_ref") or lowered.endswith("ref"):
                    result[key_text] = cls._sanitize_value(value)
                continue
            result[key_text] = cls._sanitize_value(value)
        return result

    @classmethod
    def _sanitize_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._truncate(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, GraphRef):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [cls._sanitize_value(item) for item in value[:20]]
        if isinstance(value, dict):
            return cls._sanitize_event(value)
        return cls._truncate(str(value))

    @staticmethod
    def _truncate(value: str, limit: int = 240) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: max(0, limit - 15)]}...[truncated]"

    @staticmethod
    def _policy_event_blocked(event: dict[str, Any]) -> bool:
        metadata = event.get("metadata")
        if isinstance(metadata, dict) and metadata.get("policy_audit_only") is True:
            return False
        if event.get("policy_audit_only") is True or event.get("final_allowed") is True:
            return False
        if event.get("allowed") is False:
            return True
        status = str(event.get("status") or event.get("decision") or event.get("result") or "").lower()
        return status in {"blocked", "denied", "rejected"}

    @staticmethod
    def _dedupe_traces(traces: list[ToolTrace]) -> list[ToolTrace]:
        result: list[ToolTrace] = []
        seen: set[tuple[Any, ...]] = set()
        for trace in traces:
            key = (trace.step, trace.server_id, trace.tool_name, trace.raw_output_ref, trace.trace_id)
            if key in seen:
                continue
            seen.add(key)
            result.append(trace)
        return result

    @staticmethod
    def _unique_nodes(nodes: list[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for node in nodes:
            if node.id in seen:
                continue
            seen.add(node.id)
            result.append(node)
        return result

    @staticmethod
    def _unique_edges(edges: list[AttackProcessEdge], node_ids: set[str]) -> list[AttackProcessEdge]:
        result: list[AttackProcessEdge] = []
        seen: set[str] = set()
        for edge in edges:
            if edge.id in seen or edge.source not in node_ids or edge.target not in node_ids:
                continue
            seen.add(edge.id)
            result.append(edge)
        return result

    @staticmethod
    def _summary(
        decision: PlannerDecision | None,
        stage_result: StageResult | None,
        traces: list[ToolTrace],
    ) -> str:
        if stage_result is not None:
            return f"{stage_result.agent_name} {stage_result.status}: {stage_result.summary} ({len(traces)} tool call(s))"
        if decision is not None:
            return f"planner {decision.decision}: {decision.objective}"
        return "attack log extraction"

    @classmethod
    def _cycle_visual_title(
        cls,
        cycle_index: int,
        decision: PlannerDecision | None,
        stage_result: StageResult | None,
    ) -> str:
        stage = (stage_result.stage_type if stage_result is not None else decision.selected_stage if decision is not None else "") or "planning"
        status = stage_result.status if stage_result is not None else decision.decision if decision is not None else "planned"
        target = cls._target_summary(decision.target_refs) if decision is not None else cls._stage_visual_target(stage_result)
        parts = [f"Cycle {cycle_index}", str(stage), str(status)]
        if target:
            parts.append(target)
        return " | ".join(parts)

    @classmethod
    def _cycle_visual_summary(
        cls,
        decision: PlannerDecision | None,
        stage_result: StageResult | None,
        traces: list[ToolTrace],
    ) -> str:
        if stage_result is not None:
            return f"{stage_result.summary} ({len(traces)} tool call(s))"
        if decision is not None:
            return decision.reasoning_summary or decision.objective
        return ""

    @classmethod
    def _planner_visual_title(cls, decision: PlannerDecision) -> str:
        target = cls._target_summary(decision.target_refs)
        action = decision.selected_stage or decision.decision
        if target:
            return f"规划：{action}，目标 {target}"
        return f"规划：{action}"

    @staticmethod
    def _target_summary(refs: list[GraphRef]) -> str:
        labels: list[str] = []
        for ref in refs[:3]:
            labels.append(str(ref.ref_id))
        return ", ".join(labels)

    @classmethod
    def _tool_visual_title(cls, stage_result: StageResult, trace: ToolTrace) -> str:
        target = cls._guess_target_from_trace(trace) or cls._stage_visual_target(stage_result)
        outcome = "成功" if trace.success else "失败"
        if target:
            return f"{trace.tool_name} {outcome}：{target}"
        return f"{trace.tool_name} {outcome}"

    @classmethod
    def _tool_visual_subtitle(cls, stage_result: StageResult, trace: ToolTrace) -> str:
        status = "succeeded" if trace.success else "failed"
        summary = cls._tool_summary(trace)
        return f"{stage_result.stage_type} / {status} / {summary}" if summary else f"{stage_result.stage_type} / {status}"

    @staticmethod
    def _guess_target_from_trace(trace: ToolTrace) -> str:
        for key in ("target", "host", "url", "address", "endpoint", "service", "asset"):
            value = trace.arguments.get(key)
            if value:
                return str(value)
        parsed_target = trace.parsed_output.get("target") or trace.parsed_output.get("host") or trace.parsed_output.get("url")
        return str(parsed_target) if parsed_target else ""

    @staticmethod
    def _stage_visual_target(stage_result: StageResult | None) -> str:
        if stage_result is None:
            return ""
        for bucket in (
            stage_result.runtime_hints,
            stage_result.writeback_hints,
            *(stage_result.observations[:3]),
            *(stage_result.evidence[:3]),
            *(stage_result.findings[:3]),
        ):
            if not isinstance(bucket, dict):
                continue
            for key in ("target", "host", "url", "address", "endpoint", "service_id", "asset_id"):
                value = bucket.get(key)
                if value:
                    return str(value)
        return ""


__all__ = ["AttackLogExtraction", "AttackLogExtractor"]
