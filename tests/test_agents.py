from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.agents.agent_models import DecisionRecord, ObservationRecord, OutcomeRecord
from src.core.agents.agent_pipeline import AgentPipeline, PipelineStepResult
from src.core.agents.agent_protocol import (
    AgentContext,
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.agents.critic import CriticAgent, CriticLLMReview
from src.core.agents.graph_projection import GraphProjectionAgent
from src.core.agents.kg_events import KGDeltaEvent, KGDeltaEventType
from src.core.agents.perception import PerceptionAgent, PerceptionNormalizationAdvice
from src.core.agents.planner import PlannerAgent, PlannerLLMAdvice
from src.core.agents.registry import AgentRegistry
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.state_writer import StateWriterAgent
from src.core.agents.task_builder import TaskBuilderAgent
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import AttackGraph, GraphRef as AGGraphRef
from src.core.models.events import AgentRole
from src.core.models.kg import DataAsset, Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.models.runtime import OperationRuntime, RuntimeState, WorkerRuntime, WorkerStatus
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerTaskSpec


class IllegalStructuralWorker(BaseWorkerAgent):
    def __init__(self) -> None:
        super().__init__(name="illegal_worker")

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return True

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        return AgentOutput(
            state_deltas=[
                {
                    "scope": GraphScope.KG.value,
                    "write_type": "structural",
                    "target_ref": GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host").model_dump(mode="json"),
                    "patch": {"id": "host-1"},
                }
            ]
        )


class PipelineWorker(BaseWorkerAgent):
    def __init__(self) -> None:
        super().__init__(name="test_worker")
        self._supported = {
            TaskType.SERVICE_VALIDATION.value,
            TaskType.ASSET_CONFIRMATION.value,
            TaskType.GOAL_CONDITION_VALIDATION.value,
            "service_validation",
        }

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self._supported

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        raw_result = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="validation_result",
            summary=f"validated {task_spec.task_type}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}",
            refs=task_spec.target_refs,
            extra={"validation_status": "validated"},
        )
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type="validation_result",
            success=True,
            summary=f"validated {task_spec.task_type}",
            raw_result_ref=raw_result["payload_ref"],
            confidence=0.9,
            refs=task_spec.target_refs,
            payload={"task_type": task_spec.task_type},
        )
        return AgentOutput(
            outcomes=[outcome.to_agent_output_fragment()],
            evidence=[raw_result | {"validation_status": "validated", "confidence": 0.9}],
            logs=[f"worker handled {task_spec.task_type}"],
        )


class PlannerTestLLMAdvisor:
    def advise(self, *, graph, goal_ref, candidates, planning_context):  # noqa: ANN001
        del graph, goal_ref, planning_context
        if not candidates:
            return []
        return [
            PlannerLLMAdvice(
                candidate_id=candidates[0].candidate_id,
                score_delta=0.15,
                rationale_suffix="llm 建议该候选更适合当前目标",
                metadata={"reason": "goal_alignment"},
            )
        ]


class RejectedPlannerTestLLMAdvisor:
    def advise(self, *, graph, goal_ref, candidates, planning_context):  # noqa: ANN001
        del graph, goal_ref, planning_context
        if not candidates:
            return []
        return [
            PlannerLLMAdvice(
                candidate_id=candidates[0].candidate_id,
                score_delta=0.5,
                rationale_suffix="should be rejected",
                metadata={"reason": "too_much_control"},
            ),
            PlannerLLMAdvice(
                candidate_id="unknown-candidate",
                score_delta=0.1,
                rationale_suffix="unknown target",
            ),
        ]


class CriticTestLLMAdvisor:
    def summarize_findings(self, *, findings, context, runtime_state):  # noqa: ANN001
        del context, runtime_state
        if not findings:
            return []
        return [
            CriticLLMReview(
                finding_id=findings[0].finding_id,
                summary_override=f"{findings[0].summary} (llm归纳)",
                rationale_suffix="llm 归纳为上游依赖失效导致的持续阻塞",
                metadata={"category": "failure_summary"},
            )
        ]


class RejectedCriticTestLLMAdvisor:
    def summarize_findings(self, *, findings, context, runtime_state):  # noqa: ANN001
        del context, runtime_state
        if not findings:
            return []
        return [
            CriticLLMReview(
                finding_id=findings[0].finding_id,
                summary_override="should be rejected",
                metadata={"tool_command": "nmap -A target"},
            ),
            CriticLLMReview(
                finding_id="unknown-finding",
                rationale_suffix="unknown target",
            ),
        ]


class PerceptionTestLLMNormalizer:
    def normalize(self, *, outcome, raw_result, refs):  # noqa: ANN001
        del outcome, refs
        return PerceptionNormalizationAdvice(
            normalized_result={
                **dict(raw_result),
                "observation_summary": "llm 归一化后的观察摘要",
                "evidence_kind": "normalized_tool_output",
            },
            summary_suffix="llm 已补齐结构化字段",
            metadata={"normalizer": "test"},
        )


def build_agent_context(operation_id: str = "op-1") -> AgentContext:
    return AgentContext(operation_id=operation_id, runtime_state_ref="runtime-1")


def build_agent_graph_refs() -> list[GraphRef]:
    return [
        GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
        GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph"),
    ]


def build_goal_focused_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(Service(id="svc-1", label="SSH", confidence=0.8))
    kg.add_node(DataAsset(id="asset-1", label="Objective Data", confidence=0.85))
    kg.add_node(Goal(id="goal-1", label="Validate Objective", category="data", confidence=0.9))
    kg.add_edge(HostsEdge(id="edge-host-svc", label="hosts", source="host-1", target="svc-1"))
    kg.add_edge(TargetsEdge(id="edge-goal-asset", label="targets", source="goal-1", target="asset-1"))
    return kg


def build_attack_graph() -> AttackGraph:
    return AttackGraphProjector().project(build_goal_focused_kg())


def build_planner_input() -> AgentInput:
    ag = build_attack_graph()
    goal_node = ag.get_goal_nodes()[0]
    return AgentInput(
        graph_refs=[
            GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
            GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode"),
        ],
        context=build_agent_context(),
        raw_payload={
            "ag_graph": ag.to_dict(),
            "goal_refs": [
                GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode").model_dump(mode="json")
            ],
            "planning_context": {"top_k": 1, "max_depth": 2},
        },
    )


def build_task_candidate() -> TaskCandidate:
    return TaskCandidate(
        source_action_id="action-service-1",
        task_type=TaskType.SERVICE_VALIDATION,
        input_bindings={"host_id": "host-1"},
        target_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        expected_output_refs=[AGGraphRef(graph="query", ref_id="task-output::svc", ref_type="TaskOutput")],
        estimated_cost=0.2,
        estimated_risk=0.1,
        estimated_noise=0.1,
        goal_relevance=0.9,
        resource_keys={"host:host-1"},
        parallelizable=True,
    )


def build_task_builder_input() -> AgentInput:
    decision = DecisionRecord(
        source_agent="planner_agent",
        summary="selected plan",
        confidence=0.9,
        refs=[],
        payload={"planning_candidate": {"action_ids": ["action-service-1"], "task_candidates": [build_task_candidate().model_dump(mode="json")]}},
        decision_type="plan_selection",
        score=0.9,
        target_refs=[GraphRef(graph=GraphScope.AG, ref_id="action-service-1", ref_type="ActionNode")],
        rationale="test rationale",
    )
    return AgentInput(
        graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
        decision_ref=decision.id,
        context=build_agent_context(),
        raw_payload={"decision": decision.model_dump(mode="json"), "task_candidates": [build_task_candidate().model_dump(mode="json")]},
    )


def build_ready_task_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-1",
            label="Service validation",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.READY,
            source_action_id="action-service-1",
            input_bindings={"host_id": "host-1"},
            target_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            source_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            expected_output_refs=[AGGraphRef(graph="query", ref_id="task-output::task-1", ref_type="TaskOutput")],
            estimated_cost=0.2,
            estimated_risk=0.1,
            estimated_noise=0.1,
            goal_relevance=0.9,
            resource_keys={"host:host-1"},
            parallelizable=True,
        )
    )
    return graph


def build_runtime_state() -> RuntimeState:
    runtime = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    runtime.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    return runtime


def apply_tg_deltas_to_graph(state_deltas: list[dict[str, object]]) -> TaskGraph:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    for delta in state_deltas:
        patch = dict(delta["patch"])
        if "node" in patch:
            nodes.append(dict(patch["node"]))
        if "edge" in patch:
            edges.append(dict(patch["edge"]))
    return TaskGraph.from_dict({"nodes": nodes, "edges": edges})


def make_worker_step(worker: PipelineWorker) -> PipelineStepResult:
    agent_input = AgentInput(
        graph_refs=build_agent_graph_refs(),
        task_ref="task-1",
        context=build_agent_context(),
        raw_payload={
            "task_id": "task-1",
            "task_type": TaskType.SERVICE_VALIDATION.value,
            "input_bindings": {"host_id": "host-1"},
            "resource_keys": ["host:host-1"],
        },
    )
    execution = worker.run(agent_input)
    return PipelineStepResult.from_execution(step_name="worker[1]", agent_input=agent_input, execution=execution)


def test_pipeline_worker_task_results_are_canonical_agent_task_results() -> None:
    worker = PipelineWorker()
    pipeline = AgentPipeline(agents=[worker])

    results = pipeline.worker_task_results([make_worker_step(worker)])

    assert len(results) == 1
    assert results[0].task_id == "task-1"
    assert results[0].agent_role == AgentRole.RECON_WORKER
    assert results[0].metadata["adapted_from"] == "agent_output"
    assert results[0].evidence[0].payload_ref == "runtime://worker-results/task-1"


def test_worker_cannot_write_kg_ag_tg_structural_state() -> None:
    worker = IllegalStructuralWorker()
    result = worker.run(
        AgentInput(
            graph_refs=build_agent_graph_refs(),
            task_ref="task-1",
            context=build_agent_context(),
            raw_payload={"task_type": "service_validation"},
        )
    )

    assert result.success is False
    assert "may not emit KG/AG/TG" in result.output.errors[0]


def test_state_writer_only_writes_kg() -> None:
    agent = StateWriterAgent()
    observation = ObservationRecord(
        source_agent="perception_agent",
        summary="observed host",
        confidence=0.8,
        refs=[GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
        payload={"branch": "validation_result"},
    )
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"observations": [observation.model_dump(mode="json")]},
        )
    )

    assert result.success is True
    assert result.output.state_deltas
    assert {delta["scope"] for delta in result.output.state_deltas} == {"kg"}


def test_state_writer_extracts_structured_entities_and_relations() -> None:
    agent = StateWriterAgent()
    observation = ObservationRecord(
        source_agent="recon_worker",
        summary="observed multi-host facts",
        confidence=0.9,
        refs=[GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
        payload={
            "entities": [
                {"id": "subnet-1", "type": "NetworkZone", "label": "10.0.0.0/24", "cidr": "10.0.0.0/24", "zone_kind": "subnet"},
                {"id": "cred-1", "type": "Credential", "label": "Operator Cred", "credential_kind": "password"},
                {"id": "sess-1", "type": "Session", "label": "Pivot Session", "session_kind": "shell"},
            ],
            "relations": [
                {"type": "BELONGS_TO_ZONE", "source": "host-1", "target": "subnet-1"},
                {"type": "PIVOTS_TO", "source": "host-1", "target": "host-2"},
                {"type": "REUSES_CREDENTIAL", "source": "cred-1", "target": "host-2"},
            ],
        },
    )
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"observations": [observation.model_dump(mode="json")]},
        )
    )

    patches = [delta["patch"] for delta in result.output.state_deltas]
    entity_types = {patch.get("entity_type") for patch in patches if patch.get("entity_kind") == "node"}
    relation_types = {patch.get("relation_type") for patch in patches if patch.get("entity_kind") == "edge"}
    assert "NetworkZone" in entity_types
    assert "Credential" in entity_types
    assert "Session" in entity_types
    assert "PIVOTS_TO" in relation_types
    assert "REUSES_CREDENTIAL" in relation_types


def test_state_writer_can_apply_patch_batch_to_kg_store_with_version_tracking() -> None:
    agent = StateWriterAgent()
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", confidence=0.8))
    agent_input = AgentInput(
        graph_refs=[GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")],
        context=build_agent_context(),
        raw_payload={
            "kg_version": kg.version,
            "observations": [
                ObservationRecord(
                    source_agent="recon_worker",
                    summary="observed zone relation",
                    confidence=0.9,
                    refs=[GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
                    payload={
                        "entities": [
                            {
                                "id": "subnet-1",
                                "type": "NetworkZone",
                                "label": "10.0.0.0/24",
                                "cidr": "10.0.0.0/24",
                                "zone_kind": "subnet",
                            }
                        ],
                        "relations": [
                            {"type": "BELONGS_TO_ZONE", "source": "host-1", "target": "subnet-1"}
                        ],
                    },
                ).model_dump(mode="json")
            ],
        },
    )
    run_result = agent.run(agent_input)

    kg_ref = GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")
    apply_request = agent.build_store_apply_request(
        kg_ref=kg_ref,
        state_deltas=run_result.output.state_deltas,
        agent_input=agent_input,
        base_kg_version=kg.version,
    )
    apply_result = agent.apply_to_store(store=kg, apply_request=apply_request)

    assert apply_request.resulting_kg_version == kg.version
    assert apply_result["patch_batch_id"] == apply_request.patch_batch_id
    assert kg.version > 1
    assert kg.get_node("subnet-1").type.value == "NetworkZone"
    assert kg.get_edge("belongs_to_zone::host-1::subnet-1").type.value == "BELONGS_TO_ZONE"


def test_graph_projection_only_writes_ag() -> None:
    agent = GraphProjectionAgent()
    kg_event = KGDeltaEvent(
        event_type=KGDeltaEventType.ENTITY_ADDED,
        source_agent="state_writer_agent",
        target_ref=GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="host"),
        patch={"attributes": {"confidence": 0.9, "properties": {"host_id": "host-1"}}},
    )
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"kg_event_batch": {"events": [kg_event.model_dump(mode="json")]}}
        )
    )

    assert result.success is True
    assert result.output.state_deltas
    assert {delta["scope"] for delta in result.output.state_deltas} == {"ag"}


def test_graph_projection_emits_versioned_projection_event() -> None:
    agent = GraphProjectionAgent()
    kg_event = KGDeltaEvent(
        event_type=KGDeltaEventType.RELATION_UPDATED,
        source_agent="state_writer_agent",
        target_ref=GraphRef(graph=GraphScope.KG, ref_id="e-reach", ref_type="CAN_REACH"),
        patch={"relation_type": "CAN_REACH", "source": "host-1", "target": "host-2", "attributes": {"confidence": 0.9}},
        metadata={"resulting_kg_version": 7},
    )
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"kg_event_batch": {"events": [kg_event.model_dump(mode="json")]}}
        )
    )

    assert result.success is True
    assert result.output.emitted_events
    projection_event = result.output.emitted_events[0]
    assert projection_event["source_kg_version"] == 7
    assert projection_event["metadata"]["ag_version"] >= 1


def test_task_builder_only_writes_tg() -> None:
    agent = TaskBuilderAgent()
    result = agent.run(build_task_builder_input())

    assert result.success is True
    assert result.output.state_deltas
    assert {delta["scope"] for delta in result.output.state_deltas} == {"tg"}


def test_planner_outputs_decisions_without_execution_side_effects() -> None:
    agent = PlannerAgent()
    result = agent.run(build_planner_input())

    assert result.success is True
    assert result.output.decisions
    assert result.output.state_deltas == []
    assert result.output.outcomes == []


def test_planner_can_accept_llm_candidate_advice_without_dispatch_side_effects() -> None:
    agent = PlannerAgent(llm_advisor=PlannerTestLLMAdvisor())
    result = agent.run(build_planner_input())

    assert result.success is True
    assert result.output.decisions
    candidate = result.output.decisions[0]["payload"]["planning_candidate"]
    assert candidate["metadata"]["llm_advice"]["metadata"]["reason"] == "goal_alignment"
    assert "llm 建议" in result.output.decisions[0]["rationale"]
    assert result.output.state_deltas == []


def test_planner_rejects_invalid_llm_decisions_without_changing_candidate() -> None:
    agent = PlannerAgent(llm_advisor=RejectedPlannerTestLLMAdvisor())
    result = agent.run(build_planner_input())

    assert result.success is True
    assert result.output.decisions
    candidate = result.output.decisions[0]["payload"]["planning_candidate"]
    assert "llm_advice" not in candidate["metadata"]
    assert "should be rejected" not in result.output.decisions[0]["rationale"]
    assert any(
        "planner llm decision validation accepted=0 rejected=2" in log
        for log in result.output.logs
    )


def test_critic_emits_replan_request_without_kg_pollution() -> None:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-blocked",
            label="Blocked task",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.BLOCKED,
            source_action_id="action-1",
            target_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            goal_relevance=0.1,
            reason="upstream dependency failed",
        )
    )
    agent = CriticAgent()
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"tg_graph": graph.to_dict()},
        )
    )

    assert result.success is True
    assert result.output.replan_requests
    assert all(delta["scope"] in {"tg", "runtime"} for delta in result.output.state_deltas)


def test_critic_can_accept_llm_failure_summary_without_mutating_kg() -> None:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-blocked",
            label="Blocked task",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.BLOCKED,
            source_action_id="action-1",
            target_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            goal_relevance=0.1,
            reason="upstream dependency failed",
        )
    )
    agent = CriticAgent(llm_advisor=CriticTestLLMAdvisor())
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"tg_graph": graph.to_dict()},
        )
    )

    assert result.success is True
    assert result.output.decisions
    recommendation = result.output.decisions[0]["payload"]["recommendation"]
    assert "llm 归纳" in recommendation["rationale"]
    assert all(delta["scope"] in {"tg", "runtime"} for delta in result.output.state_deltas)


def test_critic_rejects_invalid_llm_reviews_without_changing_findings() -> None:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-blocked",
            label="Blocked task",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.BLOCKED,
            source_action_id="action-service-1",
            reason="upstream dependency failed",
        )
    )
    agent = CriticAgent(llm_advisor=RejectedCriticTestLLMAdvisor())
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"tg_graph": graph.to_dict()},
        )
    )

    assert result.success is True
    assert result.output.decisions
    assert all(
        "should be rejected" not in decision["payload"]["recommendation"]["rationale"]
        for decision in result.output.decisions
    )
    assert any(
        "critic llm decision validation accepted=0 rejected=2" in log
        for log in result.output.logs
    )


def test_perception_translates_outcome_into_observation_and_evidence() -> None:
    agent = PerceptionAgent()
    outcome = OutcomeRecord(
        source_agent="test_worker",
        task_id="task-1",
        outcome_type="validation_result",
        success=True,
        summary="validated host",
        raw_result_ref="runtime://worker-results/task-1",
        refs=[GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
    )
    result = agent.run(
        AgentInput(
            graph_refs=build_agent_graph_refs(),
            task_ref="task-1",
            context=build_agent_context(),
            raw_payload={
                "outcome": outcome.model_dump(mode="json"),
                "raw_result": {"summary": "validated host", "payload_ref": "runtime://worker-results/task-1"},
            },
        )
    )

    assert result.success is True
    assert len(result.output.observations) == 1
    assert len(result.output.evidence) == 1


def test_perception_can_use_llm_only_for_output_normalization() -> None:
    agent = PerceptionAgent(llm_normalizer=PerceptionTestLLMNormalizer())
    outcome = OutcomeRecord(
        source_agent="test_worker",
        task_id="task-1",
        outcome_type="validation_result",
        success=True,
        summary="validated host",
        raw_result_ref="runtime://worker-results/task-1",
        refs=[GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
    )
    result = agent.run(
        AgentInput(
            graph_refs=build_agent_graph_refs(),
            task_ref="task-1",
            context=build_agent_context(),
            raw_payload={
                "outcome": outcome.model_dump(mode="json"),
                "raw_result": {"summary": "validated host", "payload_ref": "runtime://worker-results/task-1"},
            },
        )
    )

    assert result.success is True
    assert result.output.observations[0]["summary"].startswith("llm 归一化后的观察摘要")
    assert result.output.evidence[0]["payload"]["evidence_kind"] == "normalized_tool_output"
    assert any("llm normalizer" in log for log in result.output.logs)


def test_scheduler_emits_assignment_for_ready_task() -> None:
    agent = SchedulerAgent()
    result = agent.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=build_agent_context(),
            raw_payload={"tg_graph": build_ready_task_graph().to_dict(), "runtime_state": build_runtime_state().model_dump(mode="json")},
        )
    )

    assert result.success is True
    assert any(decision["accepted"] for decision in result.output.decisions)


def test_agent_registry_dispatches_named_agent() -> None:
    registry = AgentRegistry()
    agent = PerceptionAgent()
    registry.register(agent)
    outcome = OutcomeRecord(
        source_agent="test_worker",
        task_id="task-1",
        outcome_type="validation_result",
        success=True,
        summary="validated",
    )

    result = registry.dispatch(
        agent.name,
        AgentInput(
            graph_refs=build_agent_graph_refs(),
            task_ref="task-1",
            context=build_agent_context(),
            raw_payload={"outcome": outcome.model_dump(mode="json")},
        ),
    )

    assert result.success is True
    assert result.agent_name == agent.name


def test_agent_pipeline_runs_minimal_single_round_cycle() -> None:
    planner = PlannerAgent()
    task_builder = TaskBuilderAgent()
    scheduler = SchedulerAgent()
    worker = PipelineWorker()
    perception = PerceptionAgent()
    state_writer = StateWriterAgent()
    projection = GraphProjectionAgent()
    critic = CriticAgent()
    pipeline = AgentPipeline(
        agents=[planner, task_builder, scheduler, worker, perception, state_writer, projection, critic]
    )

    planning = pipeline.run_planning_cycle(
        operation_id="op-1",
        graph_refs=[
            GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
            GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph"),
        ],
        planner_payload=build_planner_input().raw_payload,
    )
    assert planning.success is True

    tg_graph = apply_tg_deltas_to_graph(planning.final_output.state_deltas)
    execution = pipeline.run_execution_cycle(
        operation_id="op-1",
        graph_refs=build_agent_graph_refs(),
        scheduler_payload={"tg_graph": tg_graph.to_dict(), "runtime_state": build_runtime_state().model_dump(mode="json")},
        worker_agent=worker.name,
    )
    assert execution.success is True
    worker_steps = [step for step in execution.steps if step.agent_kind == AgentKind.WORKER]
    assert worker_steps

    feedback = pipeline.run_feedback_cycle(
        operation_id="op-1",
        graph_refs=build_agent_graph_refs(),
        worker_steps=worker_steps,
        feedback_payload={"kg_ref": GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph").model_dump(mode="json"), "tg_graph": tg_graph.to_dict()},
    )

    assert feedback.success is True
    assert feedback.final_output.observations
    assert feedback.final_output.state_deltas
