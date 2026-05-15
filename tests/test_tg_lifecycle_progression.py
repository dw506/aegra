from __future__ import annotations

from src.core.agents.agent_protocol import GraphRef as ProtocolGraphRef
from src.core.agents.agent_protocol import GraphScope
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import AttackGraphTaskBuilder, TaskGenerationRequest
from src.core.models.ag import ActionNode, GraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskResult,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ProjectionRequest,
    ProjectionRequestKind,
)
from src.core.models.kg import Host
from src.core.models.kg_enums import EntityStatus
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskStatus, TaskType
from src.core.runtime.result_applier import PhaseTwoResultApplier


def test_asset_confirmation_success_progresses_service_validation_to_ready() -> None:
    kg = KnowledgeGraph()
    kg.add_node(
        Host(
            id="host-1",
            label="127.0.0.1",
            address="127.0.0.1",
            status=EntityStatus.OBSERVED,
            confidence=0.9,
        )
    )

    projector = AttackGraphProjector()
    task_builder = AttackGraphTaskBuilder()
    initial_ag = projector.project(kg)
    initial_tg_result = task_builder.build_candidates(
        initial_ag,
        TaskGenerationRequest(
            action_ids=sorted(action.id for action in initial_ag.find_actions(activatable_only=True) if isinstance(action, ActionNode)),
            include_evidence_tasks=False,
        ),
    )
    assert initial_tg_result.task_graph is not None
    initial_tg = TaskGraph.from_dict(initial_tg_result.task_graph)
    asset_task = _only_task(initial_tg, TaskType.ASSET_CONFIRMATION)
    assert asset_task.status == TaskStatus.READY

    state = RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))
    state.register_task(TaskRuntime(task_id=asset_task.id, tg_node_id=asset_task.id))
    evidence_id = "evidence-asset-confirmation-1"
    service_ref = GraphRef(graph="kg", ref_id="host-1:8080/tcp", ref_type="Service", label="http")
    host_ref = GraphRef(graph="kg", ref_id="host-1", ref_type="Host", label="127.0.0.1")

    result = AgentTaskResult(
        request_id="request-asset-confirmation",
        agent_role=AgentRole.RECON_WORKER,
        operation_id="op-1",
        task_id=asset_task.id,
        tg_node_id=asset_task.id,
        status=AgentResultStatus.SUCCEEDED,
        summary="asset confirmation discovered http service",
        evidence=[
            EvidenceArtifact(
                evidence_id=evidence_id,
                kind="asset_confirmation",
                summary="nmap observed tcp/8080",
                payload_ref="runtime://worker-results/asset-confirmation/nmap",
                refs=[host_ref],
                metadata={"tool": "nmap"},
            )
        ],
        fact_write_requests=[
            FactWriteRequest(
                kind=FactWriteKind.ENTITY_UPSERT,
                source_task_id=asset_task.id,
                subject_ref=host_ref,
                attributes={"status": EntityStatus.VALIDATED.value, "confidence": 0.95},
                confidence=0.95,
                evidence_ids=[evidence_id],
                summary="Host reachability validated",
            ),
            FactWriteRequest(
                kind=FactWriteKind.ENTITY_UPSERT,
                source_task_id=asset_task.id,
                subject_ref=service_ref,
                attributes={
                    "host_id": "host-1",
                    "port": 8080,
                    "protocol": "tcp",
                    "service_name": "http",
                    "status": EntityStatus.OBSERVED.value,
                    "confidence": 0.9,
                },
                confidence=0.9,
                evidence_ids=[evidence_id],
                summary="HTTP service discovered",
            ),
            FactWriteRequest(
                kind=FactWriteKind.RELATION_UPSERT,
                source_task_id=asset_task.id,
                subject_ref=host_ref,
                relation_type="HOSTS",
                object_ref=service_ref,
                attributes={"port": 8080, "protocol": "tcp", "confidence": 0.9},
                confidence=0.9,
                evidence_ids=[evidence_id],
                summary="host exposes http service",
            ),
        ],
        projection_requests=[
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_LOCAL_FRONTIER,
                source_task_id=asset_task.id,
                reason="asset confirmation produced service facts",
                target_refs=[host_ref, service_ref],
            )
        ],
    )

    applied = PhaseTwoResultApplier().apply(
        result,
        state,
        kg_ref=ProtocolGraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        kg_store=kg,
        attack_graph=initial_ag,
        task_graph=initial_tg,
    )

    assert applied.kg_apply_result is not None
    assert applied.ag_graph is not None
    assert applied.tg_graph is not None
    assert kg.get_node("host-1:8080/tcp").type.value == "Service"
    assert kg.get_node(evidence_id).type.value == "Evidence"

    updated_tg = TaskGraph.from_dict(applied.tg_graph)
    host_asset_tasks = [
        task
        for task in _tasks(updated_tg, TaskType.ASSET_CONFIRMATION)
        if any(ref.ref_id == "host-1" for ref in task.target_refs)
        or task.input_bindings.get("host_id") == "host-1"
    ]
    service_tasks = _tasks(updated_tg, TaskType.SERVICE_VALIDATION)

    assert host_asset_tasks
    assert all(task.status != TaskStatus.READY for task in host_asset_tasks)
    assert service_tasks
    assert any(task.status == TaskStatus.READY for task in service_tasks)
    assert all(task.status != TaskStatus.DRAFT for task in service_tasks)


def _tasks(task_graph: TaskGraph, task_type: TaskType) -> list[BaseTaskNode]:
    return [
        node
        for node in task_graph.list_nodes()
        if isinstance(node, BaseTaskNode) and node.task_type == task_type
    ]


def _only_task(task_graph: TaskGraph, task_type: TaskType) -> BaseTaskNode:
    tasks = _tasks(task_graph, task_type)
    assert len(tasks) == 1
    return tasks[0]
