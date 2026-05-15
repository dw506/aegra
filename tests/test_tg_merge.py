from __future__ import annotations

from src.core.graph.tg_merge import merge_task_graphs
from src.core.models.ag import GraphRef
from src.core.models.tg import BaseTaskEdge, DependencyType, TaskGraph, TaskNode, TaskStatus, TaskType


def _service_task(*, status: TaskStatus = TaskStatus.DRAFT) -> TaskNode:
    return TaskNode(
        id="task-service",
        label="Service validation",
        task_type=TaskType.SERVICE_VALIDATION,
        status=status,
        source_action_id="action-service",
        target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service", label="svc-1")],
    )


def test_merge_preserves_existing_task_lifecycle_and_unblocks_new_downstream() -> None:
    existing = TaskGraph()
    old_task = _service_task(status=TaskStatus.SUCCEEDED)
    old_task.attempt_count = 1
    old_task.attempts = 1
    old_task.last_outcome_ref = "runtime://results/result-1"
    existing.add_node(old_task)

    incoming = TaskGraph()
    incoming.add_node(_service_task(status=TaskStatus.DRAFT))
    incoming.add_node(
        TaskNode(
            id="task-identity",
            label="Identity context confirmation",
            task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION,
            source_action_id="action-identity",
            target_refs=[GraphRef(graph="kg", ref_id="identity-1", ref_type="Identity", label="identity-1")],
        )
    )
    incoming.add_edge(
        BaseTaskEdge(
            id="edge-service-identity",
            dependency_type=DependencyType.DEPENDS_ON,
            source="task-service",
            target="task-identity",
            label="depends_on",
        )
    )

    merged = merge_task_graphs(existing, incoming)

    service = merged.get_node("task-service")
    identity = merged.get_node("task-identity")
    assert isinstance(service, TaskNode)
    assert isinstance(identity, TaskNode)
    assert service.status == TaskStatus.SUCCEEDED
    assert service.attempt_count == 1
    assert service.attempts == 1
    assert service.last_outcome_ref == "runtime://results/result-1"
    assert identity.status == TaskStatus.READY


def test_merge_retains_existing_only_tasks_and_edges() -> None:
    existing = TaskGraph()
    existing.add_node(_service_task(status=TaskStatus.SUCCEEDED))
    existing.add_node(
        TaskNode(
            id="task-evidence",
            label="Evidence collection",
            task_type=TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL,
            source_action_id="action-evidence",
        )
    )
    existing.add_edge(
        BaseTaskEdge(
            id="edge-service-evidence",
            dependency_type=DependencyType.PRODUCES_EVIDENCE_FOR,
            source="task-service",
            target="task-evidence",
            label="produces_evidence_for",
        )
    )

    incoming = TaskGraph()
    incoming.add_node(_service_task(status=TaskStatus.DRAFT))

    merged = merge_task_graphs(existing, incoming)

    assert merged.get_node("task-service").status == TaskStatus.SUCCEEDED
    assert merged.get_node("task-evidence").task_type == TaskType.EVIDENCE_COLLECTION_AND_ARCHIVAL
    assert merged.get_edge("edge-service-evidence").dependency_type == DependencyType.PRODUCES_EVIDENCE_FOR
