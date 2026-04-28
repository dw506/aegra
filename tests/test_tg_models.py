from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.models.tg import BaseTaskEdge, BaseTaskNode, DependencyType, TaskGraph, TaskStatus, TaskType


def test_task_graph_dependency_drives_ready_state() -> None:
    graph = TaskGraph()
    first = BaseTaskNode(
        id="task-1",
        label="Asset confirmation",
        task_type=TaskType.ASSET_CONFIRMATION,
        source_action_id="action-1",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )
    second = BaseTaskNode(
        id="task-2",
        label="Service validation",
        task_type=TaskType.SERVICE_VALIDATION,
        source_action_id="action-2",
        target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
    )
    graph.add_node(first)
    graph.add_node(second)
    graph.add_edge(
        BaseTaskEdge(
            id="edge-1",
            dependency_type=DependencyType.DEPENDS_ON,
            source="task-1",
            target="task-2",
            label="depends_on",
        )
    )

    graph.refresh_blocked_states()
    assert [task.id for task in graph.ready_tasks()] == ["task-1"]
    assert graph.get_node("task-2").status == TaskStatus.PENDING

    graph.mark_task_status("task-1", TaskStatus.SUCCEEDED)
    assert [task.id for task in graph.ready_tasks()] == ["task-2"]


def test_task_graph_roundtrip_serialization() -> None:
    graph = TaskGraph()
    task = BaseTaskNode(
        id="task-1",
        label="Goal validation",
        task_type=TaskType.GOAL_CONDITION_VALIDATION,
        source_action_id="action-1",
        target_refs=[GraphRef(graph="kg", ref_id="goal-1", ref_type="Goal")],
    )
    graph.add_node(task)
    restored = TaskGraph.from_dict(graph.to_dict())

    assert restored.get_node("task-1").task_type == TaskType.GOAL_CONDITION_VALIDATION


def test_scheduler_queries_and_state_transitions() -> None:
    graph = TaskGraph()
    task = BaseTaskNode(
        id="task-1",
        label="Service validation",
        task_type=TaskType.SERVICE_VALIDATION,
        target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
        resource_keys={"service:svc-1"},
        parallelizable=False,
        approval_required=True,
        gate_ids={"approval_gate"},
        max_attempts=2,
    )
    graph.add_node(task)
    graph.refresh_blocked_states()

    assert graph.find_tasks_blocked_by_gate("approval_gate")[0].id == "task-1"
    assert graph.find_tasks_requiring_resource("service:svc-1")[0].id == "task-1"

    task.gate_ids.clear()
    graph.refresh_blocked_states()
    assert graph.find_schedulable_tasks()[0].id == "task-1"

    graph.transition_task("task-1", TaskStatus.QUEUED)
    graph.find_schedulable_tasks()
    assert graph.get_node("task-1").status == TaskStatus.QUEUED
    graph.transition_task("task-1", TaskStatus.RUNNING)
    graph.transition_task("task-1", TaskStatus.FAILED)
    assert graph.find_retryable_tasks()[0].id == "task-1"
    graph.transition_task("task-1", TaskStatus.READY)
    assert graph.get_node("task-1").status == TaskStatus.READY


def test_checkpoint_node_and_recovery_edge() -> None:
    graph = TaskGraph()
    task = BaseTaskNode(
        id="task-1",
        label="Asset confirmation",
        task_type=TaskType.ASSET_CONFIRMATION,
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )
    recovery_task = BaseTaskNode(
        id="task-2",
        label="Retry asset confirmation",
        task_type=TaskType.ASSET_CONFIRMATION,
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )
    graph.add_node(task)
    graph.add_node(recovery_task)
    checkpoint = graph.create_checkpoint_node(
        task_id="task-1",
        anchor_refs=[GraphRef(graph="ag", ref_id="state-1", ref_type="StateNode")],
    )
    edge = graph.link_recovery_anchor("task-2", checkpoint.id)

    assert checkpoint.kind == "checkpoint"
    assert edge.dependency_type == DependencyType.RECOVERS_FROM
