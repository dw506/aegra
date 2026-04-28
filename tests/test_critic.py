from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.models.tg import BaseTaskEdge, DependencyType, OutcomeNode, TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.planner.critic import TaskCriticContext, TaskGraphCritic


def build_task_graph() -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-1",
            label="Asset confirmation",
            task_type=TaskType.ASSET_CONFIRMATION,
            status=TaskStatus.READY,
            source_action_id="action-1",
            target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            input_bindings={"host_id": "host-1"},
            goal_relevance=0.8,
        )
    )
    graph.add_node(
        TaskNode(
            id="task-2",
            label="Service validation",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.BLOCKED,
            source_action_id="action-2",
            target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
            input_bindings={"service": {"id": "svc-1"}},
            goal_relevance=0.4,
            gate_ids={"approval_gate"},
        )
    )
    graph.add_edge(
        BaseTaskEdge(
            id="edge-1",
            dependency_type=DependencyType.DEPENDS_ON,
            source="task-1",
            target="task-2",
            label="depends_on",
        )
    )
    return graph


def test_task_graph_critic_finds_blocked_and_invalidated_tasks() -> None:
    graph = build_task_graph()
    critic = TaskGraphCritic()

    result = critic.critique_task_graph(
        graph,
        TaskCriticContext(invalidated_ref_keys={"kg:Service:svc-1"}),
    )

    assert "task-2" in result.blocked_task_ids
    assert "task-2" in result.invalidated_task_ids
    assert result.replan_frontiers


def test_task_graph_critic_can_attach_outcome_and_replace_subgraph() -> None:
    graph = build_task_graph()
    critic = TaskGraphCritic()

    outcome = OutcomeNode(
        id="outcome-1",
        label="Validation failed",
        outcome_type="failure",
        invalidated_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
    )
    critic.attach_outcome(graph, "task-2", outcome)

    replacement = TaskNode(
        id="task-3",
        label="Retry service validation",
        task_type=TaskType.SERVICE_VALIDATION,
        status=TaskStatus.DRAFT,
        source_action_id="action-3",
        target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
        input_bindings={"service": {"id": "svc-1"}, "mode": "retry"},
        goal_relevance=0.6,
    )
    created = critic.replace_subgraph(graph, "task-2", [replacement], [])

    assert outcome.id in {node.id for node in graph.list_nodes()}
    assert created == ["task-3"]
    assert graph.get_node("task-2").status == TaskStatus.SUPERSEDED
    frontier = critic.collect_replan_frontier(graph, "task-2")
    assert frontier.root_task_id == "task-2"
