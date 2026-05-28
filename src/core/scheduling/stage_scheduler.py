"""Scheduler boundary for stage-level TG tasks."""

from __future__ import annotations

from src.core.models.runtime import RuntimeState
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskType
from src.core.stage.models import StageTask


STAGE_TASK_TYPES = {
    TaskType.RECON_STAGE,
    TaskType.VULN_ANALYSIS_STAGE,
    TaskType.EXPLOIT_STAGE,
    TaskType.ACCESS_PIVOT_STAGE,
    TaskType.GOAL_STAGE,
}


def schedule_ready_stage_tasks(
    task_graph: TaskGraph,
    runtime_state: RuntimeState | None = None,
    *,
    limit: int | None = None,
) -> list[StageTask]:
    """Return ready TaskGraph nodes that represent stage-level work."""

    del runtime_state
    task_graph.refresh_blocked_states()
    selected: list[StageTask] = []
    for task in task_graph.find_schedulable_tasks():
        if not isinstance(task, BaseTaskNode) or task.task_type not in STAGE_TASK_TYPES:
            continue
        selected.append(StageTask.from_task_node(task))
        if limit is not None and len(selected) >= limit:
            break
    return selected


__all__ = ["STAGE_TASK_TYPES", "schedule_ready_stage_tasks"]
