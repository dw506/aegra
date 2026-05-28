"""Utilities for writing stage task proposals into a TaskGraph."""

from __future__ import annotations

from typing import Any

from src.core.models.ag import GraphRef
from src.core.models.tg import BaseTaskEdge, DependencyType, TaskGraph, TaskNode, TaskStatus
from src.core.stage.models import StageTask, StageType


class StageTaskGraphBuilder:
    """Create TG task nodes from StageTask proposals."""

    def upsert_stage_tasks(
        self,
        task_graph: TaskGraph,
        stage_tasks: list[StageTask],
        *,
        dependencies: list[dict[str, str]] | None = None,
    ) -> list[str]:
        created: list[str] = []
        for stage_task in stage_tasks:
            if stage_task.task_id in task_graph._nodes:
                continue
            task_graph.add_node(self._task_node(stage_task))
            created.append(stage_task.task_id)
        for dependency in dependencies or []:
            source = dependency.get("source") or dependency.get("from")
            target = dependency.get("target") or dependency.get("to")
            if not source or not target or source not in task_graph._nodes or target not in task_graph._nodes:
                continue
            edge = BaseTaskEdge(
                id=TaskGraph.stable_edge_id(source, target, DependencyType.DEPENDS_ON),
                dependency_type=DependencyType.DEPENDS_ON,
                source=source,
                target=target,
                label="depends_on",
            )
            if edge.id not in task_graph._edges:
                task_graph.add_edge(edge)
        task_graph.refresh_blocked_states()
        return created

    @staticmethod
    def stage_task_from_proposal(proposal: dict[str, Any], *, default_id: str) -> StageTask | None:
        raw_stage_type = proposal.get("stage_type") or proposal.get("task_type")
        if raw_stage_type is None:
            return None
        try:
            stage_type = StageType(str(raw_stage_type))
        except ValueError:
            return None
        return StageTask(
            task_id=str(proposal.get("task_id") or proposal.get("id") or default_id),
            stage_type=stage_type,
            objective=str(proposal.get("objective") or proposal.get("label") or stage_type.value),
            target_refs=StageTaskGraphBuilder._refs(proposal.get("target_refs")),
            required_context=dict(proposal.get("required_context") or {}),
            success_criteria=[str(item) for item in proposal.get("success_criteria", []) if item is not None],
            max_steps=int(proposal.get("max_steps") or 8),
            risk_level=str(proposal.get("risk_level") or "medium"),  # type: ignore[arg-type]
            priority=int(proposal.get("priority") or 50),
            metadata=dict(proposal.get("metadata") or {}),
        )

    @staticmethod
    def _task_node(stage_task: StageTask) -> TaskNode:
        return TaskNode(
            id=stage_task.task_id,
            label=stage_task.objective,
            task_type=stage_task.stage_type.task_type,
            status=TaskStatus.DRAFT,
            input_bindings={
                "objective": stage_task.objective,
                "required_context": dict(stage_task.required_context),
                "success_criteria": list(stage_task.success_criteria),
                "max_steps": stage_task.max_steps,
                "risk_level": stage_task.risk_level,
                "metadata": dict(stage_task.metadata),
            },
            target_refs=list(stage_task.target_refs),
            estimated_risk={"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 0.9}.get(stage_task.risk_level, 0.4),
            priority=stage_task.priority,
            assigned_agent=stage_task.stage_type.value.lower(),
            max_attempts=stage_task.max_steps,
            tags={"stage_task", stage_task.stage_type.value.lower()},
        )

    @staticmethod
    def _refs(value: Any) -> list[GraphRef]:
        if not isinstance(value, list):
            return []
        refs: list[GraphRef] = []
        for item in value:
            if isinstance(item, GraphRef):
                refs.append(item)
            elif isinstance(item, dict):
                try:
                    refs.append(GraphRef.model_validate(item))
                except Exception:
                    continue
        return refs


__all__ = ["StageTaskGraphBuilder"]
