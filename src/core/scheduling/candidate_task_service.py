"""Deterministic candidate collection for LLM SchedulerAgent context."""

from __future__ import annotations

from typing import Any

from src.core.models.runtime import RuntimeState, WorkerStatus
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskStatus


class CandidateTaskService:
    """Collect ready TG stage tasks without making the final scheduling choice."""

    def collect(
        self,
        task_graph: TaskGraph,
        runtime_state: RuntimeState | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        del runtime_state
        task_graph.refresh_blocked_states()
        candidates: list[dict[str, Any]] = []
        for task in task_graph.find_schedulable_tasks():
            if not isinstance(task, BaseTaskNode) or task.status != TaskStatus.READY:
                continue
            candidates.append(self._task_summary(task))
            if limit is not None and len(candidates) >= limit:
                break
        return candidates

    @staticmethod
    def _task_summary(task: BaseTaskNode) -> dict[str, Any]:
        return {
            "task_id": task.id,
            "task_type": task.task_type.value,
            "stage_type": task.task_type.value,
            "label": task.label,
            "objective": str(task.input_bindings.get("objective") or task.label),
            "status": task.status.value,
            "priority": task.priority,
            "goal_relevance": task.goal_relevance,
            "resource_keys": sorted(task.resource_keys),
            "target_refs": [ref.model_dump(mode="json") for ref in task.target_refs],
            "input_bindings": dict(task.input_bindings),
            "constraints": [str(item) for item in task.input_bindings.get("constraints", []) if item is not None],
            "success_criteria": [
                str(item) for item in task.input_bindings.get("success_criteria", []) if item is not None
            ],
            "assigned_agent": task.assigned_agent,
            "approval_required": task.approval_required,
        }


class RuntimeConstraintService:
    """Summarize runtime constraints for LLM scheduling prompts."""

    def summarize(self, runtime_state: RuntimeState | None) -> dict[str, Any]:
        if runtime_state is None:
            return {"available": False}
        return {
            "available": True,
            "operation_status": runtime_state.operation_status.value,
            "workers": [
                {
                    "worker_id": worker.worker_id,
                    "status": worker.status.value,
                    "current_task_id": worker.current_task_id,
                    "capabilities": sorted(worker.capabilities),
                    "current_load": worker.current_load,
                }
                for worker in runtime_state.workers.values()
            ],
            "idle_worker_ids": [
                worker.worker_id
                for worker in runtime_state.workers.values()
                if worker.status == WorkerStatus.IDLE
            ],
            "tasks": {
                task_id: {
                    "status": task.status.value,
                    "assigned_worker": task.assigned_worker,
                    "attempt_count": task.attempt_count,
                    "max_attempts": task.max_attempts,
                    "resource_keys": sorted(task.resource_keys),
                }
                for task_id, task in runtime_state.execution.tasks.items()
            },
            "active_locks": [
                {
                    "lock_key": lock.lock_key,
                    "owner_type": lock.owner_type,
                    "owner_id": lock.owner_id,
                    "status": lock.status.value,
                }
                for lock in runtime_state.locks.values()
            ],
            "sessions": [
                {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "bound_target": session.bound_target,
                    "bound_identity": session.bound_identity,
                }
                for session in runtime_state.sessions.values()
            ],
            "budgets": runtime_state.budgets.model_dump(mode="json"),
        }


__all__ = ["CandidateTaskService", "RuntimeConstraintService"]
