from __future__ import annotations

import pytest

from src.core.models.tg import TaskType
from src.core.workers.access_worker import AccessWorker
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.goal_worker import GoalWorker
from src.core.workers.llm_worker import LLMWorkerAgent
from src.core.workers.registry import WorkerRegistry


def test_worker_registry_default_registers_only_llm_worker() -> None:
    workers = WorkerRegistry.default().list_all()

    assert len(workers) == 1
    assert isinstance(workers[0], LLMWorkerAgent)
    assert workers[0].name == "llm_worker_agent"


def test_worker_registry_rejects_legacy_worker() -> None:
    registry = WorkerRegistry()

    with pytest.raises(TypeError, match="BaseWorkerAgent"):
        registry.register(AccessWorker())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="BaseWorkerAgent"):
        registry.register(GoalWorker())  # type: ignore[arg-type]


def test_worker_registry_selects_llm_worker_for_any_task() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
        )
    )

    assert isinstance(worker, LLMWorkerAgent)
