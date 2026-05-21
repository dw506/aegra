from __future__ import annotations

import pytest

from src.core.models.tg import TaskType
from src.core.workers.access_validation_worker import AccessValidationWorker
from src.core.workers.access_worker import AccessWorker
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.goal_validation_worker import GoalValidationWorker
from src.core.workers.goal_worker import GoalWorker
from src.core.workers.privilege_validation_worker import PrivilegeValidationWorker
from src.core.workers.registry import WorkerRegistry


def test_worker_registry_default_registers_only_primary_workers() -> None:
    workers = WorkerRegistry.default().list_all()

    assert {type(worker) for worker in workers} == {
        AccessValidationWorker,
        GoalValidationWorker,
        PrivilegeValidationWorker,
    }
    assert {worker.name for worker in workers} == {
        "access_validation_worker",
        "goal_validation_worker",
        "privilege_validation_worker",
    }


def test_worker_registry_rejects_legacy_worker() -> None:
    registry = WorkerRegistry()

    with pytest.raises(TypeError, match="BaseWorkerAgent"):
        registry.register(AccessWorker())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="BaseWorkerAgent"):
        registry.register(GoalWorker())  # type: ignore[arg-type]


def test_worker_registry_selects_access_validation_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
        )
    )

    assert isinstance(worker, AccessValidationWorker)


def test_worker_registry_selects_goal_validation_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.GOAL_CONDITION_VALIDATION.value,
        )
    )

    assert isinstance(worker, GoalValidationWorker)


def test_worker_registry_selects_privilege_validation_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.PRIVILEGE_CONFIGURATION_VALIDATION.value,
        )
    )

    assert isinstance(worker, PrivilegeValidationWorker)
