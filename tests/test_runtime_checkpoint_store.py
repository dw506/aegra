from __future__ import annotations

from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.runtime.checkpoint_store import RuntimeCheckpointManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_mark_task_lineage_records_replacement_chain() -> None:
    state = build_state()
    manager = RuntimeCheckpointManager()

    lineage = manager.mark_task_lineage(
        state,
        task_id="task-2",
        replaces_task_id="task-1",
        derived_from_checkpoint="cp-1",
    )

    assert lineage["replaces_task_id"] == "task-1"
    assert state.execution.metadata["recovery"]["task_lineage"]["task-2"]["derived_from_checkpoint"] == "cp-1"


def test_collect_recovery_context_prefers_task_checkpoint_ref() -> None:
    state = build_state()
    manager = RuntimeCheckpointManager()
    manager.create_checkpoint(state, "cp-1", ["task-1"], summary="stable")
    state.execution.tasks["task-2"] = TaskRuntime(
        task_id="task-2",
        tg_node_id="tg-2",
        checkpoint_ref="cp-1",
    )

    context = manager.collect_recovery_context(state, "task-2")

    assert context["checkpoint"]["checkpoint_id"] == "cp-1"


def test_latest_stable_checkpoint_returns_none_when_absent() -> None:
    state = build_state()
    manager = RuntimeCheckpointManager()

    assert manager.latest_stable_checkpoint(state) is None

