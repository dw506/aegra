from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntimeStatus, WorkerRuntime, WorkerStatus
from src.core.models.scope import Asset, DenylistRule, Engagement
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType
from src.core.runtime.scheduler import RuntimeScheduler
from src.core.runtime.audit_report import build_operation_audit_report


def _state(policy: dict) -> RuntimeState:
    state = RuntimeState(operation_id="op-policy", execution=OperationRuntime(operation_id="op-policy"))
    state.execution.metadata["runtime_policy"] = policy
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    return state


def _graph(*, host: str = "10.0.0.5", tags: set[str] | None = None) -> TaskGraph:
    graph = TaskGraph()
    graph.add_node(
        TaskNode(
            id="task-policy",
            label="Policy gated task",
            task_type=TaskType.SERVICE_VALIDATION,
            status=TaskStatus.READY,
            source_action_id="action-policy",
            input_bindings={"host_id": host},
            target_refs=[GraphRef(graph="kg", ref_id=host, ref_type="Host")],
            resource_keys={f"host:{host}"},
            tags=tags or set(),
            estimated_risk=0.1,
            estimated_noise=0.1,
        )
    )
    return graph


def test_scheduler_blocks_unscoped_target_and_audits_policy_decision() -> None:
    state = _state(
        {
            "engagement": Engagement(
                engagement_id="eng-1",
                assets=[Asset(kind="host", value="10.0.0.1")],
            ).model_dump(mode="json")
        }
    )

    result = RuntimeScheduler().tick(_graph(host="10.0.0.5"), state)

    assert result.selected_task_ids == []
    assert state.execution.tasks["task-policy"].status == TaskRuntimeStatus.BLOCKED
    assert any(entry["event_type"] == "policy_decision" for entry in state.execution.metadata["audit_log"])
    report = build_operation_audit_report(state)
    assert any(entry["event_type"] == "policy_decision" for entry in report["audit_log"])


def test_scheduler_blocks_denylist_and_sets_waiting_approval_for_active_exploit() -> None:
    deny_state = _state(
        {
            "engagement": Engagement(
                engagement_id="eng-1",
                assets=[Asset(kind="host", value="10.0.0.5")],
                denylist=[DenylistRule(rule_id="deny-1", kind="host", value="10.0.0.5")],
            ).model_dump(mode="json")
        }
    )
    deny_result = RuntimeScheduler().tick(_graph(host="10.0.0.5"), deny_state)
    assert deny_result.selected_task_ids == []
    assert deny_state.execution.tasks["task-policy"].status == TaskRuntimeStatus.BLOCKED

    approval_state = _state(
        {
            "engagement": Engagement(
                engagement_id="eng-1",
                assets=[Asset(kind="host", value="10.0.0.5")],
            ).model_dump(mode="json")
        }
    )
    approval_result = RuntimeScheduler().tick(_graph(host="10.0.0.5", tags={"active_exploit"}), approval_state)
    assert approval_result.selected_task_ids == []
    assert approval_state.execution.tasks["task-policy"].status == TaskRuntimeStatus.WAITING_APPROVAL
