from __future__ import annotations

from src.core.execution import ToolExecutor, ToolPlan, build_tool_plan
from src.core.models.ag import GraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.models.scope import Asset, Engagement
from src.core.models.tg import TaskNode, TaskStatus, TaskType


def _state(policy: dict | None = None) -> RuntimeState:
    state = RuntimeState(operation_id="op-tool", execution=OperationRuntime(operation_id="op-tool"))
    if policy is not None:
        state.execution.metadata["runtime_policy"] = policy
    return state


def _task(host: str = "10.0.0.5") -> TaskNode:
    return TaskNode(
        id="task-1",
        label="service validation",
        task_type=TaskType.SERVICE_VALIDATION,
        status=TaskStatus.READY,
        input_bindings={"target_address": host, "tool_hint": "nmap", "adapter": "incalmo_c2", "agent_id": "agent-1"},
        target_refs=[GraphRef(graph="kg", ref_id=host, ref_type="Host")],
        estimated_risk=0.1,
        estimated_noise=0.1,
    )


def test_tool_plan_uses_explicit_adapter_and_agent_ref() -> None:
    plan = build_tool_plan(_task())

    assert plan.adapter == "incalmo_c2"
    assert plan.command == "nmap"
    assert plan.target_agent_ref == "agent-1"


def test_tool_executor_returns_blocked_result_for_out_of_scope_task() -> None:
    state = _state(
        {
            "engagement": Engagement(
                engagement_id="eng-1",
                assets=[Asset(kind="host", value="10.0.0.1")],
            ).model_dump(mode="json")
        }
    )
    task = _task(host="10.0.0.5")

    result = ToolExecutor().execute(build_tool_plan(task), state, task=task)

    assert result.status.value == "blocked"
    assert result.metadata["tool_policy"]["gate"] == "scope"


def test_tool_executor_reports_missing_adapter_after_policy_allows() -> None:
    state = _state({"authorized_hosts": ["10.0.0.5"]})
    task = _task()

    result = ToolExecutor().execute(build_tool_plan(task), state, task=task)

    assert result.status.value == "failed"
    assert "not registered" in result.summary
