from __future__ import annotations

import sys

import pytest

from src.core.execution import ExecutionExecutor, LocalShellAdapter, ToolExecutor, ToolPlan, build_tool_plan
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


def test_local_shell_adapter_executes_allowed_command() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="python",
        adapter="local_shell",
        command=sys.executable,
        args={"argv": [sys.executable, "-c", "print('ok')"]},
    )

    result = LocalShellAdapter().execute(plan)

    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"
    assert result.stderr == ""


def test_local_shell_adapter_returns_failure_result_on_nonzero_exit() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="python",
        adapter="local_shell",
        command=sys.executable,
        args={"argv": [sys.executable, "-c", "import sys; sys.exit(7)"]},
    )

    result = LocalShellAdapter().execute(plan)

    assert result.success is False
    assert result.exit_code == 7


def test_local_shell_adapter_rejects_unallowed_command_by_default() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="powershell",
        adapter="local_shell",
        command="powershell",
    )

    result = LocalShellAdapter().execute(plan)

    assert result.success is False
    assert result.exit_code == "policy_denied"
    assert "not allowed" in result.stderr


def test_execution_executor_selects_matching_adapter() -> None:
    executor = ExecutionExecutor([LocalShellAdapter()])
    plan = ToolPlan(
        task_id="task-1",
        tool="python",
        adapter="local_shell",
        command=sys.executable,
        args={"argv": [sys.executable, "-c", "print('selected')"]},
    )

    result = executor.execute(plan)

    assert result.success is True
    assert result.stdout.strip() == "selected"
    assert result.adapter == "local_shell"


def test_execution_executor_rejects_unsupported_adapter() -> None:
    executor = ExecutionExecutor([LocalShellAdapter()])
    plan = ToolPlan(
        task_id="task-1",
        tool="whoami",
        adapter="missing_adapter",
        command="whoami",
    )

    with pytest.raises(ValueError, match="No adapter supports"):
        executor.execute(plan)
