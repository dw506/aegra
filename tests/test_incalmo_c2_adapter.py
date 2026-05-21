from __future__ import annotations

import pytest

from src.core.execution import ExecutionExecutor
from src.core.execution.adapters.incalmo_c2_adapter import IncalmoC2Adapter
from src.core.execution.tool_plan import ToolPlan


class FakeIncalmoClient:
    def __init__(self, *, exit_code: str = "0", output: str = "ok") -> None:
        self.exit_code = exit_code
        self.output = output
        self.sent: list[dict] = []

    def send_command(self, *, agent_id, command, payloads=None):
        self.sent.append({"agent_id": agent_id, "command": command, "payloads": payloads})
        return {"id": "cmd-1"}

    def wait_for_command_result(self, *, command_id, poll_interval_seconds=1.0, max_attempts=45):
        return {
            "id": command_id,
            "exit_code": self.exit_code,
            "output": self.output,
            "stderr": "" if self.exit_code == "0" else "failed",
            "status": "completed" if self.exit_code == "0" else "failed",
        }


def test_incalmo_c2_adapter_supports_matching_plan() -> None:
    adapter = IncalmoC2Adapter(FakeIncalmoClient())

    assert adapter.supports(ToolPlan(task_id="task-1", tool="whoami", adapter="incalmo_c2", command="whoami"))
    assert not adapter.supports(ToolPlan(task_id="task-1", tool="whoami", adapter="local_shell", command="whoami"))


def test_incalmo_c2_adapter_rejects_missing_agent() -> None:
    adapter = IncalmoC2Adapter(FakeIncalmoClient())
    plan = ToolPlan(task_id="task-1", tool="whoami", adapter="incalmo_c2", command="whoami")

    with pytest.raises(ValueError, match="agent"):
        adapter.execute(plan)


def test_incalmo_c2_adapter_maps_success_result() -> None:
    client = FakeIncalmoClient()
    adapter = IncalmoC2Adapter(client)
    plan = ToolPlan(
        task_id="task-1",
        tool="whoami",
        adapter="incalmo_c2",
        command="whoami",
        target_agent_ref="agent-1",
        payloads={"arg": "value"},
    )

    result = adapter.execute(plan)

    assert result.success is True
    assert result.exit_code == "0"
    assert result.stdout == "ok"
    assert result.command_id == "cmd-1"
    assert result.metadata["agent_id"] == "agent-1"
    assert client.sent == [{"agent_id": "agent-1", "command": "whoami", "payloads": {"arg": "value"}}]


def test_incalmo_c2_adapter_maps_failure_result() -> None:
    adapter = IncalmoC2Adapter(FakeIncalmoClient(exit_code="1", output=""))
    plan = ToolPlan(
        task_id="task-1",
        tool="whoami",
        adapter="incalmo_c2",
        command="whoami",
        args={"agent_id": "agent-1"},
    )

    result = adapter.execute(plan)

    assert result.success is False
    assert result.exit_code == "1"
    assert result.stderr == "failed"


def test_execution_executor_can_dispatch_to_incalmo_adapter() -> None:
    executor = ExecutionExecutor([IncalmoC2Adapter(FakeIncalmoClient())])
    plan = ToolPlan(
        task_id="task-1",
        tool="whoami",
        adapter="incalmo_c2",
        command="whoami",
        metadata={"agent_id": "agent-1"},
    )

    result = executor.execute(plan)

    assert result.success is True
    assert result.adapter == "incalmo_c2"
