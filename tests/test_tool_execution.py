from __future__ import annotations

import sys

from typing import Any

import pytest

from src.core.execution import (
    AdapterPolicyConfig,
    ExecutionExecutor,
    HttpRequestExecutionAdapter,
    LocalShellAdapter,
    MCPExecutionAdapter,
    MCPToolCallResult,
    ToolAdapterResolver,
    ToolBinding,
    ToolExecutor,
    ToolPlan,
    build_tool_plan,
)
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


class FakeMCPClient:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls: list[dict[str, Any]] = []

    def is_available(self, server_id: str | None = None) -> bool:
        return self.available

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
    ) -> MCPToolCallResult:
        self.calls.append(
            {
                "server_id": server_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "timeout_seconds": timeout_seconds,
            }
        )
        return MCPToolCallResult(success=True, content={"ok": True, "tool": tool_name}, metadata={"fixture": True})


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


def test_local_shell_adapter_honors_acceptable_exit_codes_from_plan() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="python",
        adapter="local_shell",
        command=sys.executable,
        args={"argv": [sys.executable, "-c", "import sys; sys.exit(7)"], "acceptable_exit_codes": [7]},
    )

    result = LocalShellAdapter().execute(plan)

    assert result.success is True
    assert result.exit_code == 7
    assert result.metadata["category"] == "success"


def test_local_shell_adapter_reports_timeout_category() -> None:
    plan = ToolPlan(
        task_id="task-1",
        tool="python",
        adapter="local_shell",
        command=sys.executable,
        args={"argv": [sys.executable, "-c", "import time; time.sleep(2)"]},
        timeout_seconds=1,
    )

    result = LocalShellAdapter().execute(plan)

    assert result.success is False
    assert result.exit_code == "timeout"
    assert result.metadata["category"] == "timeout"
    assert result.metadata["timed_out"] is True


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


def test_http_request_adapter_blocks_cross_origin_without_network() -> None:
    adapter = HttpRequestExecutionAdapter()
    plan = ToolPlan(
        task_id="task-1",
        tool="http_request",
        adapter="http_request",
        target="http://example.test/admin",
        args={"method": "HEAD", "same_origin": "http://127.0.0.1/"},
    )

    result = adapter.execute(plan)

    assert result.success is False
    assert result.metadata["category"] == "policy_denied"
    assert result.metadata["blocked_reason"] == "cross_origin"


def test_http_request_adapter_converts_successful_head_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.execution.adapters import http_request_adapter

    class FakeHeaders:
        def items(self) -> list[tuple[str, str]]:
            return [("content-type", "text/html")]

        def get(self, key: str) -> str | None:
            return "text/html" if key.lower() == "content-type" else None

    class FakeResponse:
        status = 204
        headers = FakeHeaders()

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(http_request_adapter, "urlopen", lambda request, timeout: FakeResponse())
    plan = ToolPlan(
        task_id="task-1",
        tool="http_request",
        adapter="http_request",
        target="http://127.0.0.1/status",
        args={"method": "HEAD", "same_origin": "http://127.0.0.1/"},
    )

    result = HttpRequestExecutionAdapter().execute(plan)

    assert result.success is True
    assert result.exit_code == 204
    assert result.metadata["reachable"] is True
    assert result.metadata["content_type"] == "text/html"


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


def test_mcp_adapter_converts_tool_result() -> None:
    client = FakeMCPClient()
    adapter = MCPExecutionAdapter(client)
    plan = ToolPlan(
        task_id="task-1",
        tool="scan_ports",
        target="10.0.0.5",
        args={"ports": "80,443"},
        metadata={"mcp_server_id": "pentest-tools", "mcp_tool_name": "nmap_scan"},
    )

    result = adapter.execute(plan)

    assert result.success is True
    assert result.adapter == "mcp"
    assert result.metadata["mcp_server_id"] == "pentest-tools"
    assert client.calls[0]["tool_name"] == "nmap_scan"
    assert client.calls[0]["arguments"]["target"] == "10.0.0.5"


def test_execution_executor_defaults_to_mcp_when_binding_is_available() -> None:
    client = FakeMCPClient()
    resolver = ToolAdapterResolver(
        bindings=[
            ToolBinding(
                tool="scan_ports",
                default_adapter="mcp",
                fallback_adapters=["local_shell"],
                mcp={"server_id": "pentest-tools", "tool_name": "nmap_scan"},
                allowed_task_types=["PORT_SCAN"],
            )
        ]
    )
    executor = ExecutionExecutor(
        [MCPExecutionAdapter(client), LocalShellAdapter()],
        resolver=resolver,
    )
    plan = ToolPlan(
        task_id="task-1",
        tool="scan_ports",
        target="10.0.0.5",
        args={"ports": "80"},
        metadata={"task_type": "PORT_SCAN"},
    )

    result = executor.execute(plan)

    assert result.adapter == "mcp"
    assert result.metadata["mcp_tool_name"] == "nmap_scan"


def test_execution_executor_falls_back_to_local_for_low_risk_task_when_mcp_unavailable() -> None:
    resolver = ToolAdapterResolver(
        policy=AdapterPolicyConfig(allow_local_fallback_for_task_types=["PORT_SCAN"]),
        bindings=[
            {
                "tool": "echo",
                "default_adapter": "mcp",
                "fallback_adapters": ["local_shell"],
                "mcp": {"server_id": "pentest-tools", "tool_name": "echo"},
                "allowed_task_types": ["PORT_SCAN"],
            }
        ],
    )
    executor = ExecutionExecutor(
        [MCPExecutionAdapter(FakeMCPClient(available=False)), LocalShellAdapter()],
        resolver=resolver,
    )
    plan = ToolPlan(
        task_id="task-1",
        tool="echo",
        command=sys.executable,
        args={"argv": [sys.executable, "-c", "print('fallback')"]},
        metadata={"task_type": "PORT_SCAN"},
    )

    result = executor.execute(plan)

    assert result.adapter == "local_shell"
    assert result.stdout.strip() == "fallback"


def test_execution_executor_fail_closed_for_sensitive_task_without_mcp() -> None:
    resolver = ToolAdapterResolver(
        policy=AdapterPolicyConfig(
            force_mcp_for_task_types=["CREDENTIAL_VALIDATION"],
            deny_local_fallback_for_task_types=["CREDENTIAL_VALIDATION"],
        ),
        bindings=[
            {
                "tool": "validate_credential",
                "default_adapter": "mcp",
                "fallback_adapters": ["local_shell"],
                "mcp": {"server_id": "credential-tools", "tool_name": "validate_credential"},
                "allowed_task_types": ["CREDENTIAL_VALIDATION"],
            }
        ],
    )
    executor = ExecutionExecutor(
        [MCPExecutionAdapter(FakeMCPClient(available=False)), LocalShellAdapter()],
        resolver=resolver,
    )
    plan = ToolPlan(
        task_id="task-1",
        tool="validate_credential",
        command=sys.executable,
        metadata={"task_type": "CREDENTIAL_VALIDATION"},
    )

    with pytest.raises(ValueError, match="MCP is not available"):
        executor.execute(plan)


def test_adapter_policy_deny_overrides_explicit_adapter() -> None:
    resolver = ToolAdapterResolver(
        policy=AdapterPolicyConfig(deny_adapters=["local_shell"]),
        bindings=[{"tool": "echo", "fallback_adapters": ["local_shell"]}],
    )
    executor = ExecutionExecutor([LocalShellAdapter()], resolver=resolver)
    plan = ToolPlan(task_id="task-1", tool="echo", adapter="local_shell", command="echo ok")

    with pytest.raises(ValueError, match="denied by policy"):
        executor.execute(plan)


def test_tool_policy_blocks_unallowlisted_mcp_tool() -> None:
    state = _state({"mcp_tool_allowlist": ["scan_ports"], "mcp_server_allowlist": ["pentest-tools"]})
    plan = ToolPlan(
        task_id="task-1",
        tool="validate_credential",
        adapter="mcp",
        metadata={"mcp_server_id": "pentest-tools", "mcp_tool_name": "validate_credential"},
    )

    result = ToolExecutor().execute(plan, state)

    assert result.status.value == "blocked"
    assert result.metadata["tool_policy"]["gate"] == "mcp_policy"
