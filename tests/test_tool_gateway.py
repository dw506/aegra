"""ToolGateway is a drop-in MCP client that resolves pivot transport behind one call."""

from __future__ import annotations

from typing import Any

from src.core.execution.mcp_client import MCPToolCallResult
from src.core.execution.tool_gateway import ToolGateway
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult


class _RecordingMCPClient:
    """Fake MCP client that records the exact call it received."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tools_listed = 0

    def is_available(self, server_id: str | None = None) -> bool:
        return True

    def list_tools(self) -> dict[str, Any]:
        self.tools_listed += 1
        return {"lab": {"available": True}}

    def call_tool(self, *, server_id: str, tool_name: str, arguments: dict[str, Any], timeout_seconds: int) -> MCPToolCallResult:
        self.calls.append(
            {"server_id": server_id, "tool_name": tool_name, "arguments": dict(arguments), "timeout_seconds": timeout_seconds}
        )
        return MCPToolCallResult(success=True, stdout="mcp-ran", exit_code=0, metadata={"server_id": server_id})


class _FakeProxyAdapter:
    """Stand-in for proxy_shell that never touches the network/subprocess."""

    name = "proxy_shell"

    def __init__(self) -> None:
        self.executed: list[ToolPlan] = []

    def supports(self, plan: ToolPlan) -> bool:
        return plan.adapter == self.name

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        self.executed.append(plan)
        return ToolExecutionResult(adapter=self.name, tool=plan.tool, success=True, stdout="pivot-ran", exit_code=0)


def _direct_route() -> dict[str, Any]:
    return {
        "route_id": "route-1",
        "session_id": "sess-1",
        "metadata": {"transport": {"adapter": "proxy_shell", "proxy_url": "socks5://127.0.0.1:9050"}},
    }


def test_no_pivot_delegates_byte_identical_to_mcp_client() -> None:
    client = _RecordingMCPClient()
    gateway = ToolGateway(client)

    result = gateway.call_tool(
        server_id="lab", tool_name="http_probe", arguments={"url": "http://t/"}, timeout_seconds=30
    )

    assert isinstance(result, MCPToolCallResult)
    assert result.stdout == "mcp-ran"
    assert client.calls == [
        {"server_id": "lab", "tool_name": "http_probe", "arguments": {"url": "http://t/"}, "timeout_seconds": 30}
    ]


def test_route_without_direct_adapter_still_uses_mcp() -> None:
    client = _RecordingMCPClient()
    gateway = ToolGateway(client, adapters=[_FakeProxyAdapter()])
    # Route exists and is referenced, but declares no direct transport adapter.
    route = {"route_id": "route-1", "metadata": {"transport": {}}}

    result = gateway.call_tool(
        server_id="lab",
        tool_name="internal_service_discover",
        arguments={"route_id": "route-1"},
        timeout_seconds=30,
        pivot_routes=[route],
    )

    assert result.stdout == "mcp-ran"
    assert len(client.calls) == 1


def test_direct_route_runs_adapter_and_adapts_result() -> None:
    client = _RecordingMCPClient()
    adapter = _FakeProxyAdapter()
    gateway = ToolGateway(client, adapters=[adapter])

    result = gateway.call_tool(
        server_id="lab",
        tool_name="run_command",
        arguments={"route_id": "route-1", "command": "id"},
        timeout_seconds=30,
        pivot_routes=[_direct_route()],
    )

    assert client.calls == []  # MCP not used
    assert len(adapter.executed) == 1
    assert adapter.executed[0].adapter == "proxy_shell"
    assert result.success is True
    assert result.stdout == "pivot-ran"
    assert result.metadata["via_pivot_transport"] is True
    assert result.metadata["adapter"] == "proxy_shell"


def test_unreferenced_route_does_not_implicitly_pivot() -> None:
    client = _RecordingMCPClient()
    adapter = _FakeProxyAdapter()
    gateway = ToolGateway(client, adapters=[adapter])

    # A direct route exists in runtime, but this call does not reference it.
    gateway.call_tool(
        server_id="lab",
        tool_name="http_probe",
        arguments={"url": "http://t/"},
        timeout_seconds=30,
        pivot_routes=[_direct_route()],
    )

    assert len(client.calls) == 1
    assert adapter.executed == []


def test_is_available_and_list_tools_pass_through() -> None:
    client = _RecordingMCPClient()
    gateway = ToolGateway(client)

    assert gateway.is_available("lab") is True
    assert gateway.list_tools() == {"lab": {"available": True}}
    assert client.tools_listed == 1
