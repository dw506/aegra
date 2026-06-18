"""Single tool-execution boundary.

Callers see exactly one interface — ``call_tool(server_id, tool_name, arguments,
timeout_seconds)`` — identical to the :class:`MCPClient` protocol. Transport
selection happens *behind* it: by default the call is delegated unchanged to the
wrapped MCP client (byte-identical to calling the client directly), and only when
an active pivot route declares a *direct* transport (network namespace / tunnel /
proxy) is the call routed through the matching execution adapter. The adapter's
``ToolExecutionResult`` is adapted back to an ``MCPToolCallResult`` so callers
never learn which transport actually ran, and the agent/planner never pick one.
"""

from __future__ import annotations

from typing import Any

from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.adapters.netns_shell_adapter import NetnsShellAdapter
from src.core.execution.adapters.proxy_shell_adapter import ProxyShellAdapter
from src.core.execution.adapters.tunnel_adapter import TunnelAdapter
from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.execution.pivot_context import PivotExecutionContextResolver
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult


# Adapter aliases that mean "wrap the egress locally" (direct transport) instead of
# letting the MCP server perform the call. Maps any declared alias to its canonical
# adapter name.
_DIRECT_TRANSPORT_ALIASES: dict[str, str] = {
    "netns_shell": "netns_shell",
    "tcp_tunnel": "tcp_tunnel",
    "tunnel": "tcp_tunnel",
    "proxy_shell": "proxy_shell",
    "socks_proxy": "proxy_shell",
    "http_proxy": "proxy_shell",
}


class ToolGateway:
    """Drop-in :class:`MCPClient` that resolves transport behind one ``call_tool``."""

    def __init__(
        self,
        mcp_client: MCPClient | None = None,
        *,
        resolver: PivotExecutionContextResolver | None = None,
        adapters: list[ExecutionAdapter] | None = None,
    ) -> None:
        self._mcp_client = mcp_client or UnavailableMCPClient()
        self._resolver = resolver or PivotExecutionContextResolver()
        default_adapters = adapters or [NetnsShellAdapter(), TunnelAdapter(), ProxyShellAdapter()]
        self._adapters: dict[str, ExecutionAdapter] = {adapter.name: adapter for adapter in default_adapters}

    # --- MCPClient protocol passthrough ---------------------------------------
    def is_available(self, server_id: str | None = None) -> bool:
        return self._mcp_client.is_available(server_id)

    def list_tools(self) -> dict[str, Any]:
        return self._mcp_client.list_tools()

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
        pivot_routes: list[dict[str, Any]] | None = None,
        sessions: list[dict[str, Any]] | None = None,
    ) -> MCPToolCallResult:
        del sessions  # reserved: routes carry transport today; sessions kept for parity
        route = self._active_route(arguments=arguments, pivot_routes=pivot_routes or [])
        adapter_name = self._direct_adapter_name(route)
        if adapter_name is None or adapter_name not in self._adapters:
            # Default path — identical to calling the MCP client directly.
            return self._as_result(
                self._mcp_client.call_tool(
                    server_id=server_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    timeout_seconds=timeout_seconds,
                )
            )
        plan = self._tool_plan(
            server_id=server_id,
            tool_name=tool_name,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
            route=route or {},
            adapter_name=adapter_name,
        )
        try:
            execution = self._adapters[adapter_name].execute(plan)
        except Exception as exc:  # noqa: BLE001 - transport faults must not crash the round
            return MCPToolCallResult(
                success=False,
                exit_code="adapter_error",
                stderr=str(exc),
                metadata={"server_id": server_id, "adapter": adapter_name, "via_pivot_transport": True},
            )
        return self._result_from_execution(execution, server_id=server_id)

    # --- helpers --------------------------------------------------------------
    @staticmethod
    def _active_route(
        *,
        arguments: dict[str, Any],
        pivot_routes: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not pivot_routes:
            return None
        route_id = arguments.get("route_id") or arguments.get("selected_route_id")
        if route_id is None:
            # No explicit route on the call — do not implicitly pivot.
            return None
        for route in pivot_routes:
            if isinstance(route, dict) and str(route.get("route_id")) == str(route_id):
                return route
        return None

    @staticmethod
    def _direct_adapter_name(route: dict[str, Any] | None) -> str | None:
        if not isinstance(route, dict):
            return None
        metadata = route.get("metadata") if isinstance(route.get("metadata"), dict) else {}
        transport = metadata.get("transport") if isinstance(metadata.get("transport"), dict) else {}
        declared = transport.get("adapter") or route.get("adapter")
        if declared is None:
            return None
        return _DIRECT_TRANSPORT_ALIASES.get(str(declared).strip().lower())

    @staticmethod
    def _transport_block(route: dict[str, Any]) -> dict[str, Any]:
        metadata = route.get("metadata") if isinstance(route.get("metadata"), dict) else {}
        transport = metadata.get("transport") if isinstance(metadata.get("transport"), dict) else {}
        return {
            "proxy_url": transport.get("proxy_url"),
            "network_namespace": transport.get("network_namespace") or transport.get("namespace"),
            "tunnel_endpoint": transport.get("tunnel_endpoint") or transport.get("endpoint"),
        }

    def _tool_plan(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
        route: dict[str, Any],
        adapter_name: str,
    ) -> ToolPlan:
        transport = self._transport_block(route)
        command = arguments.get("command")
        if command is None and isinstance(arguments.get("argv"), list):
            command = " ".join(str(part) for part in arguments["argv"])
        route_transport = (
            route.get("metadata", {}).get("transport", {})
            if isinstance(route.get("metadata"), dict)
            else {}
        )
        args: dict[str, Any] = {
            **dict(arguments),
            "selected_route": route,
            "route_id": route.get("route_id"),
            "session_id": route.get("session_id"),
            **{key: value for key, value in transport.items() if value is not None},
        }
        metadata: dict[str, Any] = {
            "execution_node_id": f"gateway-{server_id}-{tool_name}",
            "server_id": server_id,
            "mcp_tool_name": tool_name,
            "route_id": route.get("route_id"),
            "selected_route": route,
            "transport": route_transport,
            **{key: value for key, value in transport.items() if value is not None},
        }
        timeout = int(timeout_seconds) if timeout_seconds and int(timeout_seconds) > 0 else 60
        return ToolPlan(
            task_id=f"gateway-{server_id}-{tool_name}",
            tool=tool_name,
            adapter=adapter_name,
            command=str(command) if command is not None else None,
            target=str(arguments.get("target") or arguments.get("host") or "") or None,
            args=args,
            timeout_seconds=timeout,
            metadata=metadata,
        )

    @staticmethod
    def _as_result(raw: MCPToolCallResult | dict[str, Any]) -> MCPToolCallResult:
        if isinstance(raw, MCPToolCallResult):
            return raw
        return MCPToolCallResult.model_validate(raw)

    @staticmethod
    def _result_from_execution(execution: ToolExecutionResult, *, server_id: str) -> MCPToolCallResult:
        return MCPToolCallResult(
            success=execution.success,
            content=execution.metadata.get("execution_context"),
            stdout=execution.stdout,
            stderr=execution.stderr,
            exit_code=execution.exit_code,
            metadata={
                **dict(execution.metadata),
                "server_id": server_id,
                "adapter": execution.adapter,
                "via_pivot_transport": True,
            },
        )


__all__ = ["ToolGateway"]
