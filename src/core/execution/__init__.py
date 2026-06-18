"""Execution planning and adapter dispatch."""

from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.adapters.http_request_adapter import HttpRequestExecutionAdapter
from src.core.execution.adapters.local_shell_adapter import LocalShellAdapter
from src.core.execution.adapters.mcp_adapter import MCPExecutionAdapter
from src.core.execution.adapters.netns_shell_adapter import NetnsShellAdapter
from src.core.execution.adapters.proxy_shell_adapter import ProxyShellAdapter
from src.core.execution.adapters.tunnel_adapter import TunnelAdapter
from src.core.execution.adapter_resolver import AdapterPolicyConfig, AdapterResolution, ToolAdapterResolver, ToolBinding
from src.core.execution.configured_mcp_client import ConfiguredMCPClient, MCPRuntimeConfig, MCPServerConfig
from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.execution.pivot_context import PivotExecutionContext, PivotExecutionContextResolver
from src.core.execution.tool_gateway import ToolGateway
from src.core.execution.tool_plan import ToolPlan, build_tool_plan
from src.core.execution.tool_policy import ToolPolicy, ToolPolicyDecision
from src.core.execution.tool_result import ToolExecutionResult

__all__ = [
    "AdapterPolicyConfig",
    "AdapterResolution",
    "ConfiguredMCPClient",
    "ExecutionAdapter",
    "HttpRequestExecutionAdapter",
    "LocalShellAdapter",
    "MCPClient",
    "MCPExecutionAdapter",
    "MCPRuntimeConfig",
    "MCPServerConfig",
    "MCPToolCallResult",
    "NetnsShellAdapter",
    "PivotExecutionContext",
    "PivotExecutionContextResolver",
    "ProxyShellAdapter",
    "TunnelAdapter",
    "ToolExecutionResult",
    "ToolGateway",
    "ToolAdapterResolver",
    "ToolBinding",
    "ToolPlan",
    "ToolPolicy",
    "ToolPolicyDecision",
    "UnavailableMCPClient",
    "build_tool_plan",
]
