"""MCP-based tool execution.

Tools are invoked over MCP via :class:`ConfiguredMCPClient`. The former
client-side transport-adapter engine (netns/tunnel/proxy adapters, resolver,
tool plan/result, ToolGateway) has been removed: pivoting is resolved
server-side by the mcp_lab tools, so no client-side adapter selection is needed.
"""

from src.core.execution.configured_mcp_client import ConfiguredMCPClient, MCPRuntimeConfig, MCPServerConfig
from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient

__all__ = [
    "ConfiguredMCPClient",
    "MCPClient",
    "MCPRuntimeConfig",
    "MCPServerConfig",
    "MCPToolCallResult",
    "UnavailableMCPClient",
]
