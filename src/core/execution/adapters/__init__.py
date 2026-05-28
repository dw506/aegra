"""Execution adapters."""

from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.adapters.http_request_adapter import HttpRequestExecutionAdapter
from src.core.execution.adapters.local_shell_adapter import LocalShellAdapter
from src.core.execution.adapters.mcp_adapter import MCPExecutionAdapter
from src.core.execution.adapters.netns_shell_adapter import NetnsShellAdapter
from src.core.execution.adapters.proxy_shell_adapter import ProxyShellAdapter
from src.core.execution.adapters.tunnel_adapter import TunnelAdapter

__all__ = [
    "ExecutionAdapter",
    "HttpRequestExecutionAdapter",
    "LocalShellAdapter",
    "MCPExecutionAdapter",
    "NetnsShellAdapter",
    "ProxyShellAdapter",
    "TunnelAdapter",
]
