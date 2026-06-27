"""Small MCP client protocol used by the executor to call tools over MCP."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class MCPToolCallResult(BaseModel):
    """Normalized MCP tool call response returned to the executor."""

    model_config = ConfigDict(extra="allow")

    success: bool = True
    content: Any = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPClient(Protocol):
    """Minimal client contract required by MCPExecutionAdapter."""

    def list_tools(self) -> dict[str, Any]:
        """Return a catalog of configured MCP tools when available."""

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
    ) -> MCPToolCallResult | dict[str, Any]:
        """Invoke one MCP tool and return a normalized or dict-like result."""


class UnavailableMCPClient:
    """Default client used when no MCP transport has been configured."""

    def list_tools(self) -> dict[str, Any]:
        return {"available": False, "error": "MCP is not configured"}

    def call_tool(
        self,
        *,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
    ) -> MCPToolCallResult:
        return MCPToolCallResult(
            success=False,
            exit_code="mcp_unavailable",
            stderr=f"MCP server '{server_id}' is not available for tool '{tool_name}'",
        )


__all__ = ["MCPClient", "MCPToolCallResult", "UnavailableMCPClient"]
