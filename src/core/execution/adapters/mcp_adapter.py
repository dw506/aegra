"""MCP execution adapter."""

from __future__ import annotations

import json
from typing import Any

from src.core.execution.mcp_client import MCPClient, MCPToolCallResult, UnavailableMCPClient
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult


class MCPExecutionAdapter:
    """Execute adapter-neutral ToolPlans through MCP tools."""

    name = "mcp"

    def __init__(self, client: MCPClient | None = None, *, default_server_id: str | None = None) -> None:
        self._client = client or UnavailableMCPClient()
        self._default_server_id = default_server_id

    def supports(self, plan: ToolPlan) -> bool:
        if plan.adapter not in {None, self.name}:
            return False
        server_id, _ = self.resolve_binding(plan)
        if not server_id:
            return False
        return self._client.is_available(server_id)

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        server_id, tool_name = self.resolve_binding(plan)
        if not server_id:
            return self._result(
                plan,
                success=False,
                exit_code="missing_mcp_server",
                stderr=f"MCP server is not configured for tool '{plan.tool}'",
                server_id=server_id,
                tool_name=tool_name,
            )
        if not self._client.is_available(server_id):
            return self._result(
                plan,
                success=False,
                exit_code="mcp_unavailable",
                stderr=f"MCP server '{server_id}' is not available",
                server_id=server_id,
                tool_name=tool_name,
            )

        arguments = dict(plan.args)
        if plan.target is not None:
            arguments.setdefault("target", plan.target)
        try:
            raw_result = self._client.call_tool(
                server_id=server_id,
                tool_name=tool_name,
                arguments=arguments,
                timeout_seconds=plan.timeout_seconds,
            )
        except TimeoutError as exc:
            return self._result(
                plan,
                success=False,
                exit_code="timeout",
                stderr=str(exc) or f"MCP tool '{tool_name}' timed out",
                server_id=server_id,
                tool_name=tool_name,
                metadata={"timed_out": True},
            )
        except Exception as exc:
            return self._result(
                plan,
                success=False,
                exit_code="mcp_error",
                stderr=str(exc),
                server_id=server_id,
                tool_name=tool_name,
            )

        result = self._coerce_result(raw_result)
        stdout = result.stdout or self._content_to_stdout(result.content)
        metadata = {
            "mcp_server_id": server_id,
            "mcp_tool_name": tool_name,
            "tool_plan": plan.model_dump(mode="json"),
            "mcp_result": result.metadata,
        }
        return ToolExecutionResult(
            adapter=self.name,
            tool=plan.tool,
            success=result.success,
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=result.stderr,
            metadata=metadata,
        )

    def resolve_binding(self, plan: ToolPlan) -> tuple[str | None, str]:
        server_id = (
            plan.metadata.get("mcp_server_id")
            or plan.args.get("mcp_server_id")
            or self._default_server_id
        )
        tool_name = plan.metadata.get("mcp_tool_name") or plan.args.get("mcp_tool_name") or plan.tool
        return (str(server_id) if server_id is not None else None, str(tool_name))

    def _result(
        self,
        plan: ToolPlan,
        *,
        success: bool,
        exit_code: int | str | None,
        stderr: str,
        server_id: str | None,
        tool_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        payload = {
            "mcp_server_id": server_id,
            "mcp_tool_name": tool_name,
            "tool_plan": plan.model_dump(mode="json"),
        }
        if metadata:
            payload.update(metadata)
        return ToolExecutionResult(
            adapter=self.name,
            tool=plan.tool,
            success=success,
            exit_code=exit_code,
            stderr=stderr,
            metadata=payload,
        )

    @staticmethod
    def _coerce_result(value: MCPToolCallResult | dict[str, Any]) -> MCPToolCallResult:
        if isinstance(value, MCPToolCallResult):
            return value
        if isinstance(value, dict):
            return MCPToolCallResult.model_validate(value)
        if hasattr(value, "model_dump"):
            return MCPToolCallResult.model_validate(value.model_dump(mode="json"))
        raise TypeError(f"unsupported MCP tool result type: {type(value).__name__}")

    @staticmethod
    def _content_to_stdout(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True, sort_keys=True)


__all__ = ["MCPExecutionAdapter"]
