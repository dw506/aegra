"""Pre-execution policy checks for ToolPlan dispatch."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.core.execution.tool_plan import ToolPlan
from src.core.models.runtime import RuntimeState
from src.core.runtime.policy import policy_from_runtime_state
from src.core.runtime.txt_trace_logger import TxtTraceLogger


class ToolPolicyDecision(BaseModel):
    """Decision returned before adapter execution."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason: str = Field(min_length=1)
    gate: str = Field(min_length=1)
    approval_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ToolPolicy:
    """Apply Aegra runtime policy gates before tool execution."""

    def evaluate(
        self,
        plan: ToolPlan,
        runtime_state: RuntimeState,
    ) -> ToolPolicyDecision:
        """Return ALLOW/DENY/APPROVAL decision for one plan."""

        if not (plan.command or plan.tool or "").strip():
            return ToolPolicyDecision(allowed=False, gate="tool_plan", reason="tool plan tool is empty")
        runtime_policy = policy_from_runtime_state(runtime_state)
        if plan.adapter == "mcp":
            server_id = self._string(plan.metadata.get("mcp_server_id") or plan.args.get("mcp_server_id"))
            tool_name = self._string(plan.metadata.get("mcp_tool_name") or plan.args.get("mcp_tool_name") or plan.tool)
            original = self._original_mcp_decision(
                runtime_policy=runtime_policy,
                server_id=server_id,
                tool_name=tool_name,
            )
            return self._audit_only(
                runtime_state,
                gate="mcp_policy",
                reason=str(original["reason"]),
                plan=plan,
                original_allowed=bool(original["allowed"]),
                server_id=server_id,
                tool_name=tool_name,
            )
        return ToolPolicyDecision(allowed=True, gate="tool_policy", reason="tool plan allowed")

    @staticmethod
    def _audit_only(
        runtime_state: RuntimeState,
        *,
        gate: str,
        reason: str,
        plan: ToolPlan,
        original_allowed: bool,
        server_id: str | None,
        tool_name: str | None,
    ) -> ToolPolicyDecision:
        TxtTraceLogger(runtime_state.operation_id).write_block(
            "POLICY_DECISION",
            "tool policy evaluated but not enforced",
            {
                "tool": plan.tool,
                "adapter": plan.adapter,
                "server_id": server_id,
                "tool_name": tool_name,
                "original_allowed": original_allowed,
                "original_reason": reason,
                "original_policy_name": gate,
                "final_allowed": True,
                "final_reason": "MCP policy is audit-only in authorized lab mode; execution continued.",
            },
        )
        return ToolPolicyDecision(
            allowed=True,
            gate=gate,
            reason="MCP policy is audit-only in authorized lab mode; execution continued.",
            metadata={
                "policy_audit_only": True,
                "original_allowed": original_allowed,
                "original_reason": reason,
                "mcp_server_id": server_id,
                "mcp_tool_name": tool_name,
                "adapter": plan.adapter,
                "plan": plan.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _original_mcp_decision(*, runtime_policy: object, server_id: str | None, tool_name: str | None) -> dict[str, object]:
        if server_id and server_id in set(runtime_policy.mcp_server_denylist):  # type: ignore[attr-defined]
            return {"allowed": False, "reason": f"MCP server '{server_id}' is denied"}
        if tool_name and tool_name in set(runtime_policy.mcp_tool_denylist):  # type: ignore[attr-defined]
            return {"allowed": False, "reason": f"MCP tool '{tool_name}' is denied"}
        if runtime_policy.mcp_server_allowlist and server_id not in set(runtime_policy.mcp_server_allowlist):  # type: ignore[attr-defined]
            return {"allowed": False, "reason": f"MCP server '{server_id}' is not allowlisted"}
        if runtime_policy.mcp_tool_allowlist and tool_name not in set(runtime_policy.mcp_tool_allowlist):  # type: ignore[attr-defined]
            return {"allowed": False, "reason": f"MCP tool '{tool_name}' is not allowlisted"}
        return {"allowed": True, "reason": "policy allowed MCP tool call"}

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
