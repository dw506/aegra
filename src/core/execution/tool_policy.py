"""Pre-execution policy checks for ToolPlan dispatch."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.core.execution.tool_plan import ToolPlan
from src.core.models.runtime import RuntimeState
from src.core.models.tg import BaseTaskNode
from src.core.runtime.policy import policy_from_runtime_state
from src.core.runtime.policy_gate import PolicyGate, PolicyGateAction
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
        *,
        task: BaseTaskNode | None = None,
    ) -> ToolPolicyDecision:
        """Return ALLOW/DENY/APPROVAL decision for one plan."""

        if task is not None:
            gate_decision = PolicyGate().evaluate(task, runtime_state=runtime_state)
            PolicyGate().audit(runtime_state, gate_decision)
            return ToolPolicyDecision(
                allowed=gate_decision.action == PolicyGateAction.ALLOW,
                reason=gate_decision.reason,
                gate=gate_decision.gate,
                approval_id=gate_decision.approval_id,
                metadata={"policy_gate_action": gate_decision.action.value, "plan": plan.model_dump(mode="json")},
            )
        if not (plan.command or plan.tool or "").strip():
            return ToolPolicyDecision(allowed=False, gate="tool_plan", reason="tool plan tool is empty")
        runtime_policy = policy_from_runtime_state(runtime_state)
        if plan.adapter == "mcp":
            server_id = self._string(plan.metadata.get("mcp_server_id") or plan.args.get("mcp_server_id"))
            tool_name = self._string(plan.metadata.get("mcp_tool_name") or plan.args.get("mcp_tool_name") or plan.tool)
            if server_id and server_id in set(runtime_policy.mcp_server_denylist):
                return self._audit_only(runtime_state, gate="mcp_policy", reason=f"MCP server '{server_id}' is denied", plan=plan)
            if tool_name and tool_name in set(runtime_policy.mcp_tool_denylist):
                return self._audit_only(runtime_state, gate="mcp_policy", reason=f"MCP tool '{tool_name}' is denied", plan=plan)
            if runtime_policy.mcp_server_allowlist and server_id not in set(runtime_policy.mcp_server_allowlist):
                return self._audit_only(runtime_state, gate="mcp_policy", reason=f"MCP server '{server_id}' is not allowlisted", plan=plan)
            if runtime_policy.mcp_tool_allowlist and tool_name not in set(runtime_policy.mcp_tool_allowlist):
                return self._audit_only(runtime_state, gate="mcp_policy", reason=f"MCP tool '{tool_name}' is not allowlisted", plan=plan)
        return ToolPolicyDecision(allowed=True, gate="tool_policy", reason="tool plan allowed")

    @staticmethod
    def _audit_only(runtime_state: RuntimeState, *, gate: str, reason: str, plan: ToolPlan) -> ToolPolicyDecision:
        TxtTraceLogger(runtime_state.operation_id).write_block(
            "POLICY_DECISION",
            "tool policy evaluated but not enforced",
            {
                "tool": plan.tool,
                "adapter": plan.adapter,
                "original_allowed": False,
                "original_reason": reason,
                "original_policy_name": gate,
                "final_allowed": True,
                "final_reason": "PolicyGate is audit-only; execution continued.",
            },
        )
        return ToolPolicyDecision(
            allowed=True,
            gate=gate,
            reason="PolicyGate is audit-only; execution continued.",
            metadata={
                "policy_audit_only": True,
                "original_allowed": False,
                "original_reason": reason,
                "plan": plan.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
