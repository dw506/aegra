from __future__ import annotations

from src.core.stage.agents import GoalAgent, ReconAgent
from src.core.stage.base_stage_agent import StageAgentDecision, StageToolCall
from src.core.stage.models import StageExecutionRequest


class StaticAdvisor:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def decide(self, **_):
        return StageAgentDecision(
            action="call_tool",
            rationale="try tool",
            tool_call=StageToolCall(server_id="mcp", tool_name=self.tool_name, arguments={}),
        )


class MCP:
    def call_tool(self, **_):
        return {"success": True, "stdout": "ok", "metadata": {}}


def test_recon_agent_passes_catalog_tool_without_stage_allowlist_filtering() -> None:
    result = ReconAgent(advisor=StaticAdvisor("safe_vuln_validate"), mcp_client=MCP()).run(
        StageExecutionRequest(
            operation_id="op-tools",
            cycle_index=1,
            agent_name="recon_agent",
            stage_type="RECON_STAGE",
            objective="recon",
            policy_context={"authorized": True},
            max_steps=1,
            mcp_tool_catalog={"mcp": {"tools": [{"name": "safe_vuln_validate", "category": "exploit", "requires_authorization": False}]}},
        )
    )

    assert result.status != "blocked"
    assert result.tool_trace[0].policy_check["allowed"] is True
    assert result.tool_trace[0].policy_check["metadata"]["policy_audit_only"] is True
    assert result.tool_trace[0].policy_check["metadata"]["original_allowed"] is True


def test_policy_denylist_is_audit_only_without_blocking() -> None:
    result = GoalAgent(advisor=StaticAdvisor("pivot_route_probe"), mcp_client=MCP()).run(
        StageExecutionRequest(
            operation_id="op-tools",
            cycle_index=1,
            agent_name="goal_agent",
            stage_type="GOAL_STAGE",
            objective="goal",
            policy_context={"authorized": True, "mcp_tool_denylist": ["pivot_route_probe"]},
            max_steps=1,
            mcp_tool_catalog={"mcp": {"tools": [{"name": "pivot_route_probe", "category": "pivot", "requires_authorization": False}]}},
        )
    )

    assert result.status != "blocked"
    assert result.tool_trace[0].policy_check["allowed"] is True
    assert result.tool_trace[0].policy_check["metadata"]["policy_audit_only"] is True
    assert result.tool_trace[0].policy_check["metadata"]["original_allowed"] is False
