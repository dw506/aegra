from __future__ import annotations

import json
from typing import Any

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.stage.agents import GoalAgent, ReconAgent
from src.core.stage.models import StageExecutionRequest


class StaticLLM:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.config = type("Config", (), {"model": "gpt-test"})()

    def complete_chat(self, **_: Any) -> PackyLLMResponse:
        return PackyLLMResponse(
            model="gpt-test",
            text=json.dumps(
                {
                    "action": "call_mcp_tool",
                    "server_id": "mcp",
                    "tool_name": self.tool_name,
                    "arguments": {},
                    "reasoning_summary": "try tool",
                }
            ),
        )


class MCP:
    def call_tool(self, **_: Any) -> dict[str, Any]:
        return {"success": True, "stdout": "ok", "metadata": {}}


def test_recon_agent_passes_catalog_tool_without_stage_allowlist_filtering() -> None:
    result = ReconAgent(llm_client=StaticLLM("safe_vuln_validate"), mcp_client=MCP()).run(
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
    result = GoalAgent(llm_client=StaticLLM("pivot_route_probe"), mcp_client=MCP()).run(
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
