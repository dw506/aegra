from __future__ import annotations

import json
from typing import Any

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.execution.mcp_client import MCPToolCallResult
from src.core.execution.execution_agent import ExecutionAgent
from src.core.execution.models import ExecutionRequest


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


class SequenceLLM:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)
        self.config = type("Config", (), {"model": "gpt-test"})()

    def complete_chat(self, **_: Any) -> PackyLLMResponse:
        return PackyLLMResponse(model="gpt-test", text=json.dumps(self.decisions.pop(0)))


class MCP:
    def call_tool(self, **_: Any) -> dict[str, Any]:
        return {"success": True, "stdout": "ok", "metadata": {}}


class StructuredMCP:
    def call_tool(self, **_: Any) -> MCPToolCallResult:
        parsed = {
            "entities": [
                {
                    "type": "web_fingerprint",
                    "url": "http://10.20.0.21:3000/",
                    "title": "OWASP Juice Shop",
                }
            ],
            "evidence": [{"kind": "http_probe", "url": "http://10.20.0.21:3000/"}],
            "writeback_hints": {"observation_category": "web_fingerprint"},
        }
        return MCPToolCallResult(
            success=True,
            stdout='{"title":"OWASP Juice Shop"}',
            exit_code=200,
            content={"parsed": parsed, "raw_output_ref": "var/runtime/op-tools/tool-outputs/fp.json"},
        )


def test_recon_agent_passes_catalog_tool_without_stage_allowlist_filtering() -> None:
    result = ExecutionAgent(llm_client=StaticLLM("safe_vuln_validate"), mcp_client=MCP()).run(
        ExecutionRequest(
            operation_id="op-tools",
            cycle_index=1,
            agent_name="recon_agent",
            capability="recon",
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


def test_server_prefixed_tool_name_resolves_to_bare_catalog_tool() -> None:
    # LLMs sometimes namespace the tool as "<server_id>.tool"; it must still resolve
    # to the bare catalog tool instead of failing as tool_not_in_catalog.
    result = ExecutionAgent(llm_client=StaticLLM("mcp.safe_vuln_validate"), mcp_client=MCP()).run(
        ExecutionRequest(
            operation_id="op-tools",
            cycle_index=1,
            agent_name="recon_agent",
            capability="recon",
            objective="recon",
            policy_context={"authorized": True},
            max_steps=1,
            mcp_tool_catalog={"mcp": {"tools": [{"name": "safe_vuln_validate", "category": "exploit", "requires_authorization": False}]}},
        )
    )

    assert result.tool_trace[0].exit_code != "tool_not_in_catalog"
    assert result.tool_trace[0].tool_name == "safe_vuln_validate"
    assert result.tool_trace[0].policy_check["allowed"] is True


def test_policy_denylist_is_audit_only_without_blocking() -> None:
    result = ExecutionAgent(llm_client=StaticLLM("pivot_route_probe"), mcp_client=MCP()).run(
        ExecutionRequest(
            operation_id="op-tools",
            cycle_index=1,
            agent_name="goal_agent",
            capability="goal",
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


def test_max_steps_partial_preserves_successful_tool_evidence_and_parsed_output() -> None:
    result = ExecutionAgent(llm_client=StaticLLM("web_fingerprint"), mcp_client=StructuredMCP()).run(
        ExecutionRequest(
            operation_id="op-tools",
            cycle_index=2,
            agent_name="recon_agent",
            capability="recon",
            objective="fingerprint web service",
            policy_context={"authorized": True},
            max_steps=1,
            mcp_tool_catalog={"mcp": {"tools": [{"name": "web_fingerprint", "category": "fingerprint"}]}},
        )
    )

    assert result.status == "partial"
    assert result.runtime_hints["max_steps_exhausted"] is True
    assert len(result.evidence) == 1
    assert result.evidence_refs == ["var/runtime/op-tools/tool-outputs/fp.json"]
    assert result.tool_trace[0].raw_output_ref == "var/runtime/op-tools/tool-outputs/fp.json"
    assert result.tool_trace[0].parsed_output["entities"][0]["title"] == "OWASP Juice Shop"


def test_need_replan_preserves_successful_tool_evidence_and_parsed_output() -> None:
    llm = SequenceLLM(
        [
            {
                "action": "call_mcp_tool",
                "server_id": "mcp",
                "tool_name": "web_fingerprint",
                "arguments": {},
            },
            {
                "action": "need_replan",
                "summary": "need another target",
            },
        ]
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=StructuredMCP()).run(
        ExecutionRequest(
            operation_id="op-tools",
            cycle_index=2,
            agent_name="recon_agent",
            capability="recon",
            objective="fingerprint web service",
            policy_context={"authorized": True},
            max_steps=2,
            mcp_tool_catalog={"mcp": {"tools": [{"name": "web_fingerprint", "category": "fingerprint"}]}},
        )
    )

    # When LLM emits need_replan but at least one tool succeeded, status is 'partial'
    assert result.status == "partial"
    assert len(result.evidence) == 1
    assert result.evidence_refs == ["var/runtime/op-tools/tool-outputs/fp.json"]
    assert result.tool_trace[0].parsed_output["entities"][0]["title"] == "OWASP Juice Shop"
