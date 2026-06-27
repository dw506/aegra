from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.models.ag import GraphRef
from src.core.execution.execution_agent import ExecutionAgent
from src.core.execution.models import ExecutionRequest, ExecutionResult


class RecordingMCP:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def call_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "success": True,
            "stdout": "{\"precheck\": true}",
            "stderr": "",
            "exit_code": 0,
            "metadata": {"parsed_output": {"runtime_hints": {"validation_precheck_passed": True}}},
        }


class FakeStageLLM:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []
        self.config = type("Config", (), {"model": "gpt-test"})()

    def complete_chat(self, **kwargs: Any) -> PackyLLMResponse:
        self.calls.append(kwargs)
        payload = self.decisions.pop(0)
        return PackyLLMResponse(model="gpt-test", text=json.dumps(payload), usage=None)


def test_llm_driven_stage_agent_runs_stage_execution_request_main_path() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "finish",
                "status": "succeeded",
                "summary": "finished Collect service evidence",
                "confidence": 0.9,
                "handoff_suggestion": {
                    "suggested_agent": "vuln_analysis_agent",
                    "suggested_capability": "analysis",
                    "reason": "service evidence is ready for vulnerability analysis",
                    "confidence": 0.8,
                    "required_context_refs": ["kg:svc-1"],
                },
            }
        ]
    )
    request = ExecutionRequest(
        operation_id="op-1",
        cycle_index=2,
        agent_name="recon_agent",
        capability="recon",
        objective="Collect service evidence",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        required_context={"scope": "authorized"},
        success_criteria=["service evidence recorded"],
        risk_level="low",
        max_steps=3,
        graph_summary={"min_summary": {"kg_node_count": 1}},
        graph_history={"recent": []},
        runtime_context={"operation_id": "op-1"},
        policy_context={"authorized": True},
        mcp_tool_catalog={},
    )

    result = ExecutionAgent(llm_client=llm).run(request)

    assert isinstance(result, ExecutionResult)
    assert result.execution_id == "execution-op-1-2-recon_agent"
    assert result.summary == "finished Collect service evidence"
    assert len(llm.calls) == 1
    assert result.handoff_suggestion is not None
    assert result.handoff_suggestion.suggested_agent == "vuln_analysis_agent"
    assert result.handoff_suggestion.required_context_refs == ["kg:svc-1"]


def test_execution_agent_main_path_has_no_task_graph_dependency() -> None:
    source = Path("src/core/execution/execution_agent.py").read_text(encoding="utf-8")

    assert "TaskGraph" not in source
    assert "src.core.models.tg" not in source


def test_exploit_validation_precheck_infers_missing_target_url() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "call_mcp_tool",
                "server_id": "pentest-tools",
                "tool_name": "validation_precheck",
                "arguments": {"profile_id": "struts2-s2-045"},
                "reasoning_summary": "run bounded precheck",
            },
            {"action": "finish", "status": "succeeded", "summary": "validation precheck completed"},
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-validation-url",
        cycle_index=3,
        agent_name="exploit_validation_agent",
        capability="exploit",
        objective="Safely validate Struts2 candidate",
        target_refs=[
            GraphRef(graph="kg", ref_id="service::10.20.0.22:8080/tcp", ref_type="Service"),
        ],
        required_context={"profile_id": "struts2-s2-045"},
        success_criteria=["validation precheck receives a target_url"],
        risk_level="medium",
        max_steps=2,
        graph_summary={},
        graph_history={},
        runtime_context={},
        policy_context={"authorized_hosts": ["10.20.0.0/24"]},
        mcp_tool_catalog={
            "pentest-tools": {
                "tools": [
                    {
                        "name": "validation_precheck",
                        "category": "vuln",
                        "requires_authorization": True,
                    }
                ]
            }
        },
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"]["target_url"] == "http://10.20.0.22:8080/"
    assert result.tool_trace[0].arguments["target_url"] == "http://10.20.0.22:8080/"


def test_stage_agent_blocks_tool_not_in_supplied_catalog() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "call_mcp_tool",
                "server_id": "pentest-tools",
                "tool_name": "optional_missing_tool",
                "arguments": {"url": "http://example.test/"},
            },
            {"action": "finish", "status": "partial", "summary": "missing catalog tool recorded"},
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-missing-tool",
        cycle_index=1,
        agent_name="recon_agent",
        capability="recon",
        objective="Respect supplied tool catalog",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "http_probe"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "partial"
    assert mcp.calls == []
    assert result.tool_trace[0].exit_code == "tool_not_in_catalog"


def test_stage_agent_normalizes_url_target_for_web_tools() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "call_mcp_tool",
                "server_id": "pentest-tools",
                "tool_name": "web_fingerprint",
                "arguments": {"target": "http://10.0.0.5:8080"},
            },
            {"action": "finish", "status": "succeeded", "summary": "fingerprint completed"},
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-url-tool",
        cycle_index=1,
        agent_name="recon_agent",
        capability="recon",
        objective="Normalize URL target",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "web_fingerprint"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"]["url"] == "http://10.0.0.5:8080"
    assert result.tool_trace[0].arguments["url"] == "http://10.0.0.5:8080"
    assert result.tool_trace[0].arguments["operation_id"] == "op-url-tool"


def test_stage_agent_defaults_missing_server_id_to_pentest_tools_and_injects_trace_context() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "call_mcp_tool",
                "tool_name": "nmap_scan",
                "arguments": {"target": "127.0.0.1"},
            },
            {"action": "finish", "status": "succeeded", "summary": "scan completed"},
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-default-server",
        cycle_index=4,
        agent_name="recon_agent",
        capability="recon",
        objective="Default server id",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "nmap_scan"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["server_id"] == "pentest-tools"
    assert mcp.calls[0]["arguments"]["operation_id"] == "op-default-server"
    assert mcp.calls[0]["arguments"]["trace_id"] == "4-recon_agent-1-nmap_scan"


def test_stage_agent_accepts_finish_data_alias_and_preserves_structured_output() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "finish",
                "data": {
                    "status": "completed",
                    "summary": "recon produced structured targets",
                    "hosts_up": ["198.51.100.10"],
                    "service_discovery": [
                        {"host": "198.51.100.10", "port": 8080, "protocol": "tcp", "service": "http"}
                    ],
                    "evidence_refs": ["runtime://tool-output/nmap"],
                },
            }
        ]
    )
    request = ExecutionRequest(
        operation_id="op-finish-data",
        cycle_index=3,
        agent_name="recon_agent",
        capability="recon",
        objective="Preserve structured finish output",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert result.status == "succeeded"
    assert result.evidence_refs == ["runtime://tool-output/nmap"]
    structured = [item for item in result.observations if item.get("category") == "stage_structured_output"]
    assert structured
    assert structured[0]["hosts_up"] == ["198.51.100.10"]
    assert structured[0]["service_discovery"][0]["port"] == 8080


def test_stage_agent_accepts_execution_result_alias_and_string_observations() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "finish",
                "execution_result": {
                    "status": "completed",
                    "summary": "recon noted scope",
                    "observations": ["Target remains within authorized DMZ scope."],
                },
            }
        ]
    )
    request = ExecutionRequest(
        operation_id="op-execution-result-alias",
        cycle_index=1,
        agent_name="recon_agent",
        capability="recon",
        objective="Normalize finish wrapper",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert result.status == "succeeded"
    assert result.observations == [{"type": "note", "detail": "Target remains within authorized DMZ scope."}]


def test_stage_agent_repairs_invalid_finish_payload_once() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "finish",
                "result": {
                    "status": "completed",
                    "summary": "bad confidence payload",
                    "confidence": "not-a-number",
                },
            },
            {
                "status": "completed",
                "summary": "repaired payload",
                "observations": ["Repaired without adding facts."],
                "confidence": 0.7,
            },
        ]
    )
    request = ExecutionRequest(
        operation_id="op-repair-finish",
        cycle_index=1,
        agent_name="vuln_analysis_agent",
        capability="analysis",
        objective="Repair invalid finish",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert len(llm.calls) == 2
    assert result.status == "succeeded"
    assert result.summary == "repaired payload"
    assert result.observations == [{"type": "note", "detail": "Repaired without adding facts."}]


def test_stage_agent_parses_json_summary_and_candidate_findings() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "finish",
                "summary": json.dumps(
                    {
                        "status": "completed",
                        "analysis": {
                            "service_fingerprints": [
                                {
                                    "host": "198.51.100.20",
                                    "port": 80,
                                    "protocol": "http",
                                    "improved_fingerprint": {"application": "Example App"},
                                }
                            ],
                            "candidate_findings": [
                                {
                                    "target": "http://198.51.100.20/",
                                    "type": "application_identification",
                                    "statement": "Example App identified from page title.",
                                }
                            ],
                        },
                    }
                ),
            }
        ]
    )
    request = ExecutionRequest(
        operation_id="op-json-summary",
        cycle_index=2,
        agent_name="vuln_analysis_agent",
        capability="analysis",
        objective="Analyze fingerprints",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert result.status == "succeeded"
    assert result.findings[0]["summary"] == "Example App identified from page title."
    structured = [item for item in result.observations if item.get("category") == "stage_structured_output"]
    assert structured[0]["analysis"]["service_fingerprints"][0]["port"] == 80


def test_stage_agent_empty_success_finish_requests_replan() -> None:
    llm = FakeStageLLM([{"action": "finish"}])
    request = ExecutionRequest(
        operation_id="op-empty-finish",
        cycle_index=9,
        agent_name="goal_agent",
        capability="goal",
        objective="Verify goal",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert result.status == "needs_replan"
    assert "no tool results" in (result.replan_recommendation or "")


def test_stage_agent_returns_tool_server_unavailable_for_unavailable_catalog_server() -> None:
    llm = FakeStageLLM(
        [
            {
                "action": "call_mcp_tool",
                "tool_name": "nmap_scan",
                "arguments": {"target": "127.0.0.1"},
            }
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-unavailable-server",
        cycle_index=1,
        agent_name="recon_agent",
        capability="recon",
        objective="Unavailable server",
        max_steps=1,
        mcp_tool_catalog={
            "pentest-tools": {
                "available": False,
                "error": "MCP is not configured",
                "tools": [{"name": "nmap_scan"}],
            }
        },
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert mcp.calls == []
    assert result.tool_trace[0].exit_code == "tool_server_unavailable"
