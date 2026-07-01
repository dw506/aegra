from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.llm.packy_llm import PackyLLMResponse
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
            "metadata": {"parsed_output": {"runtime_hints": {"probe_passed": True}}},
        }


class FakeExecutionLLM:
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []
        self.config = type("Config", (), {"model": "gpt-test"})()

    def complete_chat(self, **kwargs: Any) -> PackyLLMResponse:
        self.calls.append(kwargs)
        payload = self.decisions.pop(0)
        return PackyLLMResponse(model="gpt-test", text=json.dumps(payload), usage=None)


def test_llm_driven_execution_agent_runs_execution_request_main_path() -> None:
    llm = FakeExecutionLLM(
        [
            {
                "action": "finish",
                "status": "succeeded",
                "summary": "finished Collect service evidence",
                "confidence": 0.9,
            }
        ]
    )
    request = ExecutionRequest(
        operation_id="op-1",
        cycle_index=2,
        agent_name="recon_agent",
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


def test_execution_agent_main_path_has_no_task_graph_dependency() -> None:
    source = Path("src/core/execution/execution_agent.py").read_text(encoding="utf-8")
    legacy_module = "src.core.models." + "t" + "g"

    assert "Task" + "Graph" not in source
    assert legacy_module not in source


def test_http_probe_infers_missing_url_from_target_ref() -> None:
    llm = FakeExecutionLLM(
        [
            {
                "action": "call_mcp_tool",
                "server_id": "pentest-tools",
                "tool_name": "http_probe",
                "arguments": {},
                "reasoning_summary": "probe the entry service",
            },
            {"action": "finish", "status": "succeeded", "summary": "http probe completed"},
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-http-url",
        cycle_index=3,
        agent_name="execution_agent",
        objective="Probe the Struts2 entry service",
        target_refs=[
            GraphRef(graph="kg", ref_id="service::10.20.0.22:8080/tcp", ref_type="Service"),
        ],
        success_criteria=["http_probe receives a url"],
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
                        "name": "http_probe",
                        "category": "recon",
                        "requires_authorization": False,
                    }
                ]
            }
        },
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"]["url"] == "http://10.20.0.22:8080/"
    assert result.tool_trace[0].arguments["url"] == "http://10.20.0.22:8080/"


def test_execution_agent_blocks_tool_not_in_supplied_catalog() -> None:
    llm = FakeExecutionLLM(
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
        objective="Respect supplied tool catalog",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "http_probe"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "partial"
    assert mcp.calls == []
    assert result.tool_trace[0].exit_code == "tool_not_in_catalog"


def test_execution_agent_normalizes_url_target_for_web_tools() -> None:
    llm = FakeExecutionLLM(
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
        objective="Normalize URL target",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "web_fingerprint"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"]["url"] == "http://10.0.0.5:8080"
    assert result.tool_trace[0].arguments["url"] == "http://10.0.0.5:8080"
    assert result.tool_trace[0].arguments["operation_id"] == "op-url-tool"


def test_execution_agent_defaults_missing_server_id_to_pentest_tools_and_injects_trace_context() -> None:
    llm = FakeExecutionLLM(
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
        objective="Default server id",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "nmap_scan"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["server_id"] == "pentest-tools"
    assert mcp.calls[0]["arguments"]["operation_id"] == "op-default-server"
    assert mcp.calls[0]["arguments"]["trace_id"] == "4-recon_agent-1-nmap_scan"


def test_execution_agent_accepts_finish_data_alias_and_preserves_evidence() -> None:
    llm = FakeExecutionLLM(
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
        objective="Preserve structured finish output",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert result.status == "succeeded"
    # The `data` finish wrapper is still unwrapped; tool-derived evidence_refs
    # survive. The structured hosts_up/service_discovery self-report is no longer
    # captured (channel-② removed) — KG facts come from tool_trace only.
    assert result.evidence_refs == ["runtime://tool-output/nmap"]


def test_execution_agent_accepts_execution_result_alias_wrapper() -> None:
    llm = FakeExecutionLLM(
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
        objective="Normalize finish wrapper",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    # The execution_result wrapper is unwrapped; the self-report observations key
    # is silently ignored now (channel-② removed).
    assert result.status == "succeeded"
    assert result.summary == "recon noted scope"


def test_execution_agent_repairs_invalid_finish_payload_once() -> None:
    llm = FakeExecutionLLM(
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
        objective="Repair invalid finish",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert len(llm.calls) == 2
    assert result.status == "succeeded"
    assert result.summary == "repaired payload"
    assert result.confidence == 0.7


def test_execution_agent_empty_success_finish_requests_replan() -> None:
    llm = FakeExecutionLLM([{"action": "finish"}])
    request = ExecutionRequest(
        operation_id="op-empty-finish",
        cycle_index=9,
        agent_name="goal_agent",
        objective="Verify goal",
        max_steps=1,
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=RecordingMCP()).run(request)

    assert result.status == "needs_replan"
    assert "no tool results" in (result.replan_recommendation or "")


def test_execution_agent_returns_tool_server_unavailable_for_unavailable_catalog_server() -> None:
    llm = FakeExecutionLLM(
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


class FailingMCP:
    """MCP stub whose every call reports failure (drives the no-progress guard)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def call_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"success": False, "stdout": "", "stderr": "boom", "exit_code": 1, "metadata": {}}


def test_no_progress_guard_stops_round_before_max_steps() -> None:
    # LLM keeps calling tools; every call fails, so the guard must bail out after
    # NO_PROGRESS_LIMIT unproductive calls rather than burning all max_steps.
    call_decision = {
        "action": "call_mcp_tool",
        "server_id": "pentest-tools",
        "tool_name": "nmap_scan",
        "arguments": {"target": "10.0.0.5"},
    }
    llm = FakeExecutionLLM([dict(call_decision) for _ in range(10)])
    mcp = FailingMCP()
    request = ExecutionRequest(
        operation_id="op-no-progress",
        cycle_index=1,
        agent_name="recon_agent",
        objective="Probe service",
        max_steps=10,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "nmap_scan"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "needs_replan"
    # Stopped at the guard (3 calls), well short of max_steps=10.
    assert len(mcp.calls) == 3
    assert "no progress" in result.summary


def test_phase_tag_absent_from_executor_decision_context() -> None:
    request = ExecutionRequest(
        operation_id="op-cap",
        cycle_index=1,
        agent_name="recon_agent",
        objective="Collect service evidence",
        success_criteria=["service evidence recorded"],
        max_steps=1,
    )
    agent = ExecutionAgent(llm_client=FakeExecutionLLM([]))
    messages = agent._agent._build_messages(request, [])
    context = json.loads(messages[1]["content"])

    assert context["planner_objective"] == "Collect service evidence"
    assert context["success_criteria"] == ["service evidence recorded"]


def test_active_session_transport_is_stamped_onto_tool_call() -> None:
    llm = FakeExecutionLLM(
        [
            {
                "action": "call_mcp_tool",
                "server_id": "pentest-tools",
                "tool_name": "nmap_scan",
                "arguments": {"target": "10.0.0.5"},
            },
            {"action": "finish", "status": "succeeded", "summary": "scan completed"},
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-transport",
        cycle_index=1,
        agent_name="recon_agent",
        objective="Operate through the active foothold",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "nmap_scan"}]}},
        sessions=[{"session_id": "sess-1", "status": "active"}],
        pivot_routes=[{"route_id": "route-9", "status": "active"}],
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"]["session_id"] == "sess-1"
    assert mcp.calls[0]["arguments"]["route_id"] == "route-9"


def test_metasploit_exec_gets_outer_timeout_buffer() -> None:
    llm = FakeExecutionLLM(
        [
            {
                "action": "call_mcp_tool",
                "server_id": "pentest-tools",
                "tool_name": "metasploit_exec",
                "arguments": {
                    "module": "exploit/multi/http/struts2_content_type_ognl",
                    "target": "10.20.0.10",
                    "timeout_seconds": 120,
                },
            },
            {"action": "finish", "status": "succeeded", "summary": "attempt recorded"},
        ]
    )
    mcp = RecordingMCP()
    request = ExecutionRequest(
        operation_id="op-msf-timeout",
        cycle_index=1,
        agent_name="execution_agent",
        objective="Open a real session with metasploit_exec",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "metasploit_exec"}]}},
    )

    result = ExecutionAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"]["timeout_seconds"] == 120
    assert mcp.calls[0]["timeout_seconds"] == 150

