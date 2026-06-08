from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.models.ag import GraphRef
from src.core.planning.models import PlannerDecision
from src.core.stage.agents import ExploitValidationAgent, ReconAgent
from src.core.stage.base_stage_agent import StageAgentDecision, StageToolCall
from src.core.stage.dispatcher import StageDispatcher
from src.core.stage.models import StageExecutionRequest, StageResult
from src.core.stage.registry import StageAgentRegistry


class FinishAdvisor:
    def __init__(self) -> None:
        self.requests: list[StageExecutionRequest] = []

    def decide(self, **kwargs: Any) -> StageAgentDecision:
        request = kwargs["request"]
        self.requests.append(request)
        return StageAgentDecision(
            action="finish",
            rationale="request satisfied",
            finish={
                "status": "succeeded",
                "summary": f"finished {request.objective}",
                "confidence": 0.9,
                "handoff_suggestion": {
                    "suggested_agent": "vuln_analysis_agent",
                    "suggested_stage": "VULN_ANALYSIS_STAGE",
                    "reason": "service evidence is ready for vulnerability analysis",
                    "confidence": 0.8,
                    "required_context_refs": ["kg:svc-1"],
                },
            },
        )


class ValidationPrecheckAdvisor:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: Any) -> StageAgentDecision:
        self.calls += 1
        if kwargs["memory"]:
            return StageAgentDecision(
                action="finish",
                rationale="precheck completed",
                finish={"status": "succeeded", "summary": "validation precheck completed"},
            )
        return StageAgentDecision(
            action="call_tool",
            rationale="run bounded precheck",
            tool_call=StageToolCall(
                server_id="pentest-tools",
                tool_name="validation_precheck",
                arguments={"profile_id": "struts2-s2-045"},
            ),
        )


class UrlTargetAdvisor:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: Any) -> StageAgentDecision:
        self.calls += 1
        if kwargs["memory"]:
            return StageAgentDecision(
                action="finish",
                rationale="fingerprint completed",
                finish={"status": "succeeded", "summary": "fingerprint completed"},
            )
        return StageAgentDecision(
            action="call_tool",
            rationale="fingerprint URL target",
            tool_call=StageToolCall(
                server_id="pentest-tools",
                tool_name="web_fingerprint",
                arguments={"target": "http://10.0.0.5:8080"},
            ),
        )


class MissingToolAdvisor:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, **kwargs: Any) -> StageAgentDecision:
        self.calls += 1
        if kwargs["memory"]:
            return StageAgentDecision(
                action="finish",
                rationale="missing tool recorded",
                finish={"status": "partial", "summary": "missing catalog tool recorded"},
            )
        return StageAgentDecision(
            action="call_tool",
            rationale="try optional tool",
            tool_call=StageToolCall(
                server_id="pentest-tools",
                tool_name="optional_missing_tool",
                arguments={"url": "http://example.test/"},
            ),
        )


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


def test_stage_agent_registry_still_registers_five_agents() -> None:
    registry = StageAgentRegistry.default()

    assert registry.resolve("RECON_STAGE").agent_name == "recon_agent"
    assert registry.resolve("VULN_ANALYSIS_STAGE").agent_name == "vuln_analysis_agent"
    assert registry.resolve("EXPLOIT_STAGE").agent_name == "exploit_validation_agent"
    assert registry.resolve("ACCESS_PIVOT_STAGE").agent_name == "access_pivot_agent"
    assert registry.resolve("GOAL_STAGE").agent_name == "goal_agent"


def test_exploit_validation_agent_name_matches_planner_and_dispatcher() -> None:
    registry = StageAgentRegistry.default()
    agent = registry.resolve("EXPLOIT_STAGE")
    decision = PlannerDecision(
        operation_id="op-1",
        cycle_index=0,
        decision="dispatch_agent",
        selected_agent="exploit_validation_agent",
        selected_stage="EXPLOIT_STAGE",
        objective="Validate exploitability",
        risk_level="medium",
        max_steps=2,
        confidence=0.8,
    )

    assert agent.agent_name == "exploit_validation_agent"
    assert registry.resolve_agent("exploit_validation_agent").stage_type == "EXPLOIT_STAGE"
    assert decision.selected_agent == agent.agent_name


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
                    "suggested_stage": "VULN_ANALYSIS_STAGE",
                    "reason": "service evidence is ready for vulnerability analysis",
                    "confidence": 0.8,
                    "required_context_refs": ["kg:svc-1"],
                },
            }
        ]
    )
    request = StageExecutionRequest(
        operation_id="op-1",
        cycle_index=2,
        agent_name="recon_agent",
        stage_type="RECON_STAGE",
        objective="Collect service evidence",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        required_context={"scope": "authorized"},
        success_criteria=["service evidence recorded"],
        risk_level="low",
        max_steps=3,
        kg_snapshot={"hosts": ["host-1"]},
        ag_process_history={"recent": []},
        runtime_context={"operation_id": "op-1"},
        policy_context={"authorized": True},
        mcp_tool_catalog={},
    )

    result = ReconAgent(llm_client=llm).run(request)

    assert isinstance(result, StageResult)
    assert result.stage_task_id == "stage-op-1-2-recon_agent"
    assert result.summary == "finished Collect service evidence"
    assert len(llm.calls) == 1
    assert result.handoff_suggestion is not None
    assert result.handoff_suggestion.suggested_agent == "vuln_analysis_agent"
    assert result.handoff_suggestion.required_context_refs == ["kg:svc-1"]


def test_base_stage_agent_main_path_has_no_task_graph_dependency() -> None:
    source = Path("src/core/stage/base_stage_agent.py").read_text(encoding="utf-8")

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
    request = StageExecutionRequest(
        operation_id="op-validation-url",
        cycle_index=3,
        agent_name="exploit_validation_agent",
        stage_type="EXPLOIT_STAGE",
        objective="Safely validate Struts2 candidate",
        target_refs=[
            GraphRef(graph="kg", ref_id="service::10.20.0.22:8080/tcp", ref_type="Service"),
        ],
        required_context={"profile_id": "struts2-s2-045"},
        success_criteria=["validation precheck receives a target_url"],
        risk_level="medium",
        max_steps=2,
        kg_snapshot={},
        ag_process_history={},
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

    result = ExploitValidationAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"]["target_url"] == "http://10.20.0.22:8080/"
    assert result.tool_traces[0].arguments["target_url"] == "http://10.20.0.22:8080/"


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
    request = StageExecutionRequest(
        operation_id="op-missing-tool",
        cycle_index=1,
        agent_name="recon_agent",
        stage_type="RECON_STAGE",
        objective="Respect supplied tool catalog",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "http_probe"}]}},
    )

    result = ReconAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "partial"
    assert mcp.calls == []
    assert result.tool_traces[0].exit_code == "tool_not_in_catalog"


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
    request = StageExecutionRequest(
        operation_id="op-url-tool",
        cycle_index=1,
        agent_name="recon_agent",
        stage_type="RECON_STAGE",
        objective="Normalize URL target",
        max_steps=2,
        mcp_tool_catalog={"pentest-tools": {"tools": [{"name": "web_fingerprint"}]}},
    )

    result = ReconAgent(llm_client=llm, mcp_client=mcp).run(request)

    assert result.status == "succeeded"
    assert mcp.calls[0]["arguments"] == {"url": "http://10.0.0.5:8080"}
    assert result.tool_traces[0].arguments == {"url": "http://10.0.0.5:8080"}
