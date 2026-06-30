"""Tests for the agentic planner loop (Step 3b): read-tool iteration + fallback."""

from __future__ import annotations

import json
from typing import Any

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.planning.graph_tools import PlannerGraphTools
from src.core.planning.planner import Planner


class ScriptedClient:
    """Returns a queued sequence of completion texts, one per turn."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.config = type("Config", (), {"model": "gpt-test"})()
        self.calls = 0

    def complete_chat(self, *, user_prompt: str, system_prompt: str | None = None, **_: Any) -> PackyLLMResponse:
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return PackyLLMResponse(model="gpt-test", text=text)


def _tools(state: RuntimeState) -> PlannerGraphTools:
    return PlannerGraphTools(
        operation_id="op-loop",
        cycle_index=1,
        kg=KnowledgeGraph(),
        ag=AttackGraph(),
        runtime_state=state,
    )


def _final_execute() -> str:
    return json.dumps(
        {
            "action": "execute",
            "directive": {
                "operation_id": "op-loop",
                "cycle_index": 1,
                "capability": "recon",
                "objective": "collect entry-zone service evidence",
                "target_refs": [],
                "risk_level": "low",
            },
            "reason": "need recon",
        }
    )


def test_planner_loop_drills_with_read_tools_then_decides() -> None:
    state = RuntimeState(operation_id="op-loop", execution=OperationRuntime(operation_id="op-loop"))
    state.execution.metadata["success_condition_progress"] = {"missing": ["entry_zone_service_discovered"]}
    tools = _tools(state)
    client = ScriptedClient(
        [
            json.dumps({"tool_call": {"tool": "get_success_progress", "arguments": {}}}),
            json.dumps({"tool_call": {"tool": "query_kg_nodes", "arguments": {"node_type": "Service"}}}),
            _final_execute(),
        ]
    )
    planner = Planner(client=client)

    outcome = planner.decide(
        goal="assess entry zone",
        graph_context={"operation_id": "op-loop", "cycle_index": 1},
        policy_context={"authorized": True},
        graph_tools=tools,
    )

    assert client.calls == 3  # two read turns + one final
    assert outcome.action == "execute"
    assert outcome.directive is not None
    assert outcome.directive.capability == "recon"
    assert outcome.metadata.get("read_steps") == 2  # two read tools were executed


def test_planner_advisory_write_tool_validation_failure_does_not_stop_decision() -> None:
    state = RuntimeState(operation_id="op-loop", execution=OperationRuntime(operation_id="op-loop"))
    tools = _tools(state)
    client = ScriptedClient(
        [
            json.dumps(
                {
                    "action": "execute",
                    "directive": {
                        "operation_id": "op-loop",
                        "cycle_index": 1,
                        "capability": "pivot",
                        "objective": "check whether any pivot route exists",
                        "target_refs": [],
                        "risk_level": "low",
                    },
                    "reason": "need pivot evidence",
                    "metadata": {
                        "planner_tool_calls": [
                            {
                                "tool": "record_attack_step",
                                "arguments": {
                                    "operation_id": "op-loop",
                                    "cycle_index": 1,
                                    "capability": "pivot",
                                    "summary": "missing status and includes outer identity fields",
                                },
                            }
                        ]
                    },
                }
            )
        ]
    )
    planner = Planner(client=client)

    outcome = planner.decide(
        goal="assess pivot",
        graph_context={"operation_id": "op-loop", "cycle_index": 1},
        graph_tools=tools,
    )

    assert outcome.action == "execute"
    assert outcome.directive is not None
    results = outcome.metadata["planner_graph_tool_results"]
    assert results[0]["tool"] == "record_attack_step"
    assert results[0]["error"] == "advisory_tool_validation_failed"
    assert "planner_attack_step_records" not in state.execution.metadata


def test_planner_exploit_round_prefers_metasploit_when_available() -> None:
    state = RuntimeState(operation_id="op-loop", execution=OperationRuntime(operation_id="op-loop"))
    tools = _tools(state)
    client = ScriptedClient(
        [
            json.dumps(
                {
                    "action": "execute",
                    "directive": {
                        "operation_id": "op-loop",
                        "cycle_index": 1,
                        "capability": "exploit",
                        "objective": "open a real session on the authorized Struts target",
                        "target_refs": [],
                        "allowed_tools": [],
                        "risk_level": "medium",
                    },
                    "reason": "need real exploit success",
                }
            )
        ]
    )
    planner = Planner(client=client)

    outcome = planner.decide(
        goal="assess exploit",
        graph_context={
            "operation_id": "op-loop",
            "cycle_index": 1,
            "mcp_tool_catalog": {
                "pentest-tools": {
                    "tools": [
                        {"name": "metasploit_exec"},
                        {"name": "session_exec"},
                    ]
                }
            },
        },
        graph_tools=tools,
    )

    assert outcome.directive is not None
    assert outcome.directive.allowed_tools[:2] == ["metasploit_exec", "session_exec"]
    assert outcome.directive.tool_hints[-1]["preferred_tool"] == "metasploit_exec"


def test_planner_loop_falls_back_to_replan_when_budget_exhausted() -> None:
    state = RuntimeState(operation_id="op-loop", execution=OperationRuntime(operation_id="op-loop"))
    tools = _tools(state)
    # Always asks for a read tool, never decides -> budget exhausted -> replan.
    client = ScriptedClient([json.dumps({"tool_call": {"tool": "get_success_progress", "arguments": {}}})])
    planner = Planner(client=client, config=None)
    planner._config.max_steps = 3  # type: ignore[attr-defined]

    outcome = planner.decide(
        goal="loop forever",
        graph_context={"operation_id": "op-loop", "cycle_index": 1},
        graph_tools=tools,
    )

    assert outcome.action == "replan"
    assert outcome.stop_condition == "planner_loop_exhausted"
    assert client.calls == 3  # bounded by max_steps


def test_planner_without_client_replans() -> None:
    planner = Planner(client=None)
    outcome = planner.decide(goal="x", graph_context={"operation_id": "op", "cycle_index": 0})
    assert outcome.action == "replan"
    assert outcome.stop_condition == "planner_llm_unavailable"
