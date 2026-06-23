from __future__ import annotations

import json

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.planning.llm_mission_planner_advisor import LLMMissionPlannerAdvisor
from src.core.planning.mission_planner_agent import MissionPlannerAgent
from src.core.planning.models import PlannerOutcome


class FakePackyClient:
    def __init__(self, response: PackyLLMResponse) -> None:
        self._response = response
        self.config = type("Config", (), {"model": "gpt-test"})()
        self.calls: list[dict[str, object]] = []

    def complete_chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> PackyLLMResponse:
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "model": model,
                "temperature": temperature,
            }
        )
        return self._response


def test_llm_planner_outcome_prompt_uses_directive_contract_not_task_graph() -> None:
    response = PackyLLMResponse(
        model="gpt-test",
        text=json.dumps(
            {
                "operation_id": "op-1",
                "cycle_index": 3,
                "action": "execute",
                "directive": {
                    "operation_id": "op-1",
                    "cycle_index": 3,
                    "capability": "recon",
                    "objective": "Collect missing service evidence",
                    "target_refs": [{"graph": "kg", "ref_id": "host-1", "ref_type": "Host"}],
                    "allowed_tools": ["nmap_scan"],
                    "tool_hints": [],
                    "max_tools": 3,
                    "success_hint": "service evidence recorded",
                    "required_context": {"reason": "evidence_gap"},
                    "risk_level": "low",
                },
                "reason": "Evidence is insufficient, so recon is next.",
                "stop_condition": None,
                "confidence": 0.8,
                "metadata": {},
            }
        ),
    )
    fake = FakePackyClient(response)
    advisor = LLMMissionPlannerAdvisor(client=fake)
    planner = MissionPlannerAgent(advisor=advisor)

    outcome = planner.decide(
        goal="validate objective",
        graph_context={
            "operation_id": "op-1",
            "cycle_index": 3,
            "min_summary": {"kg_node_count": 1, "kg_edge_count": 0, "recent_attack_steps": []},
            "success_condition_progress": {"eligible_for_stop": False, "missing": ["dmz_service_discovered"]},
            "graph_tools": {"write": ["record_finding"]},
            "mcp_tool_catalog": {"pentest-tools": {"tools": [{"name": "nmap_scan"}]}},
            "agent_capabilities": [{"agent_name": "recon_agent"}],
        },
        policy_context={"authorized": True},
    )
    prompt_text = f"{fake.calls[0]['system_prompt']}\n{fake.calls[0]['user_prompt']}"

    assert isinstance(outcome, PlannerOutcome)
    assert outcome.directive is not None
    assert outcome.directive.capability == "recon"
    assert "PlannerOutcome" in prompt_text
    assert "RoundDirective" in prompt_text
    assert "TG" not in prompt_text
    assert "TaskGraph" not in prompt_text
    assert "selected_next_task" not in prompt_text
    assert "shell commands" in prompt_text
    assert "MCP tool arguments" in prompt_text
    assert "ExecutionAgent" in prompt_text
    assert "min_summary" in prompt_text
    # Push model (option B): read tools are no longer advertised or callable, so
    # the prompt must NOT tell the LLM to pull graph detail via kg_query.
    assert "kg_query" not in prompt_text
    assert "kg_summary" not in prompt_text
    assert "ag_summary" not in prompt_text
    assert "known_assets" not in prompt_text
    assert "pivot_candidates" not in prompt_text
