from __future__ import annotations

import json

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.planning.llm_mission_planner_advisor import LLMMissionPlannerAdvisor
from src.core.planning.mission_planner_agent import MissionPlannerAgent
from src.core.planning.models import PlannerDecision


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


def test_llm_planner_decision_prompt_uses_decision_contract_not_task_graph() -> None:
    response = PackyLLMResponse(
        model="gpt-test",
        text=json.dumps(
            {
                "operation_id": "op-1",
                "cycle_index": 3,
                "decision": "dispatch_agent",
                "selected_agent": "recon_agent",
                "selected_stage": "RECON_STAGE",
                "objective": "Collect missing service evidence",
                "target_refs": [{"graph": "kg", "ref_id": "host-1", "ref_type": "Host"}],
                "required_context": {"reason": "evidence_gap"},
                "success_criteria": ["service evidence recorded"],
                "risk_level": "low",
                "max_steps": 3,
                "reasoning_summary": "Evidence is insufficient, so recon is next.",
                "handoff_acceptance": None,
                "stop_condition": None,
                "confidence": 0.8,
                "metadata": {},
            }
        ),
    )
    fake = FakePackyClient(response)
    advisor = LLMMissionPlannerAdvisor(client=fake)
    planner = MissionPlannerAgent(advisor=advisor)

    decision = planner.run(
        goal="validate objective",
        graph_context={
            "operation_id": "op-1",
            "cycle_index": 3,
            "kg_summary": {"hosts": ["host-1"]},
            "ag_summary": {"process_nodes": []},
            "runtime_summary": {"status": "running"},
        },
        policy_context={"authorized": True},
    )
    prompt_text = f"{fake.calls[0]['system_prompt']}\n{fake.calls[0]['user_prompt']}"

    assert isinstance(decision, PlannerDecision)
    assert decision.selected_agent == "recon_agent"
    assert "PlannerDecision" in prompt_text
    assert "TG" not in prompt_text
    assert "TaskGraph" not in prompt_text
    assert "selected_next_task" not in prompt_text
    assert "shell commands" in prompt_text
    assert "MCP tool arguments" in prompt_text
