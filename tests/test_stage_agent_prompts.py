from __future__ import annotations

import json
from pathlib import Path

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.stage.llm_stage_advisor import LLMStageAdvisor
from src.core.stage.models import StageExecutionRequest, StageResult


class FakePackyClient:
    def __init__(self) -> None:
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
        return PackyLLMResponse(
            model="gpt-test",
            text=json.dumps({"action": "finish", "rationale": "done", "finish": {"status": "succeeded", "summary": "done"}}),
        )


def test_all_stage_agent_prompt_files_exist() -> None:
    prompt_dir = Path("src/core/stage/prompts")

    for filename in [
        "recon_agent.md",
        "vuln_analysis_agent.md",
        "exploit_validation_agent.md",
        "access_pivot_agent.md",
        "goal_agent.md",
    ]:
        path = prompt_dir / filename
        assert path.exists()
        assert "Return only StageAgentDecision JSON" in path.read_text(encoding="utf-8")


def test_llm_stage_advisor_uses_common_and_agent_specific_prompt() -> None:
    fake = FakePackyClient()
    advisor = LLMStageAdvisor(client=fake)
    request = StageExecutionRequest(
        operation_id="op-1",
        cycle_index=1,
        agent_name="goal_agent",
        stage_type="GOAL_STAGE",
        objective="Verify goal",
        risk_level="low",
        max_steps=1,
    )

    advisor.decide(
        agent_name="goal_agent",
        stage_type="GOAL_STAGE",
        request=request,
        graph_context={},
        runtime_context={},
        policy_context={},
        memory=[],
        available_tools={},
    )

    system_prompt = str(fake.calls[0]["system_prompt"])
    assert "Call only tools present in mcp_tool_catalog" in system_prompt
    assert "You are GoalAgent" in system_prompt
    assert "runtime_hints.goal_satisfied=true" in system_prompt
    assert "You are an Aegra LLM Stage Agent" in system_prompt
    assert "parallel execution capability pool" in system_prompt
    assert "Do not call or route work to another StageAgent" in system_prompt


def test_llm_stage_advisor_normalizes_string_finish_items_and_legacy_intents() -> None:
    class LoosePackyClient(FakePackyClient):
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
            return PackyLLMResponse(
                model="gpt-test",
                text=json.dumps(
                    {
                        "action": "finish",
                        "rationale": "done",
                        "finish": {
                            "status": "succeeded",
                            "summary": "observed DMZ service",
                            "observations": ["Target remains within authorized DMZ scope."],
                            "findings": ["Confirmed reachable DMZ HTTP service."],
                            "next_stage_candidates": ["VULN_ANALYSIS_STAGE"],
                            "graph_update_intents": [
                                {"intent": "upsert_service", "match": "10.20.0.22:8080/tcp", "attributes": {"state": "open"}}
                            ],
                        },
                    }
                ),
            )

    advisor = LLMStageAdvisor(client=LoosePackyClient())
    request = StageExecutionRequest(
        operation_id="op-1",
        cycle_index=1,
        agent_name="recon_agent",
        stage_type="RECON_STAGE",
        objective="Discover DMZ service",
        risk_level="low",
        max_steps=1,
    )

    decision = advisor.decide(
        agent_name="recon_agent",
        stage_type="RECON_STAGE",
        request=request,
        graph_context={},
        runtime_context={},
        policy_context={},
        memory=[],
        available_tools={},
    )

    result = StageResult(
        operation_id=request.operation_id,
        stage_task_id="stage-recon-1",
        stage_type=request.stage_type,
        agent_name=request.agent_name,
        status=decision.finish["status"],
        summary=decision.finish["summary"],
        observations=decision.finish["observations"],
        findings=decision.finish["findings"],
        next_stage_candidates=decision.finish["next_stage_candidates"],
        graph_update_intents=decision.finish["graph_update_intents"],
    )

    assert result.observations[0]["summary"] == "Target remains within authorized DMZ scope."
    assert result.findings[0]["summary"] == "Confirmed reachable DMZ HTTP service."
    assert result.next_stage_candidates[0]["stage_type"] == "VULN_ANALYSIS_STAGE"
    assert result.graph_update_intents[0].target_graph == "KG"
    assert result.graph_update_intents[0].entity_type == "Service"
