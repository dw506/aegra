from __future__ import annotations

import json

from src.core.agents.packy_llm import PackyLLMError, PackyLLMResponse
from src.core.models.ag import GraphRef
from src.core.planning.llm_mission_planner_advisor import LLMMissionPlannerAdvisor
from src.core.planning.mission_planner_agent import MissionPlannerAgent
from src.core.stage.llm_stage_advisor import LLMStageAdvisor
from src.core.stage.models import StageTask, StageType


class FakePackyClient:
    def __init__(self, response: PackyLLMResponse | None = None, *, should_fail: bool = False) -> None:
        self._response = response
        self._should_fail = should_fail
        self.config = type("Config", (), {"base_url": "https://llm.example/v1", "model": "gpt-test"})()
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
        if self._should_fail:
            raise PackyLLMError("gateway unavailable")
        assert self._response is not None
        return self._response


def test_llm_mission_planner_advisor_accepts_stage_tasks() -> None:
    response = PackyLLMResponse(
        model="gpt-test",
        text=json.dumps(
            {
                "operation_id": "op-1",
                "stage_tasks": [
                    {
                        "task_id": "stage-recon-1",
                        "stage_type": "RECON_STAGE",
                        "objective": "Discover reachable services",
                        "target_refs": [{"graph": "kg", "ref_id": "host-1", "ref_type": "Host"}],
                    }
                ],
                "dependencies": [],
                "summary": "planned recon",
            }
        ),
    )
    fake = FakePackyClient(response)
    advisor = LLMMissionPlannerAdvisor(client=fake)

    result = advisor.propose_stage_tasks(
        goal="prove access",
        graph_context={"operation_id": "op-1", "target_refs": []},
        policy_context={},
    )

    assert result.stage_tasks[0].stage_type == StageType.RECON_STAGE
    assert result.metadata["planner"] == "llm_mission_planner"
    assert result.metadata["accepted"] is True
    assert "Do not output shell commands" in str(fake.calls[0]["system_prompt"])


def test_mission_planner_falls_back_when_llm_returns_no_tasks() -> None:
    advisor = LLMMissionPlannerAdvisor(client=FakePackyClient(PackyLLMResponse(model="gpt-test", text="not-json")))
    planner = MissionPlannerAgent(advisor=advisor)

    result = planner.run(
        goal="prove access",
        graph_context={
            "operation_id": "op-1",
            "target_refs": [{"graph": "kg", "ref_id": "host-1", "ref_type": "Host"}],
        },
    )

    assert [task.stage_type for task in result.stage_tasks] == [
        StageType.RECON_STAGE,
        StageType.VULN_ANALYSIS_STAGE,
        StageType.EXPLOIT_STAGE,
        StageType.ACCESS_PIVOT_STAGE,
        StageType.GOAL_STAGE,
    ]
    assert result.metadata["advisor_fallback"]["metadata"]["accepted"] is False


def test_llm_stage_advisor_accepts_tool_call_decision() -> None:
    response = PackyLLMResponse(
        model="gpt-test",
        text=json.dumps(
            {
                "action": "call_tool",
                "rationale": "probe authorized host",
                "tool_call": {
                    "server_id": "pentest-tools",
                    "tool_name": "safe_http_probe",
                    "arguments": {"url": "http://10.0.0.10"},
                    "timeout_seconds": 30,
                },
            }
        ),
    )
    fake = FakePackyClient(response)
    advisor = LLMStageAdvisor(client=fake)
    task = StageTask(
        task_id="stage-recon-1",
        stage_type=StageType.RECON_STAGE,
        objective="Discover service",
        target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
    )

    decision = advisor.decide(
        agent_name="recon_agent",
        stage_type=StageType.RECON_STAGE,
        task=task,
        graph_context={"operation_id": "op-1"},
        runtime_context={"operation_id": "op-1"},
        memory=[],
        available_tools={"pentest-tools": {"tools": [{"name": "safe_http_probe"}]}},
    )

    assert decision.action == "call_tool"
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "safe_http_probe"
    assert "Do not invent shell commands" in str(fake.calls[0]["system_prompt"])


def test_llm_stage_advisor_requests_replan_when_llm_fails() -> None:
    advisor = LLMStageAdvisor(client=FakePackyClient(should_fail=True))
    task = StageTask(task_id="stage-goal-1", stage_type=StageType.GOAL_STAGE, objective="Verify goal")

    decision = advisor.decide(
        agent_name="goal_agent",
        stage_type=StageType.GOAL_STAGE,
        task=task,
        graph_context={},
        runtime_context={},
        memory=[],
        available_tools={},
    )

    assert decision.action == "need_replan"
    assert "unavailable" in decision.rationale
