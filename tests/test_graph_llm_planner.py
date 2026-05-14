from __future__ import annotations

import json

from src.core.agents.graph_context import (
    GraphContext,
    GraphContextAction,
    GraphContextGoal,
    GraphContextPolicy,
    GraphContextRef,
)
from src.core.agents.graph_llm_planner import GraphLLMPlannerAdvisor, GraphLLMPlannerAdvisorConfig
from src.core.agents.packy_llm import PackyLLMError, PackyLLMResponse
from src.core.models.ag import GraphRef
from src.core.models.tg import TaskType


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


def _context() -> GraphContext:
    host_ref = GraphContextRef(graph="kg", ref_id="127.0.0.1", ref_type="Host", label="127.0.0.1")
    goal_ref = GraphContextRef(graph="ag", ref_id="goal-1", ref_type="GoalNode")
    action_ref = GraphContextRef(graph="ag", ref_id="action-validate-service", ref_type="ActionNode")
    return GraphContext(
        operation_id="op-graph-llm",
        goals=[
            GraphContextGoal(
                ref=goal_ref,
                goal_type="OBJECTIVE_SATISFIED",
                priority=90,
                business_value=0.9,
                scope_refs=[host_ref],
            )
        ],
        frontier_actions=[
            GraphContextAction(
                ref=action_ref,
                action_type="VALIDATE_SERVICE",
                activation_status="activatable",
                cost=0.1,
                risk=0.1,
                noise=0.1,
                expected_value=0.8,
                success_probability_prior=0.7,
                goal_relevance=0.9,
                approval_required=False,
                target_refs=[host_ref],
                required_capabilities=["http_probe"],
                resource_keys=["host:127.0.0.1"],
            )
        ],
        policy=GraphContextPolicy(authorized_hosts=["127.0.0.1"]),
        context_stats={"estimated_context_chars": 1000},
    )


def _accepted_response() -> PackyLLMResponse:
    return PackyLLMResponse(
        model="gpt-test",
        text=json.dumps(
            {
                "proposal_id": "proposal-1",
                "task_proposals": [
                    {
                        "proposal_id": "task-proposal-1",
                        "task_type": TaskType.SERVICE_VALIDATION.value,
                        "target_refs": [
                            {
                                "graph": "kg",
                                "ref_id": "127.0.0.1",
                                "ref_type": "Host",
                                "label": "127.0.0.1",
                            }
                        ],
                        "rationale": "confirm service state from graph frontier",
                        "expected_evidence": ["validated service metadata"],
                        "estimated_risk": 0.1,
                        "estimated_noise": 0.1,
                    }
                ],
                "rank_adjustments": [
                    {
                        "target_ref": {"graph": "ag", "ref_id": "action-validate-service", "ref_type": "ActionNode"},
                        "score_delta": 0.1,
                        "rationale": "high evidence gain",
                    }
                ],
                "metadata": {"reason": "evidence_gain"},
            }
        ),
    )


def test_graph_llm_planner_advisor_accepts_valid_proposal() -> None:
    fake = FakePackyClient(_accepted_response())
    advisor = GraphLLMPlannerAdvisor(
        client=fake,
        config=GraphLLMPlannerAdvisorConfig(model="gpt-test"),
    )

    advice = advisor.advise(
        graph_context=_context(),
        goal_refs=[GraphRef(graph="ag", ref_id="goal-1", ref_type="GoalNode")],
        recent_signals=[{"kind": "repetition", "decision": "allow"}],
    )

    assert advice.validation.accepted is True
    assert advice.proposal.proposal_id == "proposal-1"
    assert advice.proposal.task_proposals[0].task_type == TaskType.SERVICE_VALIDATION.value
    assert advice.proposal.rank_adjustments[0].target_ref.ref_id == "action-validate-service"
    assert advice.llm_metadata["model"] == "gpt-test"
    assert fake.calls[0]["temperature"] == 0.0
    assert "GraphLLMPlanProposal" in str(fake.calls[0]["user_prompt"])
    assert "Do not output shell commands" in str(fake.calls[0]["system_prompt"])


def test_graph_llm_planner_advisor_returns_empty_for_bad_json() -> None:
    advisor = GraphLLMPlannerAdvisor(
        client=FakePackyClient(PackyLLMResponse(model="gpt-test", text="not-json")),
    )

    advice = advisor.advise(graph_context=_context())

    assert advice.validation.accepted is False
    assert advice.validation.reason == "invalid graph llm planner json"
    assert advice.proposal.task_proposals == []


def test_graph_llm_planner_advisor_returns_empty_when_llm_fails() -> None:
    advisor = GraphLLMPlannerAdvisor(client=FakePackyClient(should_fail=True))

    advice = advisor.advise(graph_context=_context())

    assert advice.validation.accepted is False
    assert advice.validation.reason == "llm call failed"
    assert advice.proposal.task_proposals == []


def test_graph_llm_planner_advisor_rejects_forbidden_command() -> None:
    response = PackyLLMResponse(
        model="gpt-test",
        text=json.dumps(
            {
                "task_proposals": [
                    {
                        "task_type": TaskType.SERVICE_VALIDATION.value,
                        "target_refs": [{"graph": "kg", "ref_id": "127.0.0.1", "ref_type": "Host"}],
                        "command": "nmap -sV 127.0.0.1",
                    }
                ]
            }
        ),
    )
    advisor = GraphLLMPlannerAdvisor(client=FakePackyClient(response))

    advice = advisor.advise(graph_context=_context())

    assert advice.validation.accepted is False
    assert "command" in advice.validation.reason
    assert advice.proposal.task_proposals == []


def test_graph_llm_planner_advisor_rejects_unknown_ref() -> None:
    response = PackyLLMResponse(
        model="gpt-test",
        text=json.dumps(
            {
                "task_proposals": [
                    {
                        "task_type": TaskType.SERVICE_VALIDATION.value,
                        "target_refs": [{"graph": "kg", "ref_id": "unknown-host", "ref_type": "Host"}],
                    }
                ]
            }
        ),
    )
    advisor = GraphLLMPlannerAdvisor(client=FakePackyClient(response))

    advice = advisor.advise(graph_context=_context())

    assert advice.validation.accepted is False
    assert "unknown ref" in advice.validation.reason
    assert advice.proposal.task_proposals == []
