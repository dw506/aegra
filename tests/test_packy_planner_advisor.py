from __future__ import annotations

import json

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.packy_llm import PackyLLMError, PackyLLMResponse
from src.core.agents.packy_planner_advisor import PackyPlannerAdvisor, PackyPlannerAdvisorConfig
from src.core.agents.planner import (
    PlannerLLMDecision,
    PlannerLLMRankAdjustment,
    PlanningCandidate,
    PlanningContext,
    PlannerAgent,
)
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import AttackGraph
from src.core.models.kg import DataAsset, Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.models.tg import TaskType


class FakePackyClient:
    def __init__(self, response: PackyLLMResponse | None = None, *, should_fail: bool = False) -> None:
        self._response = response
        self._should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def complete_chat(self, *, user_prompt: str, system_prompt: str | None = None, model: str | None = None, temperature: float | None = None) -> PackyLLMResponse:  # noqa: ANN401
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


class PromptAwareFakePackyClient:
    def complete_chat(self, *, user_prompt: str, system_prompt: str | None = None, model: str | None = None, temperature: float | None = None) -> PackyLLMResponse:  # noqa: ANN401
        del system_prompt, model, temperature
        candidate_marker = '"candidate_id": "'
        start = user_prompt.index(candidate_marker) + len(candidate_marker)
        end = user_prompt.index('"', start)
        candidate_id = user_prompt[start:end]
        return PackyLLMResponse(
            model="gpt-5.2",
            text=json.dumps(
                {
                    "advice": [
                        {
                            "candidate_id": candidate_id,
                            "score_delta": 0.1,
                            "rationale_suffix": "llm 判断该路径更贴近目标",
                            "metadata": {"reason": "goal_alignment"},
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )


def _candidate(candidate_id: str, *, score: float = 0.5) -> PlanningCandidate:
    return PlanningCandidate(
        candidate_id=candidate_id,
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        action_ids=["action-1"],
        score=score,
        rationale="baseline rationale",
        target_refs=[GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
        task_candidates=[
            TaskCandidate(
                source_action_id="action-1",
                task_type=TaskType.SERVICE_VALIDATION,
                input_bindings={"host_id": "127.0.0.1", "port": 8080},
                estimated_cost=0.1,
                estimated_risk=0.1,
                estimated_noise=0.1,
                goal_relevance=0.9,
                resource_keys={"host:127.0.0.1"},
            )
        ],
    )


def _build_goal_focused_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(Service(id="svc-1", label="SSH", confidence=0.8))
    kg.add_node(DataAsset(id="asset-1", label="Objective Data", confidence=0.85))
    kg.add_node(Goal(id="goal-1", label="Validate Objective", category="data", confidence=0.9))
    kg.add_edge(HostsEdge(id="edge-host-svc", label="hosts", source="host-1", target="svc-1"))
    kg.add_edge(TargetsEdge(id="edge-goal-asset", label="targets", source="goal-1", target="asset-1"))
    return kg


def _build_planner_input() -> AgentInput:
    ag = AttackGraphProjector().project(_build_goal_focused_kg())
    goal_node = ag.get_goal_nodes()[0]
    return AgentInput(
        graph_refs=[
            GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
            GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode"),
        ],
        context=AgentContext(operation_id="op-packy-advisor-test", runtime_state_ref="runtime-1"),
        raw_payload={
            "ag_graph": ag.to_dict(),
            "goal_refs": [GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode").model_dump(mode="json")],
            "planning_context": {"top_k": 1, "max_depth": 2},
        },
    )


def test_packy_planner_advisor_parses_fenced_json_and_clamps_score_delta() -> None:
    response = PackyLLMResponse(
        model="gpt-5.2",
        text=(
            "```json\n"
            '{"advice":['
            '{"candidate_id":"cand-1","score_delta":0.5,"rationale_suffix":"更贴近目标","metadata":{"reason":"goal_alignment"}},'
            '{"candidate_id":"unknown","score_delta":0.1,"rationale_suffix":"无效"}'
            "]}\n"
            "```"
        ),
    )
    advisor = PackyPlannerAdvisor(
        client=FakePackyClient(response),
        config=PackyPlannerAdvisorConfig(max_abs_score_delta=0.2),
    )

    advice = advisor.advise(
        graph=AttackGraph(),
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        candidates=[_candidate("cand-1"), _candidate("cand-2", score=0.2)],
        planning_context=PlanningContext(),
    )

    assert isinstance(advice, PlannerLLMDecision)
    assert len(advice.rank_adjustments) == 1
    assert advice.rank_adjustments[0].candidate_id == "cand-1"
    assert advice.rank_adjustments[0].score_delta == 0.2
    assert advice.rank_adjustments[0].metadata["reason"] == "goal_alignment"
    assert advice.rank_adjustments[0].rationale_suffix == "更贴近目标"
    assert advice.decision is not None
    assert advice.decision.target_id == "goal-1"
    assert advice.validation is not None
    assert advice.validation.accepted is True


def test_packy_planner_advisor_returns_empty_list_when_gateway_fails() -> None:
    advisor = PackyPlannerAdvisor(client=FakePackyClient(should_fail=True))

    advice = advisor.advise(
        graph=AttackGraph(),
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        candidates=[_candidate("cand-1")],
        planning_context=PlanningContext(),
    )

    assert advice == []


def test_packy_planner_advisor_rejects_forbidden_tool_or_graph_fields() -> None:
    response = PackyLLMResponse(
        model="gpt-5.2",
        text=json.dumps(
            {
                "advice": [
                    {
                        "candidate_id": "cand-1",
                        "score_delta": 0.1,
                        "rationale_suffix": "should be ignored",
                        "tool_command": "nmap -A target",
                    },
                    {
                        "candidate_id": "cand-2",
                        "score_delta": 0.05,
                        "rationale_suffix": "也许更稳",
                    },
                ]
            },
            ensure_ascii=False,
        ),
    )
    advisor = PackyPlannerAdvisor(client=FakePackyClient(response))

    advice = advisor.advise(
        graph=AttackGraph(),
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        candidates=[_candidate("cand-1"), _candidate("cand-2")],
        planning_context=PlanningContext(),
    )

    assert isinstance(advice, PlannerLLMDecision)
    assert len(advice.rank_adjustments) == 1
    assert advice.rank_adjustments[0].candidate_id == "cand-2"


def test_packy_planner_advisor_returns_empty_list_for_empty_or_unparseable_output() -> None:
    advisor = PackyPlannerAdvisor(
        client=FakePackyClient(PackyLLMResponse(model="gpt-5.2", text="not json")),
    )

    assert (
        advisor.advise(
            graph=AttackGraph(),
            goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
            candidates=[_candidate("cand-1")],
            planning_context=PlanningContext(),
        )
        == []
    )
    assert (
        advisor.advise(
            graph=AttackGraph(),
            goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
            candidates=[],
            planning_context=PlanningContext(),
        )
        == []
    )


def test_packy_planner_advisor_builds_prompt_from_candidate_context() -> None:
    fake_client = FakePackyClient(response=PackyLLMResponse(model="gpt-5.2", text='{"advice": []}'))
    advisor = PackyPlannerAdvisor(
        client=fake_client,
        config=PackyPlannerAdvisorConfig(model="gpt-5.2", max_candidates=1),
    )

    advisor.advise(
        graph=AttackGraph(),
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        candidates=[_candidate("cand-1"), _candidate("cand-2", score=0.1)],
        planning_context=PlanningContext(top_k=2, max_depth=4),
    )

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["model"] == "gpt-5.2"
    assert call["temperature"] == 0.0
    assert '"candidate_id": "cand-1"' in str(call["user_prompt"])
    assert '"max_depth": 4' in str(call["user_prompt"])
    assert '"candidate_id": "cand-2"' not in str(call["user_prompt"])


def test_planner_agent_can_consume_packy_planner_advisor_without_dispatch_side_effects() -> None:
    planner_input = _build_planner_input()
    advisor = PackyPlannerAdvisor(client=PromptAwareFakePackyClient())
    planner = PlannerAgent(llm_advisor=advisor)
    result = planner.run(planner_input)

    assert result.success is True
    assert result.output.decisions
    candidate = result.output.decisions[0]["payload"]["planning_candidate"]
    assert candidate["metadata"]["llm_planner_decision"]["rank_adjustments"][0]["metadata"]["reason"] == "goal_alignment"
    assert candidate["metadata"]["llm_decision"]["target_kind"] == "planner_goal"
    assert candidate["metadata"]["llm_decision_validation"]["accepted"] is True
    assert candidate["metadata"]["llm_decision_summary"]["adopted"] is True
    assert "llm 判断该路径更贴近目标" in result.output.decisions[0]["rationale"]
    assert result.output.state_deltas == []


class StrategyAdvisor:
    def __init__(self, decision: PlannerLLMDecision) -> None:
        self._decision = decision

    def advise(self, *, graph, goal_ref, candidates, planning_context):  # noqa: ANN001
        del graph, goal_ref, candidates, planning_context
        return self._decision


def test_planner_agent_uses_strategy_decision_to_change_candidate_ordering() -> None:
    cand_1 = _candidate("cand-1", score=0.8)
    cand_2 = _candidate("cand-2", score=0.7)
    planner = PlannerAgent(
        llm_advisor=StrategyAdvisor(
            PlannerLLMDecision(
                selected_candidate_ids=["cand-2", "cand-1"],
                rank_adjustments=[
                    PlannerLLMRankAdjustment(
                        candidate_id="cand-2",
                        score_delta=0.15,
                        rationale_suffix="llm 认为更贴近当前目标",
                    )
                ],
            )
        )
    )
    logs: list[str] = []

    updated = planner._apply_llm_advice(  # noqa: SLF001
        graph=AttackGraph(),
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        candidates=[cand_1, cand_2],
        planning_context=PlanningContext(top_k=2),
        logs=logs,
    )

    assert [candidate.candidate_id for candidate in updated] == ["cand-2", "cand-1"]
    assert updated[0].score > cand_2.score
    assert "llm 认为更贴近当前目标" in updated[0].rationale
    assert updated[0].metadata["llm_decision_summary"]["adopted"] is True


def test_planner_agent_records_human_review_request_in_output_metadata() -> None:
    cand_1 = _candidate("cand-1", score=0.8)
    planner = PlannerAgent(
        llm_advisor=StrategyAdvisor(
            PlannerLLMDecision(
                selected_candidate_ids=["cand-1"],
                requires_human_review=True,
                defer_reason="目标风险较高，建议人工复核",
            )
        )
    )
    logs: list[str] = []

    updated = planner._apply_llm_advice(  # noqa: SLF001
        graph=AttackGraph(),
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        candidates=[cand_1],
        planning_context=PlanningContext(),
        logs=logs,
    )

    summary = updated[0].metadata["llm_decision_summary"]
    assert summary["requires_human_review"] is True
    assert summary["defer_reason"] == "目标风险较高，建议人工复核"


def test_planner_agent_rejects_strategy_with_unknown_candidate_and_falls_back() -> None:
    cand_1 = _candidate("cand-1", score=0.8)
    planner = PlannerAgent(
        llm_advisor=StrategyAdvisor(
            PlannerLLMDecision(
                selected_candidate_ids=["unknown-candidate"],
                rank_adjustments=[PlannerLLMRankAdjustment(candidate_id="unknown-candidate", score_delta=0.1)],
            )
        )
    )
    logs: list[str] = []

    updated = planner._apply_llm_advice(  # noqa: SLF001
        graph=AttackGraph(),
        goal_ref=GraphRef(graph=GraphScope.KG, ref_id="goal-1", ref_type="Goal"),
        candidates=[cand_1],
        planning_context=PlanningContext(),
        logs=logs,
    )

    assert updated == [cand_1]
    assert "llm_decision_summary" not in updated[0].metadata
    assert any("planner llm strategy decision rejected" in log for log in logs)
