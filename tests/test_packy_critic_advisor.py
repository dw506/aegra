from __future__ import annotations

import json

import httpx

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.critic import CriticAgent, CriticContext, CriticFinding
from src.core.agents.packy_critic_advisor import PackyCriticAdvisor
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig
from src.core.models.ag import GraphRef as AGGraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.models.tg import TaskGraph, TaskNode, TaskStatus, TaskType


def build_finding() -> CriticFinding:
    return CriticFinding(
        finding_id="finding-1",
        finding_type="permanently_blocked_tasks",
        severity="high",
        subject_refs=[GraphRef(graph=GraphScope.TG, ref_id="task-1", ref_type="Task")],
        summary="task task-1 appears permanently blocked",
        rationale="upstream dependency failed",
        metadata={"gate_ids": ["gate-1"]},
    )


def build_context() -> CriticContext:
    return CriticContext(
        failure_threshold=2,
        low_value_threshold=0.25,
        runtime_summary={"operation_status": "running"},
    )


def build_runtime_state() -> RuntimeState:
    return RuntimeState(operation_id="op-critic", execution=OperationRuntime(operation_id="op-critic"))


def test_packy_critic_advisor_parses_json_reviews_from_gateway_response() -> None:
    response_text = json_chat_response(
        """
        {
          "reviews": [
            {
              "finding_id": "finding-1",
              "summary_override": "task task-1 appears permanently blocked (llm归纳)",
              "rationale_suffix": "llm 归纳为上游依赖失效导致的持续阻塞",
              "replan_hint": "先复核上游依赖状态再触发现有 replan 流",
              "failure_summary": "上游依赖失效",
              "affected_task_ids": ["task-1"],
              "confidence": 0.82,
              "requires_human_review": true,
              "metadata": {"category": "dependency_failure"}
            },
            {
              "finding_id": "finding-unknown",
              "summary_override": "should be ignored"
            }
          ]
        }
        """.strip()
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(status_code=200, text=response_text, headers={"Content-Type": "application/json"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url="https://critic.example/v1", transport=transport) as http_client:
        advisor = PackyCriticAdvisor(
            client=PackyLLMClient(
                PackyLLMConfig(
                    api_key="critic-key",
                    base_url="https://critic.example/v1",
                    model="gpt-5.4",
                    timeout_sec=30.0,
                ),
                http_client=http_client,
            )
        )
        reviews = advisor.summarize_findings(
            findings=[build_finding()],
            context=build_context(),
            runtime_state=build_runtime_state(),
        )

    assert len(reviews) == 1
    assert reviews[0].finding_id == "finding-1"
    assert reviews[0].summary_override == "task task-1 appears permanently blocked (llm归纳)"
    assert "llm 归纳" in (reviews[0].rationale_suffix or "")
    assert reviews[0].replan_hint == "先复核上游依赖状态再触发现有 replan 流"
    assert reviews[0].metadata["category"] == "dependency_failure"
    assert reviews[0].replan_proposal is not None
    assert reviews[0].replan_proposal.affected_task_ids == ["task-1"]
    assert reviews[0].replan_proposal.confidence == 0.82
    assert reviews[0].replan_proposal.requires_human_review is True
    assert reviews[0].decision is not None
    assert reviews[0].decision.target_id == "finding-1"
    assert reviews[0].validation is not None
    assert reviews[0].validation.accepted is True


def test_packy_critic_advisor_falls_back_to_empty_reviews_on_gateway_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(status_code=500, json={"error": {"message": "gateway down"}})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url="https://critic.example/v1", transport=transport) as http_client:
        advisor = PackyCriticAdvisor(
            client=PackyLLMClient(
                PackyLLMConfig(
                    api_key="critic-key",
                    base_url="https://critic.example/v1",
                    model="gpt-5.4",
                    timeout_sec=30.0,
                ),
                http_client=http_client,
            )
        )
        reviews = advisor.summarize_findings(
            findings=[build_finding()],
            context=build_context(),
            runtime_state=build_runtime_state(),
        )

    assert reviews == []


def test_packy_critic_advisor_rejects_forbidden_direct_actions() -> None:
    response_text = json_chat_response(
        json.dumps(
            {
                "reviews": [
                    {
                        "finding_id": "finding-1",
                        "summary_override": "should be ignored",
                        "cancel_task": "task-1",
                    },
                    {
                        "finding_id": "finding-1",
                        "rationale_suffix": "只补充归因",
                        "replan_hint": "使用现有 replan 流程复核依赖",
                    },
                ]
            },
            ensure_ascii=False,
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(status_code=200, text=response_text, headers={"Content-Type": "application/json"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url="https://critic.example/v1", transport=transport) as http_client:
        advisor = PackyCriticAdvisor(
            client=PackyLLMClient(
                PackyLLMConfig(
                    api_key="critic-key",
                    base_url="https://critic.example/v1",
                    model="gpt-5.4",
                    timeout_sec=30.0,
                ),
                http_client=http_client,
            )
        )
        reviews = advisor.summarize_findings(
            findings=[build_finding()],
            context=build_context(),
            runtime_state=build_runtime_state(),
        )

    assert len(reviews) == 1
    assert reviews[0].rationale_suffix == "只补充归因"
    assert reviews[0].replan_hint == "使用现有 replan 流程复核依赖"


def test_packy_critic_advisor_returns_empty_list_for_unknown_empty_or_unparseable_output() -> None:
    for content in [
        "not json",
        json.dumps({"reviews": [{"finding_id": "unknown", "summary_override": "ignored"}]}),
        json.dumps({"reviews": []}),
    ]:
        response_text = json_chat_response(content)

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(status_code=200, text=response_text, headers={"Content-Type": "application/json"})

        transport = httpx.MockTransport(handler)
        with httpx.Client(base_url="https://critic.example/v1", transport=transport) as http_client:
            advisor = PackyCriticAdvisor(
                client=PackyLLMClient(
                    PackyLLMConfig(
                        api_key="critic-key",
                        base_url="https://critic.example/v1",
                        model="gpt-5.4",
                        timeout_sec=30.0,
                    ),
                    http_client=http_client,
                )
            )
            reviews = advisor.summarize_findings(
                findings=[build_finding()],
                context=build_context(),
                runtime_state=build_runtime_state(),
            )

        assert reviews == []


def test_critic_agent_can_integrate_packy_critic_advisor_reviews() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        prompt = payload["messages"][-1]["content"]
        prompt_payload = json.loads(prompt.split("\n\n", 1)[1])
        finding_id = prompt_payload["findings"][0]["finding_id"]
        response_text = json_chat_response(
            json.dumps(
                {
                    "reviews": [
                        {
                            "finding_id": finding_id,
                            "summary_override": "task task-1 appears permanently blocked (llm归纳)",
                            "rationale_suffix": "llm 归纳为上游依赖失效导致的持续阻塞",
                            "replan_hint": "复核上游依赖后触发现有局部 replan",
                            "failure_summary": "上游依赖失效",
                            "affected_task_ids": ["task-1"],
                            "confidence": 0.77,
                            "metadata": {"category": "dependency_failure"},
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        return httpx.Response(status_code=200, text=response_text, headers={"Content-Type": "application/json"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(base_url="https://critic.example/v1", transport=transport) as http_client:
        advisor = PackyCriticAdvisor(
            client=PackyLLMClient(
                PackyLLMConfig(
                    api_key="critic-key",
                    base_url="https://critic.example/v1",
                    model="gpt-5.4",
                    timeout_sec=30.0,
                ),
                http_client=http_client,
            )
        )
        graph = TaskGraph()
        graph.add_node(
            TaskNode(
                id="task-1",
                label="Blocked task",
                task_type=TaskType.SERVICE_VALIDATION,
                status=TaskStatus.BLOCKED,
                source_action_id="action-1",
                target_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
                goal_relevance=0.1,
                reason="upstream dependency failed",
            )
        )
        agent = CriticAgent(llm_advisor=advisor)
        result = agent.run(
            AgentInput(
                graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
                context=AgentContext(operation_id="op-critic", runtime_state_ref="runtime-1"),
                raw_payload={"tg_graph": graph.to_dict()},
            )
        )

    assert result.success is True
    assert result.output.decisions
    assert any("llm 归纳" in decision["payload"]["recommendation"]["rationale"] for decision in result.output.decisions)
    assert any("LLM replan hint" in decision["payload"]["recommendation"]["rationale"] for decision in result.output.decisions)
    assert any(
        decision["payload"]["recommendation"]["metadata"]["llm_replan_proposal"]["affected_task_ids"] == ["task-1"]
        for decision in result.output.decisions
        if decision["payload"]["recommendation"]["metadata"].get("llm_replan_proposal")
    )
    assert result.output.replan_requests
    assert any(
        request["payload"]["runtime_metadata"]["llm_replan_proposal"]["adopted"] is True
        for request in result.output.replan_requests
    )
    assert any(
        decision["payload"]["recommendation"]["metadata"]["finding_id"]
        for decision in result.output.decisions
    )


def json_chat_response(content: str) -> str:
    return (
        "{"
        '"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":'
        + json.dumps(content, ensure_ascii=False)
        + "}}]}"
    )
