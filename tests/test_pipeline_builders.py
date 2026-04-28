from __future__ import annotations

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.critic import CriticAgent, CriticLLMReview
from src.core.agents.packy_critic_advisor import PackyCriticAdvisor
from src.core.agents.packy_llm import PackyLLMConfig
from src.core.agents.packy_planner_advisor import PackyPlannerAdvisor
from src.core.agents.pipeline_builders import AgentPipelineAssemblyOptions, build_optional_agent_pipeline
from src.core.agents.planner import PlannerAgent, PlannerLLMAdvice
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.kg import DataAsset, Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus


class StaticPlannerAdvisor:
    def advise(self, *, graph, goal_ref, candidates, planning_context):  # noqa: ANN001
        del graph, goal_ref, planning_context
        if not candidates:
            return []
        return [
            PlannerLLMAdvice(
                candidate_id=candidates[0].candidate_id,
                score_delta=0.1,
                rationale_suffix="llm 认为该候选最贴近目标",
                metadata={"reason": "goal_alignment"},
            )
        ]


class StaticCriticAdvisor:
    def summarize_findings(self, *, findings, context, runtime_state):  # noqa: ANN001
        del context, runtime_state
        if not findings:
            return []
        return [
            CriticLLMReview(
                finding_id=findings[0].finding_id,
                summary_override=f"{findings[0].summary} (llm归纳)",
                rationale_suffix="llm 归纳为依赖链失效",
                metadata={"category": "dependency_failure"},
            )
        ]


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
        context=AgentContext(operation_id="op-pipeline-builder-test", runtime_state_ref="runtime-1"),
        raw_payload={
            "ag_graph": ag.to_dict(),
            "goal_refs": [GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode").model_dump(mode="json")],
            "planning_context": {"top_k": 1, "max_depth": 2},
        },
    )


def test_optional_pipeline_builder_keeps_default_planner_behavior_without_packy() -> None:
    pipeline = build_optional_agent_pipeline()

    planner = pipeline.registry.get("planner_agent")

    assert isinstance(planner, PlannerAgent)
    assert planner._llm_advisor is None  # noqa: SLF001


def test_optional_pipeline_builder_can_inject_explicit_planner_advisor() -> None:
    pipeline = build_optional_agent_pipeline(planner_llm_advisor=StaticPlannerAdvisor())
    planner_input = _build_planner_input()

    result = pipeline.run_planning_cycle(
        operation_id="op-pipeline-builder-test",
        graph_refs=planner_input.graph_refs,
        planner_payload=planner_input.raw_payload,
        context=planner_input.context,
    )

    assert result.success is True
    assert result.final_output.decisions
    candidate = result.final_output.decisions[0]["payload"]["planning_candidate"]
    assert candidate["metadata"]["llm_advice"]["metadata"]["reason"] == "goal_alignment"
    assert "llm 认为该候选最贴近目标" in result.final_output.decisions[0]["rationale"]


def test_optional_pipeline_builder_can_enable_packy_via_env_when_factory_is_monkeypatched(
    monkeypatch,
) -> None:
    sentinel_advisor = object()
    monkeypatch.setattr(PackyPlannerAdvisor, "from_env", classmethod(lambda cls: sentinel_advisor))

    pipeline = build_optional_agent_pipeline(
        options=AgentPipelineAssemblyOptions(enable_packy_planner_advisor=True)
    )

    planner = pipeline.registry.get("planner_agent")

    assert isinstance(planner, PlannerAgent)
    assert planner._llm_advisor is sentinel_advisor  # noqa: SLF001


def test_optional_pipeline_builder_can_use_explicit_llm_client_config_without_env(monkeypatch) -> None:
    def fail_from_env(cls):  # noqa: ANN001
        raise AssertionError("from_env should not be called when llm_client_config is provided")

    monkeypatch.setattr(PackyPlannerAdvisor, "from_env", classmethod(fail_from_env))

    pipeline = build_optional_agent_pipeline(
        options=AgentPipelineAssemblyOptions(enable_packy_planner_advisor=True),
        llm_client_config=PackyLLMConfig(
            api_key="planner-key",
            base_url="https://planner.example/v1",
            model="gpt-5.4",
            timeout_sec=45.0,
        ),
    )

    planner = pipeline.registry.get("planner_agent")

    assert isinstance(planner, PlannerAgent)
    assert planner._llm_advisor is not None  # noqa: SLF001
    assert planner._llm_advisor._client.config.model == "gpt-5.4"  # noqa: SLF001


def test_optional_pipeline_builder_can_inject_explicit_critic_advisor() -> None:
    pipeline = build_optional_agent_pipeline(critic_llm_advisor=StaticCriticAdvisor())

    critic = pipeline.registry.get("critic_agent")

    assert isinstance(critic, CriticAgent)
    assert critic._llm_advisor is not None  # noqa: SLF001


def test_optional_pipeline_builder_can_enable_packy_critic_via_env_when_factory_is_monkeypatched(
    monkeypatch,
) -> None:
    sentinel_advisor = object()
    monkeypatch.setattr(PackyCriticAdvisor, "from_env", classmethod(lambda cls: sentinel_advisor))

    pipeline = build_optional_agent_pipeline(
        options=AgentPipelineAssemblyOptions(enable_packy_critic_advisor=True)
    )

    critic = pipeline.registry.get("critic_agent")

    assert isinstance(critic, CriticAgent)
    assert critic._llm_advisor is sentinel_advisor  # noqa: SLF001
