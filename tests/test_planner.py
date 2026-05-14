from __future__ import annotations

from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import ActionNodeType, ConstraintNodeType, GraphRef as AGGraphRef
from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.graph_llm_models import GraphLLMPlanProposal, GraphLLMTaskProposal
from src.core.agents.graph_llm_models import GraphLLMPlanValidationResult
from src.core.agents.graph_llm_planner import GraphLLMPlannerAdvice
from src.core.agents.planner import PlannerAgent
from src.core.models.kg import DataAsset, Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.graph.tg_builder import TaskGenerationRequest
from src.core.planner.critic import AttackGraphCritic, CriticContext
from src.core.planner.planner import AttackGraphPlanner
from src.core.models.tg import TaskType


def build_goal_focused_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(Service(id="svc-1", label="SSH", confidence=0.75))
    kg.add_node(DataAsset(id="asset-1", label="Objective Data", confidence=0.8))
    kg.add_node(Goal(id="goal-1", label="Validate Objective", category="data", confidence=0.9))
    kg.add_edge(HostsEdge(id="e-host-svc", label="hosts", source="host-1", target="svc-1"))
    kg.add_edge(TargetsEdge(id="e-goal-asset", label="targets", source="goal-1", target="asset-1"))
    return kg


def test_planner_returns_top_k_candidates() -> None:
    projector = AttackGraphProjector()
    ag = projector.project(build_goal_focused_kg())
    planner = AttackGraphPlanner()
    goal_id = ag.get_goal_nodes()[0].id

    result = planner.plan(ag, goal_id=goal_id, top_k=2, max_depth=2)

    assert result.candidates
    assert result.chains
    assert result.task_candidates
    assert len(result.candidates) <= 2
    assert all(candidate.reason for candidate in result.candidates)


def test_planner_can_export_task_generation_request() -> None:
    projector = AttackGraphProjector()
    ag = projector.project(build_goal_focused_kg())
    planner = AttackGraphPlanner()
    goal_id = ag.get_goal_nodes()[0].id

    result = planner.plan(ag, goal_id=goal_id, top_k=2, max_depth=2)
    request = planner.build_task_generation_request(ag, result)

    assert isinstance(request, TaskGenerationRequest)
    assert request.candidates
    assert request.group_label == f"Plan for {result.goal.goal_label}"


def test_critic_identifies_blocked_actions() -> None:
    projector = AttackGraphProjector()
    ag = projector.project(
        build_goal_focused_kg(),
        policy_context={
            "constraints": [
                {
                    "constraint_type": ConstraintNodeType.APPROVAL_GATE.value,
                    "label": "Approval gate",
                    "hard_or_soft": "hard",
                    "properties": {"approved": False},
                }
            ]
        },
    )
    critic = AttackGraphCritic()
    findings = critic.critique(ag, context=CriticContext())

    blocked_actions = ag.find_actions(ActionNodeType.ESTABLISH_MANAGED_SESSION)
    assert blocked_actions
    assert blocked_actions[0].id in findings.blocked_action_ids


class StaticGraphLLMAdvisor:
    def __init__(self, advice: GraphLLMPlannerAdvice) -> None:
        self.advice = advice
        self.calls = 0

    def advise(self, *, graph_context, goal_refs, policy_context=None, recent_signals=None):  # noqa: ANN001
        self.calls += 1
        assert graph_context.frontier_actions
        assert goal_refs
        assert isinstance(policy_context, dict)
        assert isinstance(recent_signals, list)
        return self.advice


def _planner_input(*, enable_graph_llm: bool = False) -> AgentInput:
    ag = AttackGraphProjector().project(build_goal_focused_kg())
    goal_node = ag.get_goal_nodes()[0]
    return AgentInput(
        graph_refs=[
            GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
            GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode"),
        ],
        context=AgentContext(operation_id="op-graph-llm-planner-test"),
        raw_payload={
            "ag_graph": ag.to_dict(),
            "goal_refs": [GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode").model_dump(mode="json")],
            "policy_context": {"authorized_hosts": ["host-1"]},
            "recent_signals": [{"kind": "repetition", "decision": "allow"}],
            "planning_context": {
                "top_k": 1,
                "max_depth": 2,
                "enable_graph_llm_planning": enable_graph_llm,
            },
        },
    )


def _accepted_graph_llm_advice() -> GraphLLMPlannerAdvice:
    host_ref = AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")
    proposal = GraphLLMPlanProposal(
        proposal_id="proposal-graph-1",
        task_proposals=[
            GraphLLMTaskProposal(
                proposal_id="task-proposal-1",
                task_type=TaskType.SERVICE_VALIDATION.value,
                target_refs=[host_ref.model_dump(mode="json")],
                rationale="LLM chose service validation for evidence gain",
                expected_evidence=["validated service evidence"],
                estimated_risk=0.1,
                estimated_noise=0.1,
                priority=80,
            )
        ],
    )
    return GraphLLMPlannerAdvice(
        proposal=proposal,
        validation=GraphLLMPlanValidationResult.accepted_result(
            sanitized_payload=proposal.model_dump(mode="json"),
        ),
        llm_metadata={"model": "gpt-test", "base_url": "https://llm.example/v1"},
    )


def test_planner_default_does_not_use_graph_llm_advisor() -> None:
    advisor = StaticGraphLLMAdvisor(_accepted_graph_llm_advice())
    planner = PlannerAgent(graph_llm_advisor=advisor)

    result = planner.run(_planner_input(enable_graph_llm=False))

    assert result.success is True
    assert advisor.calls == 0
    candidate = result.output.decisions[0]["payload"]["planning_candidate"]
    assert candidate["metadata"]["source"] != "graph_llm_plan_proposal"


def test_planner_uses_accepted_graph_llm_proposal_as_task_candidate() -> None:
    advisor = StaticGraphLLMAdvisor(_accepted_graph_llm_advice())
    planner = PlannerAgent(graph_llm_advisor=advisor)

    result = planner.run(_planner_input(enable_graph_llm=True))

    assert result.success is True
    assert advisor.calls == 1
    candidate = result.output.decisions[0]["payload"]["planning_candidate"]
    assert candidate["metadata"]["source"] == "graph_llm_plan_proposal"
    assert candidate["metadata"]["graph_llm_proposal_id"] == "proposal-graph-1"
    assert candidate["metadata"]["graph_llm_metadata"]["model"] == "gpt-test"
    assert candidate["metadata"]["graph_context_summary"]["frontier_action_count"] >= 1
    task_candidate = candidate["task_candidates"][0]
    assert task_candidate["task_type"] == TaskType.SERVICE_VALIDATION.value
    assert task_candidate["source_action_id"].startswith("graph-llm::")
    assert task_candidate["target_refs"][0]["ref_id"] == "host-1"


def test_planner_falls_back_when_graph_llm_proposal_rejected() -> None:
    rejected_advice = GraphLLMPlannerAdvice.empty(reason="graph plan task proposal references unknown ref")
    advisor = StaticGraphLLMAdvisor(rejected_advice)
    planner = PlannerAgent(graph_llm_advisor=advisor)

    result = planner.run(_planner_input(enable_graph_llm=True))

    assert result.success is True
    assert advisor.calls == 1
    candidate = result.output.decisions[0]["payload"]["planning_candidate"]
    assert candidate["metadata"]["source"] != "graph_llm_plan_proposal"
    assert any("graph llm planner proposal rejected" in log for log in result.output.logs)
