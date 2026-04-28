from __future__ import annotations

from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import ActionNodeType, ConstraintNodeType
from src.core.models.kg import DataAsset, Goal, Host, HostsEdge, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.graph.tg_builder import TaskGenerationRequest
from src.core.planner.critic import AttackGraphCritic, CriticContext
from src.core.planner.planner import AttackGraphPlanner


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
