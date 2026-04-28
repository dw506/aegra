from __future__ import annotations

from src.core.models.ag import (
    ActionNode,
    ActionNodeType,
    ActivationStatus,
    GraphRef,
    StateNode,
    StateNodeType,
    TruthStatus,
)
from src.core.planner.scorer import HeuristicScorer, score_action, score_path, score_state


def test_state_and_action_scoring_are_positive() -> None:
    state = StateNode(
        id="state-1",
        node_type=StateNodeType.SERVICE_KNOWN,
        label="Service known",
        subject_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
        truth_status=TruthStatus.ACTIVE,
        confidence=0.8,
        goal_relevance=0.9,
        created_from=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
    )
    action = ActionNode(
        id="action-1",
        action_type=ActionNodeType.VALIDATE_SERVICE,
        label="Validate service",
        bound_args={"service_id": "svc-1"},
        expected_value=0.8,
        success_probability_prior=0.75,
        goal_relevance=0.9,
        cost=0.2,
        risk=0.1,
        noise=0.1,
        activation_status=ActivationStatus.ACTIVATABLE,
    )

    assert score_state(state) > 0
    assert score_action(action) > 0
    assert score_path([action]) > 0
    assert HeuristicScorer().score_action(action) == score_action(action)
