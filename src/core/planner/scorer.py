"""Heuristic scoring for Attack Graph states, actions and paths."""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import ActionNode, StateNode


class ScoringContext(BaseModel):
    """Weighting and penalty context for deterministic heuristics."""

    model_config = ConfigDict(extra="forbid")

    goal_relevance_weight: float = 0.35
    confidence_gain_weight: float = 0.15
    expected_value_weight: float = 0.2
    success_probability_weight: float = 0.15
    cost_penalty_weight: float = 0.05
    risk_penalty_weight: float = 0.05
    noise_penalty_weight: float = 0.03
    uncertainty_penalty_weight: float = 0.02
    depth_penalty_weight: float = 0.05


def score_state(state_node: StateNode, context: ScoringContext | None = None) -> float:
    """Score a state for planner prioritization."""

    ctx = context or ScoringContext()
    validation_bonus = 1.0 if state_node.truth_status.value == "validated" else 0.5
    uncertainty_penalty = 1.0 - state_node.confidence
    return max(
        0.0,
        (state_node.goal_relevance * ctx.goal_relevance_weight)
        + (state_node.confidence * ctx.confidence_gain_weight)
        + (validation_bonus * 0.1)
        - (uncertainty_penalty * ctx.uncertainty_penalty_weight),
    )


def score_action(action_node: ActionNode, context: ScoringContext | None = None) -> float:
    """Score an action node using expected value and penalty heuristics."""

    ctx = context or ScoringContext()
    uncertainty_penalty = 1.0 - action_node.success_probability_prior
    return (
        (action_node.goal_relevance * ctx.goal_relevance_weight)
        + (action_node.expected_value * ctx.expected_value_weight)
        + (action_node.success_probability_prior * ctx.success_probability_weight)
        - (action_node.cost * ctx.cost_penalty_weight)
        - (action_node.risk * ctx.risk_penalty_weight)
        - (action_node.noise * ctx.noise_penalty_weight)
        - (uncertainty_penalty * ctx.uncertainty_penalty_weight)
    )


def score_path(path: Sequence[ActionNode], context: ScoringContext | None = None) -> float:
    """Score a path as the discounted sum of action scores."""

    ctx = context or ScoringContext()
    if not path:
        return 0.0
    total = 0.0
    for depth, action in enumerate(path):
        total += score_action(action, ctx) - (depth * ctx.depth_penalty_weight)
    return total / len(path)


class HeuristicScorer:
    """Small wrapper around the module-level scoring functions."""

    def __init__(self, context: ScoringContext | None = None) -> None:
        self.context = context or ScoringContext()

    def score_action(self, action_node: ActionNode) -> float:
        """Score one action node."""

        return score_action(action_node, self.context)

    def score_state(self, state_node: StateNode) -> float:
        """Score one state node."""

        return score_state(state_node, self.context)

    def score_path(self, path: Sequence[ActionNode]) -> float:
        """Score a candidate action path."""

        return score_path(path, self.context)
