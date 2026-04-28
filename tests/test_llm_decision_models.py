from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.agents.llm_decision import (
    LLMDecision,
    LLMDecisionSource,
    LLMDecisionStatus,
    LLMDecisionValidationResult,
    LLMDecisionValidator,
    contains_forbidden_llm_decision_key,
)


def test_llm_decision_model_accepts_bounded_planner_payload() -> None:
    decision = LLMDecision(
        source=LLMDecisionSource.PLANNER,
        status=LLMDecisionStatus.ACCEPTED,
        decision_type="planner_candidate_advice",
        target_id="candidate-1",
        target_kind="planning_candidate",
        score_delta=0.1,
        rationale_suffix="更贴近目标",
        risk_notes=["低噪声路径"],
        metadata={"reason": "goal_alignment"},
    )

    assert decision.source == LLMDecisionSource.PLANNER
    assert decision.target_id == "candidate-1"
    assert decision.risk_notes == ["低噪声路径"]


def test_llm_decision_model_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LLMDecision.model_validate(
            {
                "source": "critic",
                "status": "accepted",
                "decision_type": "critic_finding_review",
                "target_id": "finding-1",
                "target_kind": "critic_finding",
                "command": "run-tool",
            }
        )


def test_llm_decision_validation_result_helpers_are_structured() -> None:
    accepted = LLMDecisionValidationResult.accepted_result(
        sanitized_payload={"target_id": "candidate-1"},
    )
    rejected = LLMDecisionValidationResult.rejected_result(reason="unknown target")

    assert accepted.accepted is True
    assert accepted.status == LLMDecisionStatus.ACCEPTED
    assert rejected.accepted is False
    assert rejected.status == LLMDecisionStatus.REJECTED


def test_forbidden_key_detection_catches_tool_or_graph_mutation_attempts() -> None:
    assert contains_forbidden_llm_decision_key({"tool_command": "nmap"}) == "tool_command"
    assert contains_forbidden_llm_decision_key({"kg_delta": {"nodes": []}}) == "kg_delta"
    assert contains_forbidden_llm_decision_key({"metadata": {"patch": {"x": 1}}}) == "patch"
    assert contains_forbidden_llm_decision_key({"rationale_suffix": "ok"}) is None


def test_llm_decision_validator_accepts_and_sanitizes_planner_decision() -> None:
    validator = LLMDecisionValidator()
    decision = LLMDecision(
        source=LLMDecisionSource.PLANNER,
        decision_type="planner_candidate_advice",
        target_id="candidate-1",
        target_kind="planning_candidate",
        score_delta=0.1,
        rationale_suffix="更贴近目标",
        risk_notes=["低噪声"],
        metadata={"reason": "goal_alignment", "api_key_hint": "must-not-leak"},
    )

    result = validator.validate_planner_decision(
        decision,
        allowed_candidate_ids={"candidate-1"},
    )

    assert result.accepted is True
    assert result.sanitized_payload["candidate_id"] == "candidate-1"
    assert result.sanitized_payload["score_delta"] == 0.1
    assert "api_key_hint" not in result.sanitized_payload["metadata"]


def test_llm_decision_validator_rejects_unknown_or_out_of_range_planner_decision() -> None:
    validator = LLMDecisionValidator()
    unknown = LLMDecision(
        source=LLMDecisionSource.PLANNER,
        decision_type="planner_candidate_advice",
        target_id="unknown",
        target_kind="planning_candidate",
    )
    out_of_range = LLMDecision(
        source=LLMDecisionSource.PLANNER,
        decision_type="planner_candidate_advice",
        target_id="candidate-1",
        target_kind="planning_candidate",
        score_delta=0.5,
    )

    assert not validator.validate_planner_decision(
        unknown,
        allowed_candidate_ids={"candidate-1"},
    ).accepted
    rejected = validator.validate_planner_decision(
        out_of_range,
        allowed_candidate_ids={"candidate-1"},
        max_abs_score_delta=0.2,
    )
    assert rejected.accepted is False
    assert "score_delta" in rejected.reason


def test_llm_decision_validator_rejects_policy_runtime_and_forbidden_payloads() -> None:
    validator = LLMDecisionValidator()
    decision = LLMDecision(
        source=LLMDecisionSource.CRITIC,
        decision_type="critic_finding_review",
        target_id="finding-1",
        target_kind="critic_finding",
        metadata={"reason": "dependency_failure"},
    )

    assert not validator.validate_critic_decision(
        decision,
        allowed_finding_ids={"finding-1"},
        policy_context={"disable_llm_decisions": True},
    ).accepted
    assert not validator.validate_critic_decision(
        decision,
        allowed_finding_ids={"finding-1"},
        runtime_summary={"llm_decisions_disabled": True},
    ).accepted
    assert not validator.validate_no_forbidden_payload({"tool_command": "nmap"}).accepted


def test_llm_decision_validator_accepts_planner_strategy_decision() -> None:
    validator = LLMDecisionValidator()
    decision = LLMDecision(
        source=LLMDecisionSource.PLANNER,
        decision_type="planner_strategy_decision",
        target_id="goal-1",
        target_kind="planner_goal",
        risk_notes=["低噪声候选优先"],
        metadata={"requires_human_review": True, "defer_reason": "等待人工复核"},
    )

    result = validator.validate_planner_strategy_decision(
        decision,
        allowed_candidate_ids={"candidate-1", "candidate-2"},
        allowed_goal_ids={"goal-1"},
        selected_candidate_ids=["candidate-2"],
        rank_adjustments=[
            {
                "candidate_id": "candidate-2",
                "score_delta": 0.1,
                "rationale_suffix": "更贴近目标",
                "risk_notes": ["低噪声"],
            }
        ],
    )

    assert result.accepted is True
    assert result.sanitized_payload["selected_candidate_ids"] == ["candidate-2"]
    assert result.sanitized_payload["requires_human_review"] is True
    assert result.sanitized_payload["rank_adjustments"][0]["score_delta"] == 0.1


def test_llm_decision_validator_rejects_planner_strategy_unknown_candidate() -> None:
    validator = LLMDecisionValidator()
    decision = LLMDecision(
        source=LLMDecisionSource.PLANNER,
        decision_type="planner_strategy_decision",
        target_id="goal-1",
        target_kind="planner_goal",
    )

    result = validator.validate_planner_strategy_decision(
        decision,
        allowed_candidate_ids={"candidate-1"},
        allowed_goal_ids={"goal-1"},
        selected_candidate_ids=["candidate-2"],
        rank_adjustments=[],
    )

    assert result.accepted is False
    assert "unknown candidate" in result.reason


def test_llm_decision_validator_accepts_critic_replan_proposal() -> None:
    validator = LLMDecisionValidator()
    decision = LLMDecision(
        source=LLMDecisionSource.CRITIC,
        decision_type="critic_replan_proposal",
        target_id="finding-1",
        target_kind="critic_finding",
        summary_override="上游依赖失效",
        replan_hint="复核依赖后走现有 replan 流程",
        metadata={"requires_human_review": True},
    )

    result = validator.validate_critic_replan_proposal(
        decision,
        allowed_finding_ids={"finding-1"},
        allowed_task_ids={"task-1"},
        affected_task_ids=["task-1"],
        confidence=0.8,
    )

    assert result.accepted is True
    assert result.sanitized_payload["affected_task_ids"] == ["task-1"]
    assert result.sanitized_payload["requires_human_review"] is True


def test_llm_decision_validator_rejects_critic_replan_proposal_unknown_task() -> None:
    validator = LLMDecisionValidator()
    decision = LLMDecision(
        source=LLMDecisionSource.CRITIC,
        decision_type="critic_replan_proposal",
        target_id="finding-1",
        target_kind="critic_finding",
        replan_hint="复核依赖后走现有 replan 流程",
    )

    result = validator.validate_critic_replan_proposal(
        decision,
        allowed_finding_ids={"finding-1"},
        allowed_task_ids={"task-1"},
        affected_task_ids=["task-2"],
        confidence=0.8,
    )

    assert result.accepted is False
    assert "unknown task_id" in result.reason
