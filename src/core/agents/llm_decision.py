"""Shared LLM decision models for bounded agent advice."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import new_record_id


class LLMDecisionStatus(str, Enum):
    """Validation status for an LLM-produced decision proposal."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class LLMDecisionSource(str, Enum):
    """Agent surface that produced or consumed an LLM decision."""

    PLANNER = "planner"
    CRITIC = "critic"
    SUPERVISOR = "supervisor"


class LLMDecision(BaseModel):
    """Normalized, bounded decision proposal returned by an LLM advisor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    decision_id: str = Field(default_factory=lambda: new_record_id("llm-decision"))
    source: LLMDecisionSource
    status: LLMDecisionStatus = LLMDecisionStatus.ACCEPTED
    decision_type: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    target_kind: str = Field(min_length=1)
    score_delta: float | None = None
    rationale_suffix: str | None = None
    summary_override: str | None = None
    risk_notes: list[str] = Field(default_factory=list)
    replan_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMDecisionValidationResult(BaseModel):
    """Result of validating a bounded LLM decision proposal."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: LLMDecisionStatus
    accepted: bool = False
    reason: str = Field(min_length=1)
    sanitized_payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def accepted_result(cls, *, sanitized_payload: dict[str, Any]) -> "LLMDecisionValidationResult":
        return cls(
            status=LLMDecisionStatus.ACCEPTED,
            accepted=True,
            reason="accepted",
            sanitized_payload=sanitized_payload,
        )

    @classmethod
    def rejected_result(
        cls,
        *,
        reason: str,
        sanitized_payload: dict[str, Any] | None = None,
    ) -> "LLMDecisionValidationResult":
        return cls(
            status=LLMDecisionStatus.REJECTED,
            accepted=False,
            reason=reason,
            sanitized_payload=sanitized_payload or {},
        )


FORBIDDEN_LLM_DECISION_KEYS = {
    "command",
    "commands",
    "shell",
    "shell_command",
    "tool_command",
    "tool_args",
    "tool_parameters",
    "execute",
    "patch",
    "state_delta",
    "kg_delta",
    "ag_delta",
    "tg_delta",
    "cancel_task",
    "replace_task",
}


def contains_forbidden_llm_decision_key(payload: dict[str, Any]) -> str | None:
    """Return the first forbidden key present in an LLM payload."""

    for key, value in payload.items():
        if key in FORBIDDEN_LLM_DECISION_KEYS:
            return key
        if isinstance(value, dict):
            nested = contains_forbidden_llm_decision_key(value)
            if nested is not None:
                return nested
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    nested = contains_forbidden_llm_decision_key(item)
                    if nested is not None:
                        return nested
    return None


class LLMDecisionValidator:
    """Validate bounded LLM decisions before agent consumption."""

    def validate_no_forbidden_payload(self, payload: dict[str, Any]) -> LLMDecisionValidationResult:
        forbidden_key = contains_forbidden_llm_decision_key(payload)
        if forbidden_key is None:
            return LLMDecisionValidationResult.accepted_result(sanitized_payload=dict(payload))
        return LLMDecisionValidationResult.rejected_result(
            reason=f"llm payload contains forbidden field: {forbidden_key}",
        )

    def validate_planner_decision(
        self,
        decision: LLMDecision,
        *,
        allowed_candidate_ids: set[str],
        max_abs_score_delta: float = 0.2,
        policy_context: dict[str, Any] | None = None,
        runtime_summary: dict[str, Any] | None = None,
    ) -> LLMDecisionValidationResult:
        common = self._validate_common(
            decision,
            expected_source=LLMDecisionSource.PLANNER,
            expected_decision_type="planner_candidate_advice",
            expected_target_kind="planning_candidate",
            allowed_target_ids=allowed_candidate_ids,
            policy_context=policy_context,
            runtime_summary=runtime_summary,
        )
        if not common.accepted:
            return common
        if decision.summary_override is not None or decision.replan_hint is not None:
            return LLMDecisionValidationResult.rejected_result(
                reason="planner decision contains critic-only fields",
            )
        score_delta = decision.score_delta or 0.0
        if abs(score_delta) > max_abs_score_delta:
            return LLMDecisionValidationResult.rejected_result(
                reason="planner score_delta exceeds allowed range",
                sanitized_payload={"target_id": decision.target_id, "score_delta": score_delta},
            )
        return LLMDecisionValidationResult.accepted_result(
            sanitized_payload={
                "candidate_id": decision.target_id,
                "score_delta": score_delta,
                "rationale_suffix": decision.rationale_suffix,
                "risk_notes": self._string_list(decision.risk_notes),
                "metadata": self._safe_metadata(decision.metadata),
            },
        )

    def validate_planner_strategy_decision(
        self,
        decision: LLMDecision,
        *,
        allowed_candidate_ids: set[str],
        allowed_goal_ids: set[str],
        selected_candidate_ids: list[str],
        rank_adjustments: list[dict[str, Any]],
        max_abs_score_delta: float = 0.2,
        policy_context: dict[str, Any] | None = None,
        runtime_summary: dict[str, Any] | None = None,
    ) -> LLMDecisionValidationResult:
        common = self._validate_common(
            decision,
            expected_source=LLMDecisionSource.PLANNER,
            expected_decision_type="planner_strategy_decision",
            expected_target_kind="planner_goal",
            allowed_target_ids=allowed_goal_ids,
            policy_context=policy_context,
            runtime_summary=runtime_summary,
        )
        if not common.accepted:
            return common
        unknown_selected = [item for item in selected_candidate_ids if item not in allowed_candidate_ids]
        if unknown_selected:
            return LLMDecisionValidationResult.rejected_result(
                reason="planner strategy selected unknown candidate_id",
                sanitized_payload={"unknown_candidate_ids": unknown_selected},
            )
        sanitized_adjustments: list[dict[str, Any]] = []
        for item in rank_adjustments:
            candidate_id = item.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id not in allowed_candidate_ids:
                return LLMDecisionValidationResult.rejected_result(
                    reason="planner strategy adjusted unknown candidate_id",
                    sanitized_payload={"candidate_id": candidate_id},
                )
            score_delta = self._coerce_float(item.get("score_delta"), default=0.0)
            if abs(score_delta) > max_abs_score_delta:
                return LLMDecisionValidationResult.rejected_result(
                    reason="planner strategy score_delta exceeds allowed range",
                    sanitized_payload={"candidate_id": candidate_id, "score_delta": score_delta},
                )
            metadata = item.get("metadata")
            sanitized_adjustments.append(
                {
                    "candidate_id": candidate_id,
                    "score_delta": score_delta,
                    "rationale_suffix": item.get("rationale_suffix")
                    if isinstance(item.get("rationale_suffix"), str)
                    else None,
                    "risk_notes": self._string_list(item.get("risk_notes") if isinstance(item.get("risk_notes"), list) else []),
                    "metadata": self._safe_metadata(metadata if isinstance(metadata, dict) else {}),
                }
            )
        return LLMDecisionValidationResult.accepted_result(
            sanitized_payload={
                "selected_candidate_ids": list(selected_candidate_ids),
                "rank_adjustments": sanitized_adjustments,
                "risk_notes": self._string_list(decision.risk_notes),
                "defer_reason": decision.metadata.get("defer_reason")
                if isinstance(decision.metadata.get("defer_reason"), str)
                else None,
                "requires_human_review": bool(decision.metadata.get("requires_human_review", False)),
                "metadata": self._safe_metadata(decision.metadata),
            },
        )

    def validate_critic_decision(
        self,
        decision: LLMDecision,
        *,
        allowed_finding_ids: set[str],
        policy_context: dict[str, Any] | None = None,
        runtime_summary: dict[str, Any] | None = None,
    ) -> LLMDecisionValidationResult:
        common = self._validate_common(
            decision,
            expected_source=LLMDecisionSource.CRITIC,
            expected_decision_type="critic_finding_review",
            expected_target_kind="critic_finding",
            allowed_target_ids=allowed_finding_ids,
            policy_context=policy_context,
            runtime_summary=runtime_summary,
        )
        if not common.accepted:
            return common
        if decision.score_delta is not None or decision.risk_notes:
            return LLMDecisionValidationResult.rejected_result(
                reason="critic decision contains planner-only fields",
                sanitized_payload={"target_id": decision.target_id},
            )
        return LLMDecisionValidationResult.accepted_result(
            sanitized_payload={
                "finding_id": decision.target_id,
                "summary_override": decision.summary_override,
                "rationale_suffix": decision.rationale_suffix,
                "replan_hint": decision.replan_hint,
                "metadata": self._safe_metadata(decision.metadata),
            },
        )

    def validate_critic_replan_proposal(
        self,
        decision: LLMDecision,
        *,
        allowed_finding_ids: set[str],
        allowed_task_ids: set[str],
        affected_task_ids: list[str],
        confidence: float,
        policy_context: dict[str, Any] | None = None,
        runtime_summary: dict[str, Any] | None = None,
    ) -> LLMDecisionValidationResult:
        common = self._validate_common(
            decision,
            expected_source=LLMDecisionSource.CRITIC,
            expected_decision_type="critic_replan_proposal",
            expected_target_kind="critic_finding",
            allowed_target_ids=allowed_finding_ids,
            policy_context=policy_context,
            runtime_summary=runtime_summary,
        )
        if not common.accepted:
            return common
        unknown_tasks = [task_id for task_id in affected_task_ids if task_id not in allowed_task_ids]
        if unknown_tasks:
            return LLMDecisionValidationResult.rejected_result(
                reason="critic replan proposal references unknown task_id",
                sanitized_payload={"unknown_task_ids": unknown_tasks},
            )
        if confidence < 0.0 or confidence > 1.0:
            return LLMDecisionValidationResult.rejected_result(
                reason="critic replan proposal confidence is outside [0, 1]",
                sanitized_payload={"confidence": confidence},
            )
        return LLMDecisionValidationResult.accepted_result(
            sanitized_payload={
                "finding_id": decision.target_id,
                "failure_summary": decision.summary_override,
                "replan_hint": decision.replan_hint,
                "affected_task_ids": list(affected_task_ids),
                "confidence": confidence,
                "requires_human_review": bool(decision.metadata.get("requires_human_review", False)),
                "metadata": self._safe_metadata(decision.metadata),
            },
        )

    def _validate_common(
        self,
        decision: LLMDecision,
        *,
        expected_source: LLMDecisionSource,
        expected_decision_type: str,
        expected_target_kind: str,
        allowed_target_ids: set[str],
        policy_context: dict[str, Any] | None,
        runtime_summary: dict[str, Any] | None,
    ) -> LLMDecisionValidationResult:
        if self._disabled_by_policy(policy_context):
            return LLMDecisionValidationResult.rejected_result(reason="llm decisions disabled by policy")
        if self._disabled_by_runtime(runtime_summary):
            return LLMDecisionValidationResult.rejected_result(reason="llm decisions disabled by runtime")
        if decision.source != expected_source:
            return LLMDecisionValidationResult.rejected_result(reason="unexpected llm decision source")
        if decision.decision_type != expected_decision_type:
            return LLMDecisionValidationResult.rejected_result(reason="unexpected llm decision type")
        if decision.target_kind != expected_target_kind:
            return LLMDecisionValidationResult.rejected_result(reason="unexpected llm decision target kind")
        if decision.target_id not in allowed_target_ids:
            return LLMDecisionValidationResult.rejected_result(
                reason="llm decision target is not in the current agent input",
                sanitized_payload={"target_id": decision.target_id},
            )
        forbidden_key = contains_forbidden_llm_decision_key(decision.model_dump(mode="json"))
        if forbidden_key is not None:
            return LLMDecisionValidationResult.rejected_result(
                reason=f"llm decision contains forbidden field: {forbidden_key}",
            )
        return LLMDecisionValidationResult.accepted_result(sanitized_payload={})

    @staticmethod
    def _disabled_by_policy(policy_context: dict[str, Any] | None) -> bool:
        policy = policy_context or {}
        return bool(policy.get("disable_llm_decisions") or policy.get("llm_decisions_disabled"))

    @staticmethod
    def _disabled_by_runtime(runtime_summary: dict[str, Any] | None) -> bool:
        runtime = runtime_summary or {}
        return bool(runtime.get("disable_llm_decisions") or runtime.get("llm_decisions_disabled"))

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if "api_key" not in key.lower() and "secret" not in key.lower()
        }

    @staticmethod
    def _string_list(items: list[str]) -> list[str]:
        return [item for item in items if isinstance(item, str) and item.strip()]

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


__all__ = [
    "FORBIDDEN_LLM_DECISION_KEYS",
    "LLMDecision",
    "LLMDecisionSource",
    "LLMDecisionStatus",
    "LLMDecisionValidationResult",
    "LLMDecisionValidator",
    "contains_forbidden_llm_decision_key",
]
