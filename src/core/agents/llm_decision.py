"""Shared LLM decision models for bounded agent advice."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.agent_models import new_record_id
from src.core.agents.graph_llm_models import (
    GraphLLMPlanProposal,
    GraphLLMPlanValidationResult,
    GraphLLMRankAdjustment,
    GraphLLMTaskProposal,
)
from src.core.models.ag import GraphRef
from src.core.models.scope import Asset
from src.core.models.task_types import TaskType
from src.core.runtime.policy import RuntimePolicy
from src.core.runtime.policy_engine import PolicyEngine


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
    "payload",
    "raw_payload",
    "reverse_shell",
    "reverse_callback",
    "destructive_action",
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

    DEFAULT_GRAPH_PLAN_REVIEW_THRESHOLD = 0.7

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

    def validate_graph_plan_proposal(
        self,
        proposal: GraphLLMPlanProposal | dict[str, Any],
        *,
        visible_refs: set[str] | list[str] | set[GraphRef] | list[GraphRef],
        policy_context: dict[str, Any] | RuntimePolicy | None = None,
        runtime_summary: dict[str, Any] | None = None,
        risk_review_threshold: float = DEFAULT_GRAPH_PLAN_REVIEW_THRESHOLD,
        noise_review_threshold: float = DEFAULT_GRAPH_PLAN_REVIEW_THRESHOLD,
    ) -> GraphLLMPlanValidationResult:
        """Validate graph-planning proposals before execution planning."""

        raw_payload = proposal if isinstance(proposal, dict) else proposal.model_dump(mode="json")
        if not isinstance(raw_payload, dict):
            return GraphLLMPlanValidationResult.rejected_result(reason="graph plan proposal must be an object")
        forbidden_key = contains_forbidden_llm_decision_key(raw_payload)
        if forbidden_key is not None:
            return GraphLLMPlanValidationResult.rejected_result(
                reason=f"graph plan proposal contains forbidden field: {forbidden_key}",
            )
        if self._disabled_by_policy_dict(policy_context):
            return GraphLLMPlanValidationResult.rejected_result(reason="graph llm planning disabled by policy")
        if self._disabled_by_runtime(runtime_summary):
            return GraphLLMPlanValidationResult.rejected_result(reason="graph llm planning disabled by runtime")

        try:
            parsed = proposal if isinstance(proposal, GraphLLMPlanProposal) else GraphLLMPlanProposal.model_validate(raw_payload)
        except ValidationError as exc:
            return GraphLLMPlanValidationResult.rejected_result(
                reason="invalid graph plan proposal schema",
                rejected_items=[{"error": str(exc)}],
            )

        visible_ref_keys = self._visible_ref_keys(visible_refs)
        requires_review = bool(parsed.requires_human_review)
        sanitized_tasks: list[dict[str, Any]] = []
        for task in parsed.task_proposals:
            task_result = self._validate_graph_task_proposal(
                task,
                visible_ref_keys=visible_ref_keys,
                policy_context=policy_context,
                risk_review_threshold=risk_review_threshold,
                noise_review_threshold=noise_review_threshold,
            )
            if not task_result.accepted:
                return task_result
            requires_review = requires_review or task_result.requires_human_review
            sanitized_tasks.append(task_result.sanitized_payload)

        sanitized_adjustments: list[dict[str, Any]] = []
        for adjustment in parsed.rank_adjustments:
            if adjustment.target_ref.key() not in visible_ref_keys:
                return GraphLLMPlanValidationResult.rejected_result(
                    reason="graph plan rank adjustment references unknown ref",
                    sanitized_payload={"target_ref": adjustment.target_ref.model_dump(mode="json")},
                )
            sanitized_adjustments.append(self._sanitize_rank_adjustment(adjustment))

        return GraphLLMPlanValidationResult.accepted_result(
            requires_human_review=requires_review,
            reason="accepted_requires_human_review" if requires_review else "accepted",
            sanitized_payload={
                "proposal_id": parsed.proposal_id,
                "task_proposals": sanitized_tasks,
                "rank_adjustments": sanitized_adjustments,
                "replan_hint": parsed.replan_hint,
                "risk_notes": self._string_list(parsed.risk_notes),
                "requires_human_review": requires_review,
                "metadata": self._safe_metadata(parsed.metadata),
            },
        )

    def _validate_graph_task_proposal(
        self,
        task: GraphLLMTaskProposal,
        *,
        visible_ref_keys: set[str],
        policy_context: dict[str, Any] | RuntimePolicy | None,
        risk_review_threshold: float,
        noise_review_threshold: float,
    ) -> GraphLLMPlanValidationResult:
        task_type = self._coerce_task_type(task.task_type)
        if task_type is None:
            return GraphLLMPlanValidationResult.rejected_result(
                reason="graph plan task_type is not allowed",
                sanitized_payload={"task_type": task.task_type},
            )
        if not task.target_refs:
            return GraphLLMPlanValidationResult.rejected_result(reason="graph plan task proposal has no target_refs")
        unknown_refs = [ref.model_dump(mode="json") for ref in task.target_refs if ref.key() not in visible_ref_keys]
        if unknown_refs:
            return GraphLLMPlanValidationResult.rejected_result(
                reason="graph plan task proposal references unknown ref",
                sanitized_payload={"unknown_refs": unknown_refs},
            )
        scope_denial = self._scope_denial_for_refs(task.target_refs, policy_context)
        if scope_denial is not None:
            return GraphLLMPlanValidationResult.rejected_result(
                reason="graph plan task proposal is outside runtime policy scope",
                sanitized_payload=scope_denial,
            )
        requires_review = bool(task.requires_human_review)
        if task.estimated_risk >= risk_review_threshold or task.estimated_noise >= noise_review_threshold:
            requires_review = True
        return GraphLLMPlanValidationResult.accepted_result(
            requires_human_review=requires_review,
            reason="accepted_requires_human_review" if requires_review else "accepted",
            sanitized_payload={
                "proposal_id": task.proposal_id,
                "task_type": task_type.value,
                "target_refs": [ref.model_dump(mode="json") for ref in task.target_refs],
                "rationale": task.rationale,
                "expected_evidence": self._string_list(task.expected_evidence),
                "tool_hint": task.tool_hint,
                "params": self._safe_metadata(task.params),
                "estimated_risk": task.estimated_risk,
                "estimated_noise": task.estimated_noise,
                "priority": task.priority,
                "requires_human_review": requires_review,
                "metadata": self._safe_metadata(task.metadata),
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
    def _disabled_by_policy_dict(policy_context: dict[str, Any] | RuntimePolicy | None) -> bool:
        if isinstance(policy_context, RuntimePolicy):
            return False
        policy = policy_context or {}
        return bool(
            policy.get("disable_llm_decisions")
            or policy.get("llm_decisions_disabled")
            or policy.get("disable_graph_llm_planning")
            or policy.get("graph_llm_planning_disabled")
        )

    @staticmethod
    def _disabled_by_runtime(runtime_summary: dict[str, Any] | None) -> bool:
        runtime = runtime_summary or {}
        return bool(runtime.get("disable_llm_decisions") or runtime.get("llm_decisions_disabled"))

    @staticmethod
    def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metadata.items()
            if "api_key" not in key.lower()
            and "secret" not in key.lower()
            and key not in FORBIDDEN_LLM_DECISION_KEYS
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

    @staticmethod
    def _coerce_task_type(value: str) -> TaskType | None:
        try:
            return TaskType(value)
        except ValueError:
            try:
                return TaskType[value]
            except KeyError:
                return None

    @staticmethod
    def _visible_ref_keys(refs: set[str] | list[str] | set[GraphRef] | list[GraphRef]) -> set[str]:
        keys: set[str] = set()
        for ref in refs:
            if isinstance(ref, GraphRef):
                keys.add(ref.key())
            else:
                keys.add(str(ref))
        return keys

    def _scope_denial_for_refs(
        self,
        refs: list[GraphRef],
        policy_context: dict[str, Any] | RuntimePolicy | None,
    ) -> dict[str, Any] | None:
        policy = self._runtime_policy(policy_context)
        if policy is None:
            return None
        engine = PolicyEngine(policy)
        for ref in refs:
            asset = self._asset_from_ref(ref)
            if asset is None:
                continue
            decision = engine.evaluate_target_scope(asset)
            if decision.decision != "allow":
                return {
                    "ref": ref.model_dump(mode="json"),
                    "target": decision.target,
                    "policy_reason": decision.reason,
                }
        return None

    @staticmethod
    def _runtime_policy(policy_context: dict[str, Any] | RuntimePolicy | None) -> RuntimePolicy | None:
        if policy_context is None:
            return None
        if isinstance(policy_context, RuntimePolicy):
            return policy_context
        policy_payload = policy_context.get("runtime_policy") if isinstance(policy_context.get("runtime_policy"), dict) else policy_context
        try:
            return RuntimePolicy.model_validate(policy_payload)
        except ValidationError:
            return None

    @staticmethod
    def _asset_from_ref(ref: GraphRef) -> Asset | None:
        ref_type = (ref.ref_type or "").lower()
        if ref_type not in {"host", "service", "url", "domain"}:
            return None
        value = ref.label or ref.ref_id
        if ref_type == "service":
            return Asset(kind="service", value=value)
        if ref_type == "url":
            return Asset(kind="url", value=value)
        if ref_type == "domain":
            return Asset(kind="domain", value=value)
        return Asset(kind="host", value=value)

    def _sanitize_rank_adjustment(self, adjustment: GraphLLMRankAdjustment) -> dict[str, Any]:
        return {
            "target_ref": adjustment.target_ref.model_dump(mode="json"),
            "score_delta": adjustment.score_delta,
            "rationale": adjustment.rationale,
            "metadata": self._safe_metadata(adjustment.metadata),
        }


def validate_graph_plan_proposal(
    proposal: GraphLLMPlanProposal | dict[str, Any],
    *,
    visible_refs: set[str] | list[str] | set[GraphRef] | list[GraphRef],
    policy_context: dict[str, Any] | RuntimePolicy | None = None,
    runtime_summary: dict[str, Any] | None = None,
    risk_review_threshold: float = LLMDecisionValidator.DEFAULT_GRAPH_PLAN_REVIEW_THRESHOLD,
    noise_review_threshold: float = LLMDecisionValidator.DEFAULT_GRAPH_PLAN_REVIEW_THRESHOLD,
) -> GraphLLMPlanValidationResult:
    """Validate one graph LLM proposal using the default validator."""

    return LLMDecisionValidator().validate_graph_plan_proposal(
        proposal,
        visible_refs=visible_refs,
        policy_context=policy_context,
        runtime_summary=runtime_summary,
        risk_review_threshold=risk_review_threshold,
        noise_review_threshold=noise_review_threshold,
    )


__all__ = [
    "FORBIDDEN_LLM_DECISION_KEYS",
    "LLMDecision",
    "LLMDecisionSource",
    "LLMDecisionStatus",
    "LLMDecisionValidationResult",
    "LLMDecisionValidator",
    "contains_forbidden_llm_decision_key",
    "validate_graph_plan_proposal",
]
