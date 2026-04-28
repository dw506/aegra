"""Critic agent for closed-loop control over KG/AG/TG/Runtime execution state.

The Critic identifies low-value or invalid task/branch patterns and emits only
recommendations: replan requests, cancel/replace suggestions, and advisory
state deltas. It never mutates KG facts directly.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import DecisionRecord, ReplanRequestRecord, StateDeltaRecord, new_record_id
from src.core.agents.agent_protocol import (
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.agents.llm_decision import (
    LLMDecision,
    LLMDecisionSource,
    LLMDecisionStatus,
    LLMDecisionValidationResult,
    LLMDecisionValidator,
)
from src.core.models.runtime import OutcomeCacheEntry, RuntimeState, TaskRuntimeStatus
from src.core.models.tg import BaseTaskNode, ReplanFrontier, TaskGraph, TaskStatus
from src.core.planner.critic import TaskCriticContext, TaskGraphCritic


class CriticContext(BaseModel):
    """Agent-layer context for Critic closed-loop analysis."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    duplicate_threshold: int = Field(default=2, ge=2)
    failure_threshold: int = Field(default=2, ge=1)
    low_value_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    permanently_blocked_reasons: set[str] = Field(
        default_factory=lambda: {"waiting for scheduling gate", "upstream dependency failed"}
    )
    invalidated_ref_keys: set[str] = Field(default_factory=set)
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    recent_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    critic_hints: dict[str, Any] = Field(default_factory=dict)


class CriticFinding(BaseModel):
    """Structured finding emitted by the Critic agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    finding_id: str = Field(default_factory=lambda: new_record_id("critic-finding"))
    finding_type: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    subject_refs: list[GraphRef] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CriticRecommendation(BaseModel):
    """Structured recommendation emitted by the Critic agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    recommendation_id: str = Field(default_factory=lambda: new_record_id("critic-rec"))
    recommendation_type: str = Field(min_length=1)
    action: str = Field(min_length=1)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    subject_refs: list[GraphRef] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    patch: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CriticLLMReview(BaseModel):
    """可选的 LLM 失败归纳结果。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    finding_id: str = Field(min_length=1)
    rationale_suffix: str | None = None
    summary_override: str | None = None
    replan_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    decision: LLMDecision | None = None
    validation: LLMDecisionValidationResult | None = None
    replan_proposal: "CriticLLMReplanProposal | None" = None


class CriticLLMReplanProposal(BaseModel):
    """Bounded LLM replan proposal attached to an existing critic finding."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    finding_id: str = Field(min_length=1)
    failure_summary: str | None = None
    replan_hint: str = Field(min_length=1)
    affected_task_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    requires_human_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    decision: LLMDecision | None = None
    validation: LLMDecisionValidationResult | None = None


class CriticLLMAdvisor(Protocol):
    """Critic 可选 LLM 接口。

    中文注释：
    LLM 只负责失败原因归纳和解释增强，不能直接取消任务、改写 KG，或触发底层动作。
    """

    def summarize_findings(
        self,
        *,
        findings: Sequence[CriticFinding],
        context: CriticContext,
        runtime_state: RuntimeState | None,
    ) -> list[CriticLLMReview]:
        """Return optional finding-level summaries."""


class CriticAgent(BaseAgent):
    """Analyze execution quality and emit advisory cancel/replace/replan guidance."""

    def __init__(self, name: str = "critic_agent", llm_advisor: CriticLLMAdvisor | None = None) -> None:
        self._task_critic = TaskGraphCritic()
        self._llm_advisor = llm_advisor
        self._llm_decision_validator = LLMDecisionValidator()
        self._last_llm_decision_validation_summary: dict[str, int] = {"accepted": 0, "rejected": 0}
        self._last_llm_decision_rejected_reasons: list[str] = []
        super().__init__(
            name=name,
            kind=AgentKind.CRITIC,
            write_permission=WritePermission(
                scopes=[GraphScope.TG, GraphScope.RUNTIME],
                allow_structural_write=False,
                allow_state_write=True,
                allow_event_emit=True,
            ),
        )

    def validate_input(self, agent_input: AgentInput) -> None:
        """Require at least TG context and critique input summaries."""

        super().validate_input(agent_input)
        if not any(ref.graph == GraphScope.TG for ref in agent_input.graph_refs):
            raise ValueError("critic input requires at least one TG ref")
        self._resolve_task_graph(agent_input)

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """Detect critique patterns and emit advisory decisions/replan requests."""

        task_graph = self._resolve_task_graph(agent_input)
        runtime_state = self._resolve_runtime_state(agent_input)
        context = self._resolve_critic_context(agent_input, runtime_state=runtime_state)

        findings: list[CriticFinding] = []
        recommendations: list[CriticRecommendation] = []

        findings.extend(self._detect_duplicate_tasks(task_graph=task_graph, context=context))
        findings.extend(self._detect_blocked_tasks(task_graph=task_graph, context=context))
        findings.extend(self._detect_failure_saturation(task_graph=task_graph, runtime_state=runtime_state, context=context))
        findings.extend(self._detect_low_value_paths(task_graph=task_graph, context=context))
        findings.extend(self._detect_invalidated_by_new_evidence(task_graph=task_graph, context=context))
        findings = self._apply_llm_review(findings=findings, context=context, runtime_state=runtime_state)

        recommendations.extend(self._build_cancel_recommendations(findings=findings))
        recommendations.extend(self._build_replacement_recommendations(findings=findings, task_graph=task_graph))
        recommendations.extend(self._build_score_adjustment_hints(findings=findings))

        replan_requests = self._build_replan_requests(
            findings=findings,
            task_graph=task_graph,
            context=context,
        )
        decisions = self._emit_decisions(recommendations=recommendations)
        state_deltas = self._build_advisory_state_deltas(recommendations=recommendations)

        logs = [
            f"critic emitted {len(findings)} finding(s)",
            f"critic emitted {len(recommendations)} recommendation(s)",
            f"critic emitted {len(replan_requests)} replan request(s)",
            "critic llm decision validation "
            f"accepted={self._last_llm_decision_validation_summary['accepted']} "
            f"rejected={self._last_llm_decision_validation_summary['rejected']}",
            "critic output is advisory only and does not modify KG fact state",
        ]
        logs.extend(f"critic llm decision rejected: {reason}" for reason in self._last_llm_decision_rejected_reasons)

        return AgentOutput(
            replan_requests=[request.to_agent_output_fragment() for request in replan_requests],
            decisions=[decision.to_agent_output_fragment() for decision in decisions],
            state_deltas=state_deltas,
            logs=logs,
        )

    # 中文注释：
    # Critic 的 LLM 只能补充归纳，不改变建议生成规则本身。
    def _apply_llm_review(
        self,
        *,
        findings: Sequence[CriticFinding],
        context: CriticContext,
        runtime_state: RuntimeState | None,
    ) -> list[CriticFinding]:
        if self._llm_advisor is None or not findings:
            self._last_llm_decision_validation_summary = {"accepted": 0, "rejected": 0}
            self._last_llm_decision_rejected_reasons = []
            return list(findings)
        reviews = self._llm_advisor.summarize_findings(
            findings=findings,
            context=context,
            runtime_state=runtime_state,
        )
        if not reviews:
            self._last_llm_decision_validation_summary = {"accepted": 0, "rejected": 0}
            self._last_llm_decision_rejected_reasons = []
            return list(findings)
        allowed_finding_ids = {finding.finding_id for finding in findings}
        allowed_task_ids = self._task_ids_from_findings(findings)
        reviews_by_id: dict[str, CriticLLMReview] = {}
        rejected_count = 0
        rejected_reasons: list[str] = []
        policy_context = context.critic_hints if isinstance(context.critic_hints, dict) else {}
        for review in reviews:
            decision = review.decision or self._critic_decision_from_review(review)
            validation = self._llm_decision_validator.validate_critic_decision(
                decision,
                allowed_finding_ids=allowed_finding_ids,
                policy_context=policy_context,
                runtime_summary=context.runtime_summary,
            )
            if not validation.accepted:
                rejected_count += 1
                rejected_reasons.append(validation.reason)
                continue
            proposal = None
            if review.replan_proposal is not None:
                proposal = self._validated_replan_proposal(
                    proposal=review.replan_proposal,
                    allowed_finding_ids=allowed_finding_ids,
                    allowed_task_ids=allowed_task_ids,
                    policy_context=policy_context,
                    runtime_summary=context.runtime_summary,
                )
                if proposal is None:
                    rejected_count += 1
                    rejected_reasons.append("critic replan proposal rejected")
                    continue
            reviews_by_id[review.finding_id] = review.model_copy(
                update={
                    "decision": decision,
                    "validation": validation,
                    "summary_override": validation.sanitized_payload.get("summary_override"),
                    "rationale_suffix": validation.sanitized_payload.get("rationale_suffix"),
                    "replan_hint": validation.sanitized_payload.get("replan_hint"),
                    "metadata": validation.sanitized_payload.get("metadata", {}),
                    "replan_proposal": proposal,
                }
            )
        if not reviews_by_id:
            self._last_llm_decision_validation_summary = {"accepted": 0, "rejected": rejected_count}
            self._last_llm_decision_rejected_reasons = rejected_reasons
            return list(findings)
        self._last_llm_decision_validation_summary = {
            "accepted": len(reviews_by_id),
            "rejected": rejected_count,
        }
        self._last_llm_decision_rejected_reasons = rejected_reasons
        updated: list[CriticFinding] = []
        for finding in findings:
            review = reviews_by_id.get(finding.finding_id)
            if review is None:
                updated.append(finding)
                continue
            rationale = finding.rationale
            if review.rationale_suffix:
                rationale = f"{rationale}; {review.rationale_suffix}"
            metadata = dict(finding.metadata)
            metadata["llm_review"] = review.model_dump(mode="json")
            if review.decision is not None:
                metadata["llm_decision"] = review.decision.model_dump(mode="json")
            if review.validation is not None:
                metadata["llm_decision_validation"] = review.validation.model_dump(mode="json")
            if review.replan_hint:
                metadata["llm_replan_hint"] = review.replan_hint
            if review.replan_proposal is not None:
                metadata["llm_replan_proposal"] = review.replan_proposal.model_dump(mode="json")
                metadata["runtime_metadata"] = {
                    "llm_replan_proposal": {
                        "adopted": True,
                        "finding_id": review.replan_proposal.finding_id,
                        "affected_task_ids": list(review.replan_proposal.affected_task_ids),
                        "requires_human_review": review.replan_proposal.requires_human_review,
                    }
                }
            updated.append(
                finding.model_copy(
                    update={
                        "summary": review.summary_override or finding.summary,
                        "rationale": rationale,
                        "metadata": metadata,
                    }
                )
            )
        return updated

    def _validated_replan_proposal(
        self,
        *,
        proposal: CriticLLMReplanProposal,
        allowed_finding_ids: set[str],
        allowed_task_ids: set[str],
        policy_context: dict[str, Any],
        runtime_summary: dict[str, Any],
    ) -> CriticLLMReplanProposal | None:
        decision = proposal.decision or self._critic_decision_from_replan_proposal(proposal)
        validation = self._llm_decision_validator.validate_critic_replan_proposal(
            decision,
            allowed_finding_ids=allowed_finding_ids,
            allowed_task_ids=allowed_task_ids,
            affected_task_ids=proposal.affected_task_ids,
            confidence=proposal.confidence,
            policy_context=policy_context,
            runtime_summary=runtime_summary,
        )
        if not validation.accepted:
            return None
        sanitized = validation.sanitized_payload
        return proposal.model_copy(
            update={
                "decision": decision,
                "validation": validation,
                "failure_summary": sanitized.get("failure_summary"),
                "replan_hint": sanitized.get("replan_hint") or proposal.replan_hint,
                "affected_task_ids": list(sanitized.get("affected_task_ids", [])),
                "confidence": float(sanitized.get("confidence", proposal.confidence)),
                "requires_human_review": bool(sanitized.get("requires_human_review", False)),
                "metadata": sanitized.get("metadata", {}),
            }
        )

    @staticmethod
    def _critic_decision_from_review(review: CriticLLMReview) -> LLMDecision:
        return LLMDecision(
            source=LLMDecisionSource.CRITIC,
            status=LLMDecisionStatus.ACCEPTED,
            decision_type="critic_finding_review",
            target_id=review.finding_id,
            target_kind="critic_finding",
            summary_override=review.summary_override,
            rationale_suffix=review.rationale_suffix,
            replan_hint=review.replan_hint,
            metadata=dict(review.metadata),
        )

    @staticmethod
    def _critic_decision_from_replan_proposal(proposal: CriticLLMReplanProposal) -> LLMDecision:
        return LLMDecision(
            source=LLMDecisionSource.CRITIC,
            status=LLMDecisionStatus.ACCEPTED,
            decision_type="critic_replan_proposal",
            target_id=proposal.finding_id,
            target_kind="critic_finding",
            summary_override=proposal.failure_summary,
            replan_hint=proposal.replan_hint,
            metadata={
                **dict(proposal.metadata),
                "affected_task_ids": list(proposal.affected_task_ids),
                "confidence": proposal.confidence,
                "requires_human_review": proposal.requires_human_review,
            },
        )

    def task_critic_bridge(
        self,
        *,
        task_graph: TaskGraph,
        context: CriticContext,
    ) -> dict[str, Any]:
        """Compatibility hook for the existing `TaskGraphCritic` / TG patcher path."""

        return self._task_critic.critique_task_graph(
            task_graph,
            context=TaskCriticContext(
                low_value_threshold=context.low_value_threshold,
                failure_threshold=context.failure_threshold,
                invalidated_ref_keys=set(context.invalidated_ref_keys),
            ),
        ).model_dump(mode="json")

    def _detect_duplicate_tasks(
        self,
        *,
        task_graph: TaskGraph,
        context: CriticContext,
    ) -> list[CriticFinding]:
        """Detect duplicated TG tasks with equivalent source action/type/bindings."""

        critique = self._task_critic.critique_task_graph(
            task_graph,
            context=TaskCriticContext(
                low_value_threshold=context.low_value_threshold,
                failure_threshold=context.failure_threshold,
                invalidated_ref_keys=set(),
            ),
        )
        findings: list[CriticFinding] = []
        for raw in critique.findings:
            if raw.kind != "duplicate_task":
                continue
            subject_refs = [
                GraphRef(graph=GraphScope.TG, ref_id=task_id, ref_type="Task")
                for task_id in critique.duplicate_task_ids
            ]
            findings.append(
                CriticFinding(
                    finding_type="duplicate_tasks",
                    severity=raw.severity,
                    subject_refs=subject_refs,
                    summary="duplicate task instances detected in TG",
                    rationale=raw.reason,
                    metadata={"recommendation": raw.recommendation},
                )
            )
            break
        return findings

    def _detect_blocked_tasks(
        self,
        *,
        task_graph: TaskGraph,
        context: CriticContext,
    ) -> list[CriticFinding]:
        """Detect permanently blocked tasks."""

        findings: list[CriticFinding] = []
        for task in self._task_nodes(task_graph):
            if task.status != TaskStatus.BLOCKED:
                continue
            permanently_blocked = bool(task.gate_ids) or task.reason in context.permanently_blocked_reasons
            if not permanently_blocked:
                upstream = task_graph.predecessors(task.id)
                permanently_blocked = any(
                    dep.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SUPERSEDED}
                    for dep in upstream
                )
            if permanently_blocked:
                findings.append(
                    CriticFinding(
                        finding_type="permanently_blocked_tasks",
                        severity="high",
                        subject_refs=[GraphRef(graph=GraphScope.TG, ref_id=task.id, ref_type="Task")],
                        summary=f"task {task.id} appears permanently blocked",
                        rationale=task.reason or "task remains blocked with no viable upstream recovery",
                        metadata={"gate_ids": sorted(task.gate_ids)},
                    )
                )
        return findings

    def _detect_failure_saturation(
        self,
        *,
        task_graph: TaskGraph,
        runtime_state: RuntimeState | None,
        context: CriticContext,
    ) -> list[CriticFinding]:
        """Detect repeated failures and failure saturation."""

        findings: list[CriticFinding] = []
        recent_outcomes = [
            outcome if isinstance(outcome, OutcomeCacheEntry) else OutcomeCacheEntry.model_validate(outcome)
            for outcome in context.recent_outcomes
        ]
        failure_counts: dict[str, int] = {}
        for outcome in recent_outcomes:
            if "fail" in outcome.outcome_type.lower():
                failure_counts[outcome.task_id] = failure_counts.get(outcome.task_id, 0) + 1
        if runtime_state is not None:
            for task_runtime in runtime_state.execution.tasks.values():
                if task_runtime.status in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.TIMED_OUT}:
                    failure_counts[task_runtime.task_id] = max(
                        failure_counts.get(task_runtime.task_id, 0),
                        task_runtime.attempt_count,
                    )

        for task in self._task_nodes(task_graph):
            if failure_counts.get(task.id, 0) >= context.failure_threshold or (
                task.status == TaskStatus.FAILED and task.attempt_count >= context.failure_threshold
            ):
                findings.append(
                    CriticFinding(
                        finding_type="repeated_failures",
                        severity="high",
                        subject_refs=[GraphRef(graph=GraphScope.TG, ref_id=task.id, ref_type="Task")],
                        summary=f"task {task.id} is failure-saturated",
                        rationale="task or its recent outcomes crossed the configured failure threshold",
                        metadata={"failure_count": failure_counts.get(task.id, task.attempt_count)},
                    )
                )
        return findings

    def _detect_low_value_paths(
        self,
        *,
        task_graph: TaskGraph,
        context: CriticContext,
    ) -> list[CriticFinding]:
        """Detect low-value TG paths/tasks using simple value scoring."""

        findings: list[CriticFinding] = []
        for task in self._task_nodes(task_graph):
            score = max(
                0.0,
                task.goal_relevance
                - (task.estimated_cost * 0.2)
                - (task.estimated_risk * 0.4)
                - (task.estimated_noise * 0.2),
            )
            if score < context.low_value_threshold:
                findings.append(
                    CriticFinding(
                        finding_type="low_value_paths",
                        severity="medium",
                        subject_refs=[GraphRef(graph=GraphScope.TG, ref_id=task.id, ref_type="Task")],
                        summary=f"task {task.id} sits on a low-value path",
                        rationale=f"value score {score:.3f} is below threshold {context.low_value_threshold:.3f}",
                        metadata={"score": score},
                    )
                )
        return findings

    def _detect_invalidated_by_new_evidence(
        self,
        *,
        task_graph: TaskGraph,
        context: CriticContext,
    ) -> list[CriticFinding]:
        """Detect tasks invalidated by new evidence or ref invalidation hints."""

        findings: list[CriticFinding] = []
        invalidated = set(context.invalidated_ref_keys)
        if not invalidated:
            return findings

        for task in self._task_nodes(task_graph):
            task_keys = self._task_ref_keys(task)
            if task_keys & invalidated:
                findings.append(
                    CriticFinding(
                        finding_type="invalidated_by_new_evidence",
                        severity="high",
                        subject_refs=[GraphRef(graph=GraphScope.TG, ref_id=task.id, ref_type="Task")],
                        summary=f"task {task.id} was invalidated by new evidence",
                        rationale="new evidence invalidated one or more task references or assumptions",
                        metadata={"invalidated_ref_keys": sorted(task_keys & invalidated)},
                    )
                )
        return findings

    def _build_replan_requests(
        self,
        *,
        findings: Sequence[CriticFinding],
        task_graph: TaskGraph,
        context: CriticContext,
    ) -> list[ReplanRequestRecord]:
        """Build replan requests from Critic findings."""

        requests: list[ReplanRequestRecord] = []
        for finding in findings:
            if finding.finding_type not in {
                "permanently_blocked_tasks",
                "repeated_failures",
                "invalidated_by_new_evidence",
                "low_value_paths",
            }:
                continue
            task_refs = [ref for ref in finding.subject_refs if ref.graph == GraphScope.TG]
            if not task_refs:
                continue
            task_id = task_refs[0].ref_id
            frontier = self._collect_replan_frontier(task_graph=task_graph, task_id=task_id)
            proposal = self._llm_replan_proposal_payload(finding)
            requests.append(
                ReplanRequestRecord(
                    source_agent=self.name,
                    summary=f"Replan requested for task {task_id}",
                    confidence=float(proposal.get("confidence", 0.8)) if proposal else 0.8,
                    refs=task_refs,
                    payload={
                        "frontier": frontier.model_dump(mode="json"),
                        "finding": finding.model_dump(mode="json"),
                        "llm_replan_proposal": proposal,
                        "runtime_metadata": {
                            "llm_replan_proposal": {
                                "adopted": bool(proposal),
                                "finding_id": finding.finding_id,
                                "affected_task_ids": proposal.get("affected_task_ids", []) if proposal else [],
                                "requires_human_review": bool(proposal.get("requires_human_review", False)) if proposal else False,
                            }
                        },
                    },
                    trigger_task_id=task_id,
                    reason=finding.summary,
                    affected_refs=list(task_refs),
                    severity=finding.severity,
                )
            )
        return requests

    def _build_cancel_recommendations(
        self,
        *,
        findings: Sequence[CriticFinding],
    ) -> list[CriticRecommendation]:
        """Build cancel suggestions from critic findings."""

        recommendations: list[CriticRecommendation] = []
        for finding in findings:
            if finding.finding_type not in {"duplicate_tasks", "permanently_blocked_tasks", "low_value_paths"}:
                continue
            proposal = self._llm_replan_proposal_payload(finding)
            recommendations.append(
                CriticRecommendation(
                    recommendation_type="cancel_suggestion",
                    action="suggest_cancel",
                    priority=0.9 if finding.severity == "high" else 0.6,
                    subject_refs=list(finding.subject_refs),
                    summary=f"Cancel suggestion for {finding.finding_type}",
                    rationale=self._rationale_with_replan_proposal(finding.rationale, proposal),
                    patch={"suggested_status": TaskStatus.CANCELLED.value},
                    metadata={
                        "finding_id": finding.finding_id,
                        **self._llm_finding_metadata(finding),
                        "llm_replan_proposal": proposal,
                    },
                )
            )
        return recommendations

    def _build_replacement_recommendations(
        self,
        *,
        findings: Sequence[CriticFinding],
        task_graph: TaskGraph,
    ) -> list[CriticRecommendation]:
        """Build replacement suggestions from critic findings."""

        recommendations: list[CriticRecommendation] = []
        for finding in findings:
            if finding.finding_type not in {"duplicate_tasks", "repeated_failures", "invalidated_by_new_evidence"}:
                continue
            task_refs = [ref for ref in finding.subject_refs if ref.graph == GraphScope.TG]
            if not task_refs:
                continue
            task_id = task_refs[0].ref_id
            frontier = self._collect_replan_frontier(task_graph=task_graph, task_id=task_id)
            proposal = self._llm_replan_proposal_payload(finding)
            recommendations.append(
                CriticRecommendation(
                    recommendation_type="replacement_suggestion",
                    action="suggest_replace",
                    priority=0.85,
                    subject_refs=task_refs,
                    summary=f"Replacement suggestion for task {task_id}",
                    rationale=self._rationale_with_replan_proposal(finding.rationale, proposal),
                    patch={
                        "suggested_status": TaskStatus.SUPERSEDED.value,
                        "replacement_frontier": frontier.model_dump(mode="json"),
                    },
                    metadata={
                        "finding_id": finding.finding_id,
                        **self._llm_finding_metadata(finding),
                        "llm_replan_proposal": proposal,
                    },
                )
            )
        return recommendations

    def _build_score_adjustment_hints(
        self,
        *,
        findings: Sequence[CriticFinding],
    ) -> list[CriticRecommendation]:
        """Build score adjustment hints for low-value or failure-saturated branches."""

        recommendations: list[CriticRecommendation] = []
        for finding in findings:
            if finding.finding_type not in {"low_value_paths", "repeated_failures"}:
                continue
            proposal = self._llm_replan_proposal_payload(finding)
            recommendations.append(
                CriticRecommendation(
                    recommendation_type="score_adjustment_hint",
                    action="adjust_score",
                    priority=0.5,
                    subject_refs=list(finding.subject_refs),
                    summary=f"Score adjustment hint for {finding.finding_type}",
                    rationale=self._rationale_with_replan_proposal(finding.rationale, proposal),
                    patch={"score_delta": -0.2 if finding.finding_type == "low_value_paths" else -0.35},
                    metadata={
                        "finding_id": finding.finding_id,
                        **self._llm_finding_metadata(finding),
                        "llm_replan_proposal": proposal,
                    },
                )
            )
        return recommendations

    def _emit_decisions(
        self,
        *,
        recommendations: Sequence[CriticRecommendation],
    ) -> list[DecisionRecord]:
        """Emit Critic recommendations as structured decisions."""

        decisions: list[DecisionRecord] = []
        for recommendation in recommendations:
            decisions.append(
                DecisionRecord(
                    source_agent=self.name,
                    summary=recommendation.summary,
                    confidence=recommendation.priority,
                    refs=list(recommendation.subject_refs),
                    payload={"recommendation": recommendation.model_dump(mode="json")},
                    decision_type=recommendation.recommendation_type,
                    score=recommendation.priority,
                    target_refs=list(recommendation.subject_refs),
                    rationale=recommendation.rationale,
                )
            )
        return decisions

    def _build_advisory_state_deltas(
        self,
        *,
        recommendations: Sequence[CriticRecommendation],
    ) -> list[dict[str, Any]]:
        """Build advisory TG/runtime patches from Critic recommendations."""

        deltas: list[dict[str, Any]] = []
        for recommendation in recommendations:
            for ref in recommendation.subject_refs:
                if ref.graph not in {GraphScope.TG, GraphScope.RUNTIME}:
                    continue
                deltas.append(
                    StateDeltaRecord(
                        source_agent=self.name,
                        summary=recommendation.summary,
                        graph_scope=ref.graph,
                        delta_type="suggestion_patch",
                        target_ref=ref,
                        patch={
                            "advisory": True,
                            "recommendation": recommendation.model_dump(mode="json"),
                        },
                        payload={"patch_kind": "critic_suggestion"},
                    ).to_agent_output_fragment()
                )
        return deltas

    @staticmethod
    def _task_nodes(task_graph: TaskGraph) -> list[BaseTaskNode]:
        """Return TG task nodes in stable order."""

        return sorted(
            (node for node in task_graph.list_nodes() if isinstance(node, BaseTaskNode)),
            key=lambda item: item.id,
        )

    def _collect_replan_frontier(self, *, task_graph: TaskGraph, task_id: str) -> ReplanFrontier:
        """Bridge to the existing TG critic/frontier collector."""

        return self._task_critic.collect_replan_frontier(task_graph, task_id)

    @staticmethod
    def _task_ids_from_findings(findings: Sequence[CriticFinding]) -> set[str]:
        return {
            ref.ref_id
            for finding in findings
            for ref in finding.subject_refs
            if ref.graph == GraphScope.TG
        }

    @staticmethod
    def _llm_replan_proposal_payload(finding: CriticFinding) -> dict[str, Any] | None:
        payload = finding.metadata.get("llm_replan_proposal")
        return dict(payload) if isinstance(payload, dict) else None

    @staticmethod
    def _llm_finding_metadata(finding: CriticFinding) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in ("llm_decision", "llm_decision_validation", "llm_review"):
            value = finding.metadata.get(key)
            if value is not None:
                metadata[key] = value
        return metadata

    @staticmethod
    def _rationale_with_replan_proposal(rationale: str, proposal: dict[str, Any] | None) -> str:
        if not proposal:
            return rationale
        hint = proposal.get("replan_hint")
        if not isinstance(hint, str) or not hint:
            return rationale
        return f"{rationale}; LLM replan hint: {hint}"

    def _resolve_task_graph(self, agent_input: AgentInput) -> TaskGraph:
        """Resolve TG snapshot from raw payload."""

        raw_graph = agent_input.raw_payload.get("tg_graph") or agent_input.raw_payload.get("task_graph")
        if isinstance(raw_graph, TaskGraph):
            return raw_graph
        if isinstance(raw_graph, dict):
            return TaskGraph.from_dict(raw_graph)
        raise ValueError("critic input requires raw_payload.tg_graph or equivalent TaskGraph snapshot")

    def _resolve_runtime_state(self, agent_input: AgentInput) -> RuntimeState | None:
        """Resolve optional runtime state snapshot."""

        raw_state = agent_input.raw_payload.get("runtime_state")
        if isinstance(raw_state, RuntimeState):
            return raw_state
        if isinstance(raw_state, dict) and raw_state:
            return RuntimeState.model_validate(raw_state)
        return None

    def _resolve_critic_context(
        self,
        agent_input: AgentInput,
        *,
        runtime_state: RuntimeState | None,
    ) -> CriticContext:
        """Normalize Critic context from raw payload and runtime state."""

        runtime_summary = self._coerce_mapping(agent_input.raw_payload.get("runtime_summary"))
        if runtime_state is not None and not runtime_summary:
            runtime_summary = {
                "operation_status": runtime_state.operation_status.value,
                "failed_task_ids": [
                    task_id
                    for task_id, task in runtime_state.execution.tasks.items()
                    if task.status in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.TIMED_OUT}
                ],
            }
        payload = {
            "runtime_summary": runtime_summary,
            "recent_outcomes": self._coerce_list_of_mappings(agent_input.raw_payload.get("recent_outcomes")),
            **self._coerce_mapping(agent_input.raw_payload.get("critic_context")),
        }
        return CriticContext.model_validate(payload)

    @staticmethod
    def _task_ref_keys(task: BaseTaskNode) -> set[str]:
        """Return stable ref-key variants used for invalidation matching."""

        keys: set[str] = set()
        for ref in [*task.target_refs, *task.source_refs]:
            keys.add(ref.key())
            if ref.ref_type:
                keys.add(f"{ref.graph}:{ref.ref_type}:{ref.ref_id}")
        return keys

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        """Return a shallow mapping copy or an empty mapping."""

        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_list_of_mappings(value: Any) -> list[dict[str, Any]]:
        """Normalize scalar/list payloads into a list of mappings."""

        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [dict(item) for item in items if isinstance(item, dict)]


__all__ = [
    "CriticAgent",
    "CriticContext",
    "CriticFinding",
    "CriticLLMAdvisor",
    "CriticLLMReplanProposal",
    "CriticLLMReview",
    "CriticRecommendation",
]
