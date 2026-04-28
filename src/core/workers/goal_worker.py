"""Worker responsible for goal-condition validation."""

from __future__ import annotations

from typing import Any

from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    AgentTaskResult,
    CheckpointHint,
    CriticSignal,
    CriticSignalSeverity,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ObservationRecord,
    ProjectionRequest,
    ProjectionRequestKind,
    ReplanHint,
    ReplanScope,
    RuntimeBudgetDelta,
    RuntimeControlRequest,
    RuntimeControlType,
)
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorker
from src.core.workers.goal_validator import GoalEvaluation, GoalValidator, MetadataGoalValidator
from src.core.workers.goal_command_validator import CommandGoalValidator


class GoalWorker(BaseWorker):
    """Worker that evaluates whether a goal condition is satisfied."""

    agent_role = AgentRole.GOAL_WORKER
    supported_task_types = frozenset({TaskType.GOAL_CONDITION_VALIDATION})
    capabilities = frozenset({"validate_goal", "emit_replan_hints", "propose_goal_assertions"})

    def __init__(self, *, validator: GoalValidator | None = None) -> None:
        self._validator = validator or CommandGoalValidator(fallback=MetadataGoalValidator())

    def default_intent(self, task_type: TaskType) -> AgentTaskIntent:
        return AgentTaskIntent.VALIDATE_GOAL

    def handle_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Validate a goal condition and emit structured follow-up guidance."""

        evaluation = self._validator.evaluate(request)
        if evaluation.blocked:
            return self._result(
                request,
                status=AgentResultStatus.BLOCKED,
                summary=evaluation.failure_reason or f"goal validation blocked for task {request.context.task_id}",
                error_message=evaluation.failure_reason,
                outcome_payload={"goal_satisfied": False, "goal_evaluation": self._goal_evaluation_payload(evaluation)},
            )
        if evaluation.failure_reason and not evaluation.satisfied and not evaluation.missing_requirements:
            return self._result(
                request,
                status=AgentResultStatus.FAILED,
                summary=evaluation.failure_reason,
                error_message=evaluation.failure_reason,
                outcome_payload={"goal_satisfied": False, "goal_evaluation": self._goal_evaluation_payload(evaluation)},
            )
        goal_satisfied = bool(evaluation.satisfied)
        goal_evaluation = self._goal_evaluation_payload(evaluation)
        primary_ref = request.target_refs[0] if request.target_refs else None
        observation = ObservationRecord(
            category="goal",
            summary=f"Goal worker evaluated {request.task_label}",
            confidence=float(evaluation.confidence),
            refs=list(request.target_refs),
            payload={"goal_satisfied": goal_satisfied, "goal_evaluation": goal_evaluation},
        )
        evidence_payload_ref = f"runtime://outcomes/{request.context.task_id}/goal"
        if evaluation.supporting_evidence:
            first_ref = evaluation.supporting_evidence[0].get("payload_ref")
            if first_ref is not None:
                evidence_payload_ref = str(first_ref)
        evidence = EvidenceArtifact(
            kind="goal_validation",
            summary=f"Goal validation evidence for task {request.context.task_id}",
            payload_ref=evidence_payload_ref,
            refs=list(request.target_refs),
            metadata={"supporting_evidence": list(evaluation.supporting_evidence)},
        )
        fact_write_requests = self._build_fact_writes(
            request=request,
            primary_ref=primary_ref,
            goal_evaluation=goal_evaluation,
            goal_satisfied=goal_satisfied,
            evidence_id=evidence.evidence_id,
            confidence=observation.confidence,
        )
        projection_requests = [
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_LOCAL_FRONTIER,
                source_task_id=request.context.task_id,
                reason="goal validation may change the local planning frontier",
                target_refs=list(request.target_refs),
                metadata={"validated_ref_ids": list(evaluation.validated_ref_ids)},
            )
        ]
        runtime_requests = [
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=request.context.task_id,
                budget_delta=RuntimeBudgetDelta(operations=1),
                reason="goal validation consumes one operation budget unit",
            )
        ]
        if goal_satisfied:
            checkpoint_hints = [
                CheckpointHint(
                    source_task_id=request.context.task_id,
                    summary=f"Goal satisfied at task {request.context.task_id}",
                    created_after_tasks=[request.context.task_id],
                )
            ]
            return self._result(
                request,
                status=AgentResultStatus.SUCCEEDED,
                summary=f"goal satisfied for task {request.context.task_id}",
                observations=[observation],
                evidence=[evidence],
                fact_write_requests=fact_write_requests,
                projection_requests=projection_requests,
                runtime_requests=runtime_requests,
                checkpoint_hints=checkpoint_hints,
                outcome_payload={"goal_satisfied": True, "goal_evaluation": goal_evaluation},
            )

        critic_signal = CriticSignal(
            source_task_id=request.context.task_id,
            kind="goal_unsatisfied",
            severity=CriticSignalSeverity.HIGH,
            reason="goal validation failed on the current local branch",
            task_ids=[request.context.task_id],
            invalidated_ref_keys=[ref.key() for ref in request.target_refs],
        )
        replan_hint = ReplanHint(
            source_task_id=request.context.task_id,
            scope=ReplanScope.LOCAL,
            reason="goal remains unsatisfied after validation",
            task_ids=[request.context.task_id],
            invalidated_ref_keys=[ref.key() for ref in request.target_refs],
            metadata={"missing_requirements": list(evaluation.missing_requirements)},
        )
        runtime_requests.append(
            RuntimeControlRequest(
                request_type=RuntimeControlType.REQUEST_REPLAN,
                source_task_id=request.context.task_id,
                reason="goal worker determined that local replanning is required",
                metadata={"scope": ReplanScope.LOCAL.value},
            )
        )
        return self._result(
            request,
            status=AgentResultStatus.NEEDS_REPLAN,
            summary=f"goal remains unsatisfied for task {request.context.task_id}",
            observations=[observation],
            evidence=[evidence],
            fact_write_requests=fact_write_requests,
            projection_requests=projection_requests,
            runtime_requests=runtime_requests,
            critic_signals=[critic_signal],
            replan_hints=[replan_hint],
            outcome_payload={"goal_satisfied": False, "goal_evaluation": goal_evaluation},
        )

    @staticmethod
    def _goal_evaluation_payload(evaluation: GoalEvaluation) -> dict[str, Any]:
        return {
            "satisfied": evaluation.satisfied,
            "missing_requirements": list(evaluation.missing_requirements),
            "validated_ref_ids": list(evaluation.validated_ref_ids),
            "supporting_evidence": [dict(item) for item in evaluation.supporting_evidence],
            "confidence": float(evaluation.confidence),
            "blocked": bool(evaluation.blocked),
            "failure_reason": evaluation.failure_reason,
            **dict(evaluation.metadata),
        }

    @staticmethod
    def _build_fact_writes(
        *,
        request: AgentTaskRequest,
        primary_ref: Any,
        goal_evaluation: dict[str, Any],
        goal_satisfied: bool,
        evidence_id: str,
        confidence: float,
    ) -> list[FactWriteRequest]:
        requests: list[FactWriteRequest] = []
        if primary_ref is None:
            return requests

        graph_ref_cls = primary_ref.__class__
        goal_ref = (
            primary_ref
            if (primary_ref.ref_type or "").lower() == "goal"
            else graph_ref_cls(graph="kg", ref_id=f"goal::{primary_ref.ref_id}", ref_type="Goal")
        )
        evidence_ref = graph_ref_cls(graph="kg", ref_id=evidence_id, ref_type="Evidence")
        requests.append(
            FactWriteRequest(
                kind=FactWriteKind.ENTITY_UPSERT,
                source_task_id=request.context.task_id,
                subject_ref=goal_ref,
                attributes={
                    "goal_satisfied": goal_satisfied,
                    "missing_requirements": list(goal_evaluation.get("missing_requirements", [])),
                    "validated_ref_ids": list(goal_evaluation.get("validated_ref_ids", [])),
                    "supporting_evidence": list(goal_evaluation.get("supporting_evidence", [])),
                },
                confidence=confidence,
                evidence_ids=[evidence_id],
                summary=f"Goal state for {goal_ref.ref_id}",
            )
        )
        requests.append(
            FactWriteRequest(
                kind=FactWriteKind.RELATION_UPSERT,
                source_task_id=request.context.task_id,
                subject_ref=goal_ref,
                relation_type="SUPPORTED_BY",
                object_ref=evidence_ref,
                attributes={"validation": "goal"},
                confidence=confidence,
                evidence_ids=[evidence_id],
                summary=f"Goal evidence supports {goal_ref.ref_id}",
            )
        )
        if goal_ref.ref_id != primary_ref.ref_id:
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=goal_ref,
                    relation_type="TARGETS",
                    object_ref=primary_ref,
                    attributes={"goal_satisfied": goal_satisfied},
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Goal {goal_ref.ref_id} targets {primary_ref.ref_id}",
                )
            )

        for item in goal_evaluation.get("missing_requirements", []):
            requirement = str(item).strip()
            if not requirement:
                continue
            finding_ref = graph_ref_cls(
                graph="kg",
                ref_id=f"missing-requirement::{goal_ref.ref_id}::{requirement}",
                ref_type="Finding",
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=finding_ref,
                    attributes={
                        "finding_kind": "missing_requirement",
                        "requirement": requirement,
                        "goal_ref": goal_ref.ref_id,
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Missing requirement {requirement}",
                )
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=finding_ref,
                    relation_type="RELATED_TO",
                    object_ref=goal_ref,
                    attributes={"requirement": requirement},
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Missing requirement {requirement} relates to {goal_ref.ref_id}",
                )
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=finding_ref,
                    relation_type="SUPPORTED_BY",
                    object_ref=evidence_ref,
                    attributes={"validation": "goal"},
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Evidence supports missing requirement {requirement}",
                )
            )

        return requests


__all__ = ["GoalWorker"]
