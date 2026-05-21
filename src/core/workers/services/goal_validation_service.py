"""Domain service for goal validation worker behavior."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import AgentInput
from src.core.models.events import (
    AgentResultStatus,
    AgentTaskRequest,
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
from src.core.models.ag import GraphRef as EventGraphRef
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.goal_command_validator import CommandGoalValidator
from src.core.workers.goal_validator import GoalEvaluation, GoalValidator, MetadataGoalValidator
from src.core.workers.services.result_builders import WorkerDomainResult


class GoalValidationRequest(BaseModel):
    """Domain input consumed by `GoalValidationService`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    operation_id: str
    task_id: str
    task_type: str
    task_label: str
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    target_refs: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    context_metadata: dict[str, Any] = Field(default_factory=dict)
    resource_keys: list[str] = Field(default_factory=list)

    @classmethod
    def from_task_spec(
        cls,
        *,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> "GoalValidationRequest":
        raw = dict(agent_input.raw_payload)
        metadata = dict(raw.get("metadata", {}))
        constraints = dict(task_spec.constraints)
        if not any(key in metadata for key in {"goal_validator_output", "goal_evaluation", "goal_validator_command"}):
            metadata.update(cls._default_agent_goal_metadata(task_spec))
        return cls(
            operation_id=agent_input.context.operation_id,
            task_id=task_spec.task_id,
            task_type=task_spec.task_type,
            task_label=str(raw.get("task_label") or task_spec.task_type),
            input_bindings=dict(task_spec.input_bindings),
            target_refs=list(task_spec.target_refs),
            metadata=metadata,
            constraints=constraints,
            context_metadata=dict(agent_input.context.extra),
            resource_keys=list(task_spec.resource_keys),
        )

    @classmethod
    def from_legacy_request(cls, request: AgentTaskRequest) -> "GoalValidationRequest":
        return cls(
            operation_id=request.context.operation_id,
            task_id=request.context.task_id,
            task_type=request.context.task_type.value,
            task_label=request.task_label,
            input_bindings=dict(request.input_bindings),
            target_refs=list(request.target_refs),
            metadata=dict(request.metadata),
            constraints={},
            context_metadata=dict(request.context.metadata),
            resource_keys=sorted(request.context.resource_keys),
        )

    @classmethod
    def _default_agent_goal_metadata(cls, task_spec: WorkerTaskSpec) -> dict[str, Any]:
        if task_spec.task_type in {"goal_reached_verification"}:
            goal_ref = (
                task_spec.input_bindings.get("goal")
                or task_spec.input_bindings.get("goal_id")
                or cls._primary_ref_id(task_spec.target_refs, preferred_type="Goal")
                or "unknown-goal"
            )
            target_state = (
                task_spec.input_bindings.get("target_state")
                or task_spec.constraints.get("expected_state")
                or "reached"
            )
            goal_satisfied = bool(task_spec.constraints.get("goal_satisfied", True))
            return {
                "goal_validator_output": {
                    "satisfied": goal_satisfied,
                    "validated_ref_ids": cls._ref_ids(task_spec.target_refs),
                    "confidence": 0.84 if goal_satisfied else 0.76,
                    "goal_ref": goal_ref,
                    "target_state": target_state,
                    "source": "goal_reached_verification",
                }
            }
        if task_spec.task_type == "target_state_confirmation":
            target_ref = (
                task_spec.input_bindings.get("target")
                or task_spec.input_bindings.get("target_id")
                or cls._primary_ref_id(task_spec.target_refs, preferred_type="Target")
                or cls._primary_ref_id(task_spec.target_refs, preferred_type="State")
                or "unknown-target"
            )
            expected_state = (
                task_spec.input_bindings.get("expected_state")
                or task_spec.constraints.get("expected_state")
                or "confirmed"
            )
            observed_state = (
                task_spec.input_bindings.get("observed_state")
                or task_spec.constraints.get("observed_state")
                or expected_state
            )
            goal_satisfied = observed_state == expected_state
            return {
                "goal_validator_output": {
                    "satisfied": goal_satisfied,
                    "validated_ref_ids": cls._ref_ids(task_spec.target_refs),
                    "confidence": 0.82 if goal_satisfied else 0.7,
                    "target_ref": target_ref,
                    "expected_state": expected_state,
                    "observed_state": observed_state,
                    "source": "target_state_confirmation",
                }
            }
        return {}

    @staticmethod
    def _primary_ref_id(refs: list[Any], *, preferred_type: str) -> str | None:
        for ref in refs:
            if (getattr(ref, "ref_type", None) or "").lower() == preferred_type.lower():
                return str(getattr(ref, "ref_id"))
        return str(getattr(refs[0], "ref_id")) if refs else None

    @staticmethod
    def _ref_ids(refs: list[Any]) -> list[str]:
        return [str(getattr(ref, "ref_id")) for ref in refs if getattr(ref, "ref_id", None)]


class GoalValidationService:
    """Evaluate goal state and build domain-level worker results."""

    def __init__(self, validator: GoalValidator | None = None) -> None:
        self._validator = validator or CommandGoalValidator(fallback=MetadataGoalValidator())

    def validate(self, request: GoalValidationRequest) -> WorkerDomainResult:
        evaluation = self._validator.evaluate(request)
        goal_evaluation = self._goal_evaluation_payload(evaluation)

        if evaluation.blocked:
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.BLOCKED.value,
                summary=evaluation.failure_reason or f"goal validation blocked for task {request.task_id}",
                confidence=float(evaluation.confidence),
                raw_payload={
                    "status": AgentResultStatus.BLOCKED.value,
                    "goal_satisfied": False,
                    "goal_evaluation": goal_evaluation,
                    "error_message": evaluation.failure_reason,
                },
            )
        if evaluation.failure_reason and not evaluation.satisfied and not evaluation.missing_requirements:
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.FAILED.value,
                summary=evaluation.failure_reason,
                confidence=float(evaluation.confidence),
                raw_payload={
                    "status": AgentResultStatus.FAILED.value,
                    "goal_satisfied": False,
                    "goal_evaluation": goal_evaluation,
                    "error_message": evaluation.failure_reason,
                },
            )

        goal_satisfied = bool(evaluation.satisfied)
        event_refs = [self._event_ref(ref) for ref in request.target_refs]
        primary_ref = event_refs[0] if event_refs else None
        observation = ObservationRecord(
            category="goal",
            summary=f"Goal worker evaluated {request.task_label}",
            confidence=float(evaluation.confidence),
            refs=event_refs,
            payload={"goal_satisfied": goal_satisfied, "goal_evaluation": goal_evaluation},
        )
        evidence_payload_ref = f"runtime://outcomes/{request.task_id}/goal"
        if evaluation.supporting_evidence:
            first_ref = evaluation.supporting_evidence[0].get("payload_ref")
            if first_ref is not None:
                evidence_payload_ref = str(first_ref)
        evidence = EvidenceArtifact(
            kind="goal_validation",
            summary=f"Goal validation evidence for task {request.task_id}",
            payload_ref=evidence_payload_ref,
            refs=event_refs,
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
                source_task_id=request.task_id,
                reason="goal validation may change the local planning frontier",
                target_refs=event_refs,
                metadata={"validated_ref_ids": list(evaluation.validated_ref_ids)},
            )
        ]
        runtime_requests = [
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=request.task_id,
                budget_delta=RuntimeBudgetDelta(operations=1),
                reason="goal validation consumes one operation budget unit",
            )
        ]
        checkpoint_hints: list[CheckpointHint] = []
        critic_signals: list[CriticSignal] = []
        replan_hints: list[ReplanHint] = []

        if goal_satisfied:
            status = AgentResultStatus.SUCCEEDED
            summary = f"goal satisfied for task {request.task_id}"
            checkpoint_hints = [
                CheckpointHint(
                    source_task_id=request.task_id,
                    summary=f"Goal satisfied at task {request.task_id}",
                    created_after_tasks=[request.task_id],
                )
            ]
        else:
            status = AgentResultStatus.NEEDS_REPLAN
            summary = f"goal remains unsatisfied for task {request.task_id}"
            critic_signals = [
                CriticSignal(
                    source_task_id=request.task_id,
                    kind="goal_unsatisfied",
                    severity=CriticSignalSeverity.HIGH,
                    reason="goal validation failed on the current local branch",
                    task_ids=[request.task_id],
                    invalidated_ref_keys=[self._ref_key(ref) for ref in event_refs],
                )
            ]
            replan_hints = [
                ReplanHint(
                    source_task_id=request.task_id,
                    scope=ReplanScope.LOCAL,
                    reason="goal remains unsatisfied after validation",
                    task_ids=[request.task_id],
                    invalidated_ref_keys=[self._ref_key(ref) for ref in event_refs],
                    metadata={"missing_requirements": list(evaluation.missing_requirements)},
                )
            ]
            runtime_requests.append(
                RuntimeControlRequest(
                    request_type=RuntimeControlType.REQUEST_REPLAN,
                    source_task_id=request.task_id,
                    reason="goal worker determined that local replanning is required",
                    metadata={"scope": ReplanScope.LOCAL.value},
                )
            )

        raw_payload = {
            "status": status.value,
            "goal_satisfied": goal_satisfied,
            "goal_evaluation": goal_evaluation,
            "fact_write_requests": [item.model_dump(mode="json") for item in fact_write_requests],
            "projection_requests": [item.model_dump(mode="json") for item in projection_requests],
            "runtime_requests": [item.model_dump(mode="json") for item in runtime_requests],
            "checkpoint_hints": [item.model_dump(mode="json") for item in checkpoint_hints],
            "critic_signals": [item.model_dump(mode="json") for item in critic_signals],
            "replan_hints": [item.model_dump(mode="json") for item in replan_hints],
        }
        return WorkerDomainResult(
            success=status == AgentResultStatus.SUCCEEDED,
            status=status.value,
            summary=summary,
            confidence=float(evaluation.confidence),
            observations=[observation.model_dump(mode="json")],
            evidence=[evidence.model_dump(mode="json")],
            fact_write_requests=raw_payload["fact_write_requests"],
            projection_requests=raw_payload["projection_requests"],
            runtime_requests=raw_payload["runtime_requests"],
            critic_signals=raw_payload["critic_signals"],
            replan_hints=raw_payload["replan_hints"],
            raw_payload=raw_payload,
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

    @classmethod
    def _build_fact_writes(
        cls,
        *,
        request: GoalValidationRequest,
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
            if (getattr(primary_ref, "ref_type", None) or "").lower() == "goal"
            else graph_ref_cls(graph="kg", ref_id=f"goal::{primary_ref.ref_id}", ref_type="Goal")
        )
        evidence_ref = graph_ref_cls(graph="kg", ref_id=evidence_id, ref_type="Evidence")
        requests.append(
            FactWriteRequest(
                kind=FactWriteKind.ENTITY_UPSERT,
                source_task_id=request.task_id,
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
                source_task_id=request.task_id,
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
                    source_task_id=request.task_id,
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
                    source_task_id=request.task_id,
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
                    source_task_id=request.task_id,
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
                    source_task_id=request.task_id,
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

    @staticmethod
    def _ref_key(ref: Any) -> str:
        key = getattr(ref, "key", None)
        if callable(key):
            return str(key())
        graph = getattr(ref, "graph", "")
        graph_value = getattr(graph, "value", graph)
        return f"{graph_value}:{getattr(ref, 'ref_id', '')}"

    @staticmethod
    def _event_ref(ref: Any) -> EventGraphRef:
        if isinstance(ref, EventGraphRef):
            return ref
        graph = getattr(ref, "graph", "kg")
        graph_value = getattr(graph, "value", graph)
        return EventGraphRef(
            graph=str(graph_value),
            ref_id=str(getattr(ref, "ref_id")),
            ref_type=getattr(ref, "ref_type", None),
            label=getattr(ref, "label", None),
        )


__all__ = ["GoalValidationRequest", "GoalValidationService"]
