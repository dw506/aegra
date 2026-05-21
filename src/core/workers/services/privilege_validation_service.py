"""Domain service for privilege validation worker behavior."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import AgentInput
from src.core.models.ag import GraphRef as EventGraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentTaskRequest,
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
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.services.result_builders import WorkerDomainResult


class PrivilegeValidationRequest(BaseModel):
    """Domain input consumed by `PrivilegeValidationService`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    operation_id: str
    task_id: str
    task_type: str
    task_label: str
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    target_refs: list[Any] = Field(default_factory=list)
    source_refs: list[Any] = Field(default_factory=list)
    expected_output_refs: list[Any] = Field(default_factory=list)
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
    ) -> "PrivilegeValidationRequest":
        raw = dict(agent_input.raw_payload)
        metadata = dict(raw.get("metadata", {}))
        if "privilege_validation" not in metadata:
            metadata.update(cls._default_agent_privilege_metadata(task_spec))
        return cls(
            operation_id=agent_input.context.operation_id,
            task_id=task_spec.task_id,
            task_type=task_spec.task_type,
            task_label=str(raw.get("task_label") or task_spec.task_type),
            input_bindings=dict(task_spec.input_bindings),
            target_refs=list(task_spec.target_refs),
            source_refs=list(raw.get("source_refs", [])),
            expected_output_refs=list(raw.get("expected_output_refs", [])),
            metadata=metadata,
            constraints=dict(task_spec.constraints),
            context_metadata=dict(agent_input.context.extra),
            resource_keys=list(task_spec.resource_keys),
        )

    @classmethod
    def from_legacy_request(cls, request: AgentTaskRequest) -> "PrivilegeValidationRequest":
        return cls(
            operation_id=request.context.operation_id,
            task_id=request.context.task_id,
            task_type=request.context.task_type.value,
            task_label=request.task_label,
            input_bindings=dict(request.input_bindings),
            target_refs=list(request.target_refs),
            source_refs=list(request.source_refs),
            expected_output_refs=list(request.expected_output_refs),
            metadata=dict(request.metadata),
            constraints={},
            context_metadata=dict(request.context.metadata),
            resource_keys=sorted(request.context.resource_keys),
        )

    @classmethod
    def _default_agent_privilege_metadata(cls, task_spec: WorkerTaskSpec) -> dict[str, Any]:
        principal = (
            task_spec.input_bindings.get("principal")
            or task_spec.input_bindings.get("identity")
            or cls._primary_ref_id(task_spec.target_refs, preferred_type="Identity")
            or "unknown-principal"
        )
        if task_spec.task_type == "privilege_escalation_verification":
            target_state = (
                task_spec.input_bindings.get("target_privilege_state")
                or task_spec.constraints.get("expected_state")
                or "elevated"
            )
            escalation_path = (
                task_spec.input_bindings.get("escalation_path")
                or task_spec.constraints.get("path")
                or "unknown-path"
            )
            return {
                "privilege_validation": {
                    "validated": bool(task_spec.constraints.get("verified", True)),
                    "required_level": str(target_state),
                    "principal": principal,
                    "target_privilege_state": target_state,
                    "escalation_path": escalation_path,
                    "source": "privilege_escalation_verification",
                }
            }
        privilege_level = (
            task_spec.input_bindings.get("privilege_level")
            or task_spec.constraints.get("required_privilege")
            or "unknown"
        )
        return {
            "privilege_validation": {
                "validated": bool(task_spec.constraints.get("privilege_validated", True)),
                "required_level": str(privilege_level),
                "principal": principal,
                "session_id": task_spec.constraints.get("session_id") or task_spec.input_bindings.get("session_id"),
                "source": "privilege_validation",
            }
        }

    @staticmethod
    def _primary_ref_id(refs: list[Any], *, preferred_type: str) -> str | None:
        for ref in refs:
            if (getattr(ref, "ref_type", None) or "").lower() == preferred_type.lower():
                return str(getattr(ref, "ref_id"))
        return str(getattr(refs[0], "ref_id")) if refs else None


class PrivilegeValidationService:
    """Validate privilege state and build domain-level worker results."""

    def validate(self, request: PrivilegeValidationRequest) -> WorkerDomainResult:
        privilege_validation = self._privilege_validation_view(request)
        confidence = float(request.metadata.get("confidence", privilege_validation.get("confidence", 0.8)))
        event_refs = [self._event_ref(ref) for ref in request.target_refs]
        primary_ref = event_refs[0] if event_refs else None

        if privilege_validation.get("blocked"):
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.BLOCKED.value,
                summary=privilege_validation.get("failure_reason") or f"privilege validation blocked for task {request.task_id}",
                confidence=confidence,
                raw_payload={
                    "status": AgentResultStatus.BLOCKED.value,
                    "validated": False,
                    "privilege_validation": privilege_validation,
                    "error_message": privilege_validation.get("failure_reason"),
                },
            )

        validated = bool(privilege_validation.get("validated", True))
        observation = ObservationRecord(
            category="privilege",
            summary=f"Privilege worker validated {request.task_label}",
            confidence=confidence,
            refs=event_refs,
            payload={
                "validated": validated,
                "privilege_validation": privilege_validation,
            },
        )
        evidence = EvidenceArtifact(
            kind="privilege_validation",
            summary=f"Privilege validation evidence for task {request.task_id}",
            payload_ref=f"runtime://outcomes/{request.task_id}/privilege",
            refs=event_refs,
            metadata={"privilege_validation": privilege_validation},
        )
        fact_write_requests = self._build_fact_writes(
            request=request,
            primary_ref=primary_ref,
            privilege_validation=privilege_validation,
            evidence_id=evidence.evidence_id,
            confidence=observation.confidence,
        )
        projection_requests = [
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_TARGETS,
                source_task_id=request.task_id,
                reason="validated privilege state may unlock additional projected actions",
                target_refs=event_refs,
                metadata={"required_level": privilege_validation.get("required_level")},
            )
        ]
        runtime_requests = [
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=request.task_id,
                budget_delta=RuntimeBudgetDelta(
                    operations=1,
                    risk=float(request.metadata.get("risk_cost", 0.1)),
                ),
                reason="privilege validation consumes runtime risk budget",
            )
        ]
        critic_signals: list[CriticSignal] = []
        replan_hints: list[ReplanHint] = []
        privilege_gap_detected = not validated or bool(request.metadata.get("privilege_gap_detected"))
        if privilege_gap_detected:
            critic_signals.append(
                CriticSignal(
                    source_task_id=request.task_id,
                    kind="privilege_gap",
                    severity=CriticSignalSeverity.HIGH,
                    reason="observed privilege state does not satisfy required privilege level",
                    task_ids=[request.task_id],
                    invalidated_ref_keys=[self._ref_key(ref) for ref in event_refs],
                )
            )
            replan_hints.append(
                ReplanHint(
                    source_task_id=request.task_id,
                    scope=ReplanScope.LOCAL,
                    reason="privilege validation exposed a privilege gap on the current branch",
                    task_ids=[request.task_id],
                    invalidated_ref_keys=[self._ref_key(ref) for ref in event_refs],
                )
            )
            runtime_requests.append(
                RuntimeControlRequest(
                    request_type=RuntimeControlType.REQUEST_REPLAN,
                    source_task_id=request.task_id,
                    reason="privilege validation exposed a privilege gap on the current branch",
                    metadata={"scope": ReplanScope.LOCAL.value},
                )
            )

        raw_payload = {
            "status": AgentResultStatus.SUCCEEDED.value,
            "validated": True,
            "privilege_satisfied": validated,
            "privilege_validation": privilege_validation,
            "fact_write_requests": [item.model_dump(mode="json") for item in fact_write_requests],
            "projection_requests": [item.model_dump(mode="json") for item in projection_requests],
            "runtime_requests": [item.model_dump(mode="json") for item in runtime_requests],
            "critic_signals": [item.model_dump(mode="json") for item in critic_signals],
            "replan_hints": [item.model_dump(mode="json") for item in replan_hints],
        }
        return WorkerDomainResult(
            success=True,
            status=AgentResultStatus.SUCCEEDED.value,
            summary=f"privilege validation completed for task {request.task_id}",
            confidence=confidence,
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
    def _privilege_validation_view(request: PrivilegeValidationRequest) -> dict[str, Any]:
        raw = PrivilegeValidationService._metadata_view(request, "privilege_validation")
        if "validated" not in raw:
            raw = {"validated": not bool(request.metadata.get("privilege_gap_detected", False)), **raw}
        principal = (
            PrivilegeValidationService._string(raw.get("principal"))
            or PrivilegeValidationService._string(request.input_bindings.get("principal"))
            or PrivilegeValidationService._string(request.input_bindings.get("identity"))
        )
        return {
            "validated": bool(raw.get("validated", True)),
            "required_level": PrivilegeValidationService._string(raw.get("required_level")),
            "blocked": bool(raw.get("blocked", False)),
            "failure_reason": PrivilegeValidationService._string(raw.get("failure_reason"))
            or PrivilegeValidationService._string(raw.get("reason")),
            "principal": principal,
            "evidence": raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {"source": "request_metadata"},
            "raw_payload": dict(raw),
            **{
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "validated",
                    "required_level",
                    "blocked",
                    "failure_reason",
                    "reason",
                    "principal",
                    "evidence",
                    "raw_payload",
                }
            },
        }

    @staticmethod
    def _build_fact_writes(
        *,
        request: PrivilegeValidationRequest,
        primary_ref: EventGraphRef | None,
        privilege_validation: dict[str, Any],
        evidence_id: str,
        confidence: float,
    ) -> list[FactWriteRequest]:
        requests: list[FactWriteRequest] = []
        if primary_ref is None:
            return requests

        evidence_ref = primary_ref.__class__(graph="kg", ref_id=evidence_id, ref_type="Evidence")
        required_level = PrivilegeValidationService._string(privilege_validation.get("required_level")) or "current"
        privilege_ref = primary_ref.__class__(
            graph="kg",
            ref_id=f"privilege::{primary_ref.ref_id}::{required_level}",
            ref_type="PrivilegeState",
        )
        requests.append(
            FactWriteRequest(
                kind=FactWriteKind.ENTITY_UPSERT,
                source_task_id=request.task_id,
                subject_ref=privilege_ref,
                attributes={
                    "validated": privilege_validation.get("validated", True),
                    "required_level": privilege_validation.get("required_level"),
                    "principal": privilege_validation.get("principal"),
                    "source": privilege_validation.get("source"),
                },
                confidence=confidence,
                evidence_ids=[evidence_id],
                summary=f"Privilege state for {primary_ref.ref_id}",
            )
        )
        requests.append(
            FactWriteRequest(
                kind=FactWriteKind.RELATION_UPSERT,
                source_task_id=request.task_id,
                subject_ref=primary_ref,
                relation_type="HAS_PRIVILEGE_STATE",
                object_ref=privilege_ref,
                attributes={"validated": privilege_validation.get("validated", True)},
                confidence=confidence,
                evidence_ids=[evidence_id],
                summary=f"{primary_ref.ref_id} has privilege state {privilege_ref.ref_id}",
            )
        )
        requests.append(
            FactWriteRequest(
                kind=FactWriteKind.RELATION_UPSERT,
                source_task_id=request.task_id,
                subject_ref=privilege_ref,
                relation_type="SUPPORTED_BY",
                object_ref=evidence_ref,
                attributes={"validation": "privilege"},
                confidence=confidence,
                evidence_ids=[evidence_id],
                summary=f"Privilege evidence supports {privilege_ref.ref_id}",
            )
        )
        return requests

    @staticmethod
    def _metadata_view(request: PrivilegeValidationRequest, *keys: str) -> dict[str, Any]:
        for key in keys:
            raw = request.metadata.get(key)
            if isinstance(raw, dict):
                return dict(raw)
            raw = request.context_metadata.get(key)
            if isinstance(raw, dict):
                return dict(raw)
        return {}

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

    @staticmethod
    def _ref_key(ref: Any) -> str:
        key = getattr(ref, "key", None)
        if callable(key):
            return str(key())
        graph = getattr(ref, "graph", "")
        graph_value = getattr(graph, "value", graph)
        return f"{graph_value}:{getattr(ref, 'ref_id', '')}"

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["PrivilegeValidationRequest", "PrivilegeValidationService"]
