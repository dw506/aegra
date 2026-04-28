"""Worker for access validation and identity-context checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    AgentTaskResult,
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
from src.core.models.runtime import CredentialStatus, RuntimeState
from src.core.models.tg import TaskType
from src.core.runtime.credential_manager import RuntimeCredentialManager
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager
from src.core.runtime.session_manager import RuntimeSessionManager
from src.core.workers.access_validators import (
    CredentialValidatorAdapter,
    PrivilegeValidatorAdapter,
    SessionProbeAdapter,
)
from src.core.workers.base import BaseWorker
from src.core.workers.tool_runner import ToolRunner


@dataclass(slots=True)
class AccessExecutionContext:
    """Normalized runtime snapshot and validator inputs consumed by AccessWorker."""

    runtime_state: RuntimeState | None
    session_probe: dict[str, Any]
    credential_validation: dict[str, Any]
    reachability: dict[str, Any]
    privilege_validation: dict[str, Any]
    selected_session_id: str | None
    selected_route: dict[str, Any]
    require_session: bool
    require_credential: bool
    bound_identity: str | None
    bound_target: str | None
    confidence: float

    @property
    def session_usable(self) -> bool:
        return bool(self.selected_session_id and self.session_probe.get("usable", False))

    @property
    def credential_status(self) -> str:
        return str(self.credential_validation.get("status", CredentialStatus.UNKNOWN.value)).lower()


class AccessWorker(BaseWorker):
    """Worker that validates access paths without mutating the fact graph directly."""

    agent_role = AgentRole.ACCESS_WORKER
    supported_task_types = frozenset(
        {
            TaskType.IDENTITY_CONTEXT_CONFIRMATION,
            TaskType.PRIVILEGE_CONFIGURATION_VALIDATION,
        }
    )
    capabilities = frozenset({"validate_access", "request_sessions", "propose_access_facts"})

    def __init__(self, *, tool_runner: ToolRunner | None = None) -> None:
        self._tool_runner = tool_runner or ToolRunner()
        self._session_manager = RuntimeSessionManager()
        self._credential_manager = RuntimeCredentialManager()
        self._pivot_route_manager = RuntimePivotRouteManager()
        self._session_probe = SessionProbeAdapter(
            session_manager=self._session_manager,
            tool_runner=self._tool_runner,
        )
        self._credential_validator = CredentialValidatorAdapter(
            credential_manager=self._credential_manager,
            tool_runner=self._tool_runner,
        )
        self._privilege_validator = PrivilegeValidatorAdapter()

    def default_intent(self, task_type: TaskType) -> AgentTaskIntent:
        return AgentTaskIntent.VALIDATE_ACCESS

    def handle_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Validate access-oriented tasks and emit runtime/session intents."""

        primary_ref = request.target_refs[0] if request.target_refs else None
        execution_context = self._build_execution_context(request, primary_ref=primary_ref)

        if not execution_context.reachability.get("reachable", True):
            return self._result(
                request,
                status=AgentResultStatus.BLOCKED,
                summary=f"target is not reachable for task {request.context.task_id}",
                outcome_payload={
                    "blocked_on": "reachability",
                    "validated": False,
                    "reachability": execution_context.reachability,
                    "selected_route": execution_context.selected_route,
                },
            )

        if execution_context.session_probe.get("blocked"):
            return self._result(
                request,
                status=AgentResultStatus.BLOCKED,
                summary=execution_context.session_probe.get("failure_reason")
                or f"session probe blocked for task {request.context.task_id}",
                error_message=execution_context.session_probe.get("failure_reason"),
                outcome_payload={
                    "blocked_on": "session_probe",
                    "validated": False,
                    "session_probe": execution_context.session_probe,
                    "selected_route": execution_context.selected_route,
                },
            )

        if execution_context.require_session and not execution_context.session_usable:
            runtime_requests = [
                RuntimeControlRequest(
                    request_type=RuntimeControlType.OPEN_SESSION,
                    source_task_id=request.context.task_id,
                    lease_seconds=int(request.metadata.get("lease_seconds", 300)),
                    reuse_policy=str(request.metadata.get("reuse_policy", "exclusive")),
                    reason="access validation requires a live runtime session",
                    metadata={
                        "bound_identity": execution_context.bound_identity,
                        "bound_target": execution_context.bound_target or (primary_ref.ref_id if primary_ref else None),
                        "selected_route_id": execution_context.selected_route.get("route_id"),
                        "source_host": execution_context.selected_route.get("source_host"),
                        "via_host": execution_context.selected_route.get("via_host"),
                        "protocol": execution_context.selected_route.get("protocol"),
                    },
                )
            ]
            return self._result(
                request,
                status=AgentResultStatus.BLOCKED,
                summary=f"session required before executing {request.context.task_id}",
                runtime_requests=runtime_requests,
                outcome_payload={
                    "blocked_on": "session",
                    "session_probe": execution_context.session_probe,
                    "selected_route": execution_context.selected_route,
                },
            )

        if execution_context.credential_validation.get("blocked"):
            return self._result(
                request,
                status=AgentResultStatus.BLOCKED,
                summary=execution_context.credential_validation.get("failure_reason")
                or f"credential validator blocked for task {request.context.task_id}",
                error_message=execution_context.credential_validation.get("failure_reason"),
                outcome_payload={
                    "blocked_on": "credential_validator",
                    "validated": False,
                    "credential_status": execution_context.credential_status,
                    "credential_validation": execution_context.credential_validation,
                },
            )

        if execution_context.require_credential and execution_context.credential_status != CredentialStatus.VALID.value:
            return self._result(
                request,
                status=AgentResultStatus.FAILED,
                summary=f"credential validation failed for task {request.context.task_id}",
                outcome_payload={
                    "validated": False,
                    "credential_status": execution_context.credential_status,
                    "credential_validation": execution_context.credential_validation,
                },
            )

        observation = ObservationRecord(
            category="access",
            summary=f"Access worker validated {request.task_label}",
            confidence=execution_context.confidence,
            refs=list(request.target_refs),
            payload={
                "session_id": execution_context.selected_session_id,
                "session_probe": execution_context.session_probe,
                "credential_status": execution_context.credential_status,
                "credential_validation": execution_context.credential_validation,
                "reachable": execution_context.reachability.get("reachable", True),
                "reachability": execution_context.reachability,
                "selected_route": execution_context.selected_route,
                "privilege_validation": execution_context.privilege_validation,
            },
        )
        evidence = EvidenceArtifact(
            kind="access_validation",
            summary=f"Access validation evidence for task {request.context.task_id}",
            payload_ref=f"runtime://outcomes/{request.context.task_id}/access",
            refs=list(request.target_refs),
            metadata={
                "selected_route_id": execution_context.selected_route.get("route_id"),
                "session_probe": execution_context.session_probe,
                "credential_validation": execution_context.credential_validation,
            },
        )
        fact_write_requests = self._build_fact_writes(
            request=request,
            primary_ref=primary_ref,
            execution_context=execution_context,
            evidence_id=evidence.evidence_id,
            confidence=observation.confidence,
        )
        projection_requests = [
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_TARGETS,
                source_task_id=request.context.task_id,
                reason="validated access may unlock additional projected actions",
                target_refs=list(request.target_refs),
                metadata={
                    "selected_route_id": execution_context.selected_route.get("route_id"),
                    "reachability_via": execution_context.reachability.get("via"),
                },
            )
        ]
        critic_signals = []
        replan_hints = []
        privilege_gap_detected = (
            not execution_context.privilege_validation.get("validated", True)
            or bool(request.metadata.get("privilege_gap_detected"))
        )
        if privilege_gap_detected:
            critic_signals.append(
                CriticSignal(
                    source_task_id=request.context.task_id,
                    kind="privilege_gap",
                    severity=CriticSignalSeverity.HIGH,
                    reason="observed access path does not satisfy required privilege level",
                    task_ids=[request.context.task_id],
                )
            )
            replan_hints.append(
                ReplanHint(
                    source_task_id=request.context.task_id,
                    scope=ReplanScope.LOCAL,
                    reason="access validation exposed a privilege gap on the current branch",
                    task_ids=[request.context.task_id],
                )
            )
        runtime_requests = [
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=request.context.task_id,
                budget_delta=RuntimeBudgetDelta(
                    operations=1,
                    risk=float(request.metadata.get("risk_cost", 0.1)),
                ),
                reason="access validation consumes runtime risk budget",
            )
        ]
        if privilege_gap_detected:
            runtime_requests.append(
                RuntimeControlRequest(
                    request_type=RuntimeControlType.REQUEST_REPLAN,
                    source_task_id=request.context.task_id,
                    reason="access validation exposed a privilege gap on the current branch",
                    metadata={"scope": ReplanScope.LOCAL.value},
                )
            )
        return self._result(
            request,
            status=AgentResultStatus.SUCCEEDED,
            summary=f"access validation completed for task {request.context.task_id}",
            observations=[observation],
            evidence=[evidence],
            fact_write_requests=fact_write_requests,
            projection_requests=projection_requests,
            runtime_requests=runtime_requests,
            critic_signals=critic_signals,
            replan_hints=replan_hints,
            outcome_payload={
                "session_id": execution_context.selected_session_id,
                "validated": True,
                "credential_status": execution_context.credential_status,
                "credential_validation": execution_context.credential_validation,
                "reachable": execution_context.reachability.get("reachable", True),
                "reachability": execution_context.reachability,
                "selected_route": execution_context.selected_route,
                "privilege_validation": execution_context.privilege_validation,
            },
        )

    def _build_execution_context(
        self,
        request: AgentTaskRequest,
        *,
        primary_ref: Any,
    ) -> AccessExecutionContext:
        runtime_state = self._runtime_snapshot(request)
        bound_target = self._string(request.metadata.get("bound_target")) or (primary_ref.ref_id if primary_ref is not None else None)
        bound_identity = self._string(request.metadata.get("bound_identity"))
        selected_route = self._select_route(
            request,
            runtime_state=runtime_state,
            destination_host=bound_target,
        )
        session_probe = self._session_probe_view(
            request,
            runtime_state=runtime_state,
            bound_target=bound_target,
            bound_identity=bound_identity,
        )
        credential_validation = self._credential_validation_view(
            request,
            runtime_state=runtime_state,
            target_id=bound_target,
        )
        reachability = self._reachability_view(request, selected_route=selected_route, session_probe=session_probe)
        privilege_validation = self._privilege_validation_view(request)
        return AccessExecutionContext(
            runtime_state=runtime_state,
            session_probe=session_probe,
            credential_validation=credential_validation,
            reachability=reachability,
            privilege_validation=privilege_validation,
            selected_session_id=self._string(session_probe.get("session_id")),
            selected_route=selected_route,
            require_session=bool(request.metadata.get("require_session", True)),
            require_credential=bool(request.metadata.get("require_credential", False)),
            bound_identity=bound_identity,
            bound_target=bound_target,
            confidence=float(request.metadata.get("confidence", 0.85)),
        )

    def _runtime_snapshot(self, request: AgentTaskRequest) -> RuntimeState | None:
        raw = request.metadata.get("runtime_snapshot") or request.context.metadata.get("runtime_snapshot")
        if raw is None:
            return None
        if isinstance(raw, RuntimeState):
            return raw.model_copy(deep=True)
        if isinstance(raw, dict):
            return RuntimeState.model_validate(raw)
        raise ValueError("runtime_snapshot must be a RuntimeState or serialized runtime snapshot")

    def _session_probe_view(
        self,
        request: AgentTaskRequest,
        *,
        runtime_state: RuntimeState | None,
        bound_target: str | None,
        bound_identity: str | None,
    ) -> dict[str, Any]:
        return self._session_probe.probe(
            request=request,
            runtime_state=runtime_state,
            bound_target=bound_target,
            bound_identity=bound_identity,
        ).model_dump(mode="json")

    def _credential_validation_view(
        self,
        request: AgentTaskRequest,
        *,
        runtime_state: RuntimeState | None,
        target_id: str | None,
    ) -> dict[str, Any]:
        return self._credential_validator.validate(
            request=request,
            runtime_state=runtime_state,
            target_id=target_id,
        ).model_dump(mode="json")

    def _select_route(
        self,
        request: AgentTaskRequest,
        *,
        runtime_state: RuntimeState | None,
        destination_host: str | None,
    ) -> dict[str, Any]:
        route_view = self._metadata_view(request, "selected_route", "pivot_route")
        if runtime_state is None or destination_host is None:
            return route_view

        reachability = self._metadata_view(request, "reachability", "host_reachability")
        selected_route_id = self._string(route_view.get("route_id"))
        if selected_route_id is not None:
            try:
                route = self._pivot_route_manager.get_route(runtime_state, selected_route_id)
            except ValueError:
                route = None
            if route is not None:
                return self._route_payload(route, route_view)

        prefer_pivot = bool(request.metadata.get("prefer_pivot_route"))
        via = self._string(reachability.get("via"))
        if via == "pivot" or prefer_pivot:
            route = self._pivot_route_manager.select_best_route(
                runtime_state,
                destination_host,
                source_host=self._string(reachability.get("source_id")),
            )
            if route is not None:
                return self._route_payload(route, route_view)
        return route_view

    def _reachability_view(
        self,
        request: AgentTaskRequest,
        *,
        selected_route: dict[str, Any],
        session_probe: dict[str, Any],
    ) -> dict[str, Any]:
        raw = self._metadata_view(request, "reachability", "host_reachability")
        if "reachable" not in raw:
            raw = {"reachable": bool(request.metadata.get("reachable", True)), **raw}
        if selected_route.get("route_id"):
            raw.setdefault("via", "pivot")
            raw.setdefault("route_id", selected_route.get("route_id"))
            raw.setdefault("source_id", selected_route.get("source_host"))
            raw.setdefault("source_type", "Host")
        elif self._string(session_probe.get("session_id")):
            raw.setdefault("via", "session")
        else:
            raw.setdefault("via", "direct")
        return raw

    @staticmethod
    def _privilege_validation_view(request: AgentTaskRequest) -> dict[str, Any]:
        return PrivilegeValidatorAdapter().validate(request=request).model_dump(mode="json")

    @staticmethod
    def _build_fact_writes(
        *,
        request: AgentTaskRequest,
        primary_ref: Any,
        execution_context: AccessExecutionContext,
        evidence_id: str,
        confidence: float,
    ) -> list[FactWriteRequest]:
        requests: list[FactWriteRequest] = []
        evidence_ref = (
            request.target_refs[0].__class__(graph="kg", ref_id=evidence_id, ref_type="Evidence")
            if request.target_refs
            else None
        )

        if primary_ref is not None:
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=primary_ref,
                    attributes={
                        "access_validated": True,
                        "reachable": execution_context.reachability.get("reachable", True),
                        "credential_status": execution_context.credential_status,
                        "session_id": execution_context.selected_session_id,
                        "route_id": execution_context.selected_route.get("route_id"),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Validated access state for {primary_ref.ref_id}",
                )
            )
            if evidence_ref is not None:
                requests.append(
                    FactWriteRequest(
                        kind=FactWriteKind.RELATION_UPSERT,
                        source_task_id=request.context.task_id,
                        subject_ref=primary_ref,
                        relation_type="SUPPORTED_BY",
                        object_ref=evidence_ref,
                        attributes={"validation": "access"},
                        confidence=confidence,
                        evidence_ids=[evidence_id],
                        summary=f"Access evidence supports {primary_ref.ref_id}",
                    )
                )

        if execution_context.selected_session_id:
            session_ref = request.target_refs[0].__class__(
                graph="kg",
                ref_id=execution_context.selected_session_id,
                ref_type="Session",
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=session_ref,
                    attributes={
                        "session_state": "available",
                        "session_available": True,
                        "bound_target": (primary_ref.ref_id if primary_ref is not None else None),
                        "selected_route_id": execution_context.selected_route.get("route_id"),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Session available {execution_context.selected_session_id}",
                )
            )
            if primary_ref is not None:
                requests.append(
                    FactWriteRequest(
                        kind=FactWriteKind.RELATION_UPSERT,
                        source_task_id=request.context.task_id,
                        subject_ref=session_ref,
                        relation_type="SESSION_ON",
                        object_ref=primary_ref,
                        attributes={"reachable_via": execution_context.reachability.get("via")},
                        confidence=confidence,
                        evidence_ids=[evidence_id],
                        summary=f"Session {execution_context.selected_session_id} established on {primary_ref.ref_id}",
                    )
                )

        credential_id = AccessWorker._string(execution_context.credential_validation.get("credential_id"))
        if credential_id:
            credential_ref = request.target_refs[0].__class__(graph="kg", ref_id=credential_id, ref_type="Credential")
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=credential_ref,
                    attributes={
                        "credential_status": execution_context.credential_status,
                        "validated": execution_context.credential_status == "valid",
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Credential validation state for {credential_id}",
                )
            )
            if primary_ref is not None:
                requests.append(
                    FactWriteRequest(
                        kind=FactWriteKind.RELATION_UPSERT,
                        source_task_id=request.context.task_id,
                        subject_ref=credential_ref,
                        relation_type="AUTHENTICATES_AS",
                        object_ref=primary_ref,
                        attributes={"credential_status": execution_context.credential_status},
                        confidence=confidence,
                        evidence_ids=[evidence_id],
                        summary=f"Credential {credential_id} authenticates toward {primary_ref.ref_id}",
                    )
                )

        if primary_ref is not None and execution_context.reachability.get("reachable", True):
            source_ref = request.target_refs[0].__class__(
                graph="kg",
                ref_id=AccessWorker._string(execution_context.reachability.get("source_id"))
                or AccessWorker._string(execution_context.selected_route.get("route_id"))
                or f"reachability::{request.context.task_id}",
                ref_type=(
                    "PivotRoute"
                    if execution_context.selected_route.get("route_id")
                    else (AccessWorker._string(execution_context.reachability.get("source_type")) or "Host")
                ),
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=source_ref,
                    relation_type="CAN_REACH",
                    object_ref=primary_ref,
                    attributes={
                        "reachable_via": execution_context.reachability.get("via")
                        or ("session" if execution_context.selected_session_id else "direct"),
                        "route_id": execution_context.selected_route.get("route_id"),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Reachability validated toward {primary_ref.ref_id}",
                )
            )

        if primary_ref is not None:
            privilege_ref = request.target_refs[0].__class__(
                graph="kg",
                ref_id=(
                    f"privilege::{primary_ref.ref_id}::"
                    f"{AccessWorker._string(execution_context.privilege_validation.get('required_level')) or 'current'}"
                ),
                ref_type="PrivilegeState",
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=privilege_ref,
                    attributes={
                        "validated": execution_context.privilege_validation.get("validated", True),
                        "required_level": execution_context.privilege_validation.get("required_level"),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Privilege state for {primary_ref.ref_id}",
                )
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.context.task_id,
                    subject_ref=primary_ref,
                    relation_type="HAS_PRIVILEGE_STATE",
                    object_ref=privilege_ref,
                    attributes={"validated": execution_context.privilege_validation.get("validated", True)},
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"{primary_ref.ref_id} has privilege state {privilege_ref.ref_id}",
                )
            )

        return requests

    @staticmethod
    def _metadata_view(request: AgentTaskRequest, *keys: str) -> dict[str, Any]:
        for key in keys:
            raw = request.metadata.get(key)
            if isinstance(raw, dict):
                return dict(raw)
            raw = request.context.metadata.get(key)
            if isinstance(raw, dict):
                return dict(raw)
        return {}

    @staticmethod
    def _route_payload(route: Any, existing: dict[str, Any]) -> dict[str, Any]:
        return existing | {
            "route_id": route.route_id,
            "destination_host": route.destination_host,
            "source_host": route.source_host,
            "via_host": route.via_host,
            "session_id": route.session_id,
            "protocol": route.protocol,
            "status": route.status.value,
        }

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["AccessWorker"]
