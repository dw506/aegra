"""Domain service for access validation worker behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import AgentInput
from src.core.execution.executor import ExecutionExecutor
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult
from src.core.models.ag import GraphRef as EventGraphRef
from src.core.models.events import (
    AgentResultStatus,
    AgentTaskRequest,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ObservationRecord,
    ProjectionRequest,
    ProjectionRequestKind,
    RuntimeBudgetDelta,
    RuntimeControlRequest,
    RuntimeControlType,
)
from src.core.models.runtime import CredentialStatus, RuntimeState
from src.core.runtime.credential_manager import RuntimeCredentialManager
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager
from src.core.runtime.session_manager import RuntimeSessionManager
from src.core.workers.access_validators import CredentialValidatorAdapter, SessionProbeAdapter
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.services.result_builders import WorkerDomainResult
from src.core.workers.tool_runner import ToolRunner


class AccessValidationRequest(BaseModel):
    """Domain input consumed by `AccessValidationService`."""

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
    session_id: str | None = None

    @classmethod
    def from_task_spec(
        cls,
        *,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> "AccessValidationRequest":
        raw = dict(agent_input.raw_payload)
        metadata = dict(raw.get("metadata", {}))
        metadata.update(cls._default_agent_access_metadata(task_spec))
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
            session_id=cls._string(task_spec.constraints.get("session_id") or task_spec.input_bindings.get("session_id")),
        )

    @classmethod
    def from_legacy_request(cls, request: AgentTaskRequest) -> "AccessValidationRequest":
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
            session_id=request.context.session_id,
        )

    @classmethod
    def _default_agent_access_metadata(cls, task_spec: WorkerTaskSpec) -> dict[str, Any]:
        session_id = cls._string(task_spec.constraints.get("session_id") or task_spec.input_bindings.get("session_id"))
        route = task_spec.input_bindings.get("route") or task_spec.input_bindings.get("path") or task_spec.constraints.get("path")
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["session_probe"] = {"session_id": session_id, "status": "active", "usable": True}
        if route:
            metadata["reachability"] = {"reachable": True, "via": "session" if session_id else "direct", "path": route}
        return metadata

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


@dataclass(slots=True)
class AccessExecutionContext:
    """Normalized access/session/credential context for one service run."""

    runtime_state: RuntimeState | None
    session_probe: dict[str, Any]
    credential_validation: dict[str, Any]
    reachability: dict[str, Any]
    selected_session_id: str | None
    selected_route: dict[str, Any]
    require_session: bool
    require_credential: bool
    bound_identity: str | None
    bound_target: str | None
    confidence: float
    tool_execution: dict[str, Any] | None = None

    @property
    def session_usable(self) -> bool:
        return bool(self.selected_session_id and self.session_probe.get("usable", False))

    @property
    def credential_status(self) -> str:
        return str(self.credential_validation.get("status", CredentialStatus.UNKNOWN.value)).lower()


class AccessValidationService:
    """Validate access path, session and credential state."""

    def __init__(
        self,
        *,
        tool_runner: ToolRunner | None = None,
        executor: ExecutionExecutor | None = None,
    ) -> None:
        self._executor = executor
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

    def validate(self, request: AccessValidationRequest) -> WorkerDomainResult:
        tool_execution = self._run_session_probe_if_available(request)
        if tool_execution is not None:
            request = self._request_with_tool_execution(request, tool_execution)
        primary_ref = self._event_ref(request.target_refs[0]) if request.target_refs else None
        context = self._build_execution_context(request, primary_ref=primary_ref)

        if not context.reachability.get("reachable", True):
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.BLOCKED.value,
                summary=f"target is not reachable for task {request.task_id}",
                raw_payload={
                    "status": AgentResultStatus.BLOCKED.value,
                    "blocked_on": "reachability",
                    "validated": False,
                    "reachability": context.reachability,
                    "selected_route": context.selected_route,
                },
            )

        if context.session_probe.get("blocked"):
            failure_reason = context.session_probe.get("failure_reason")
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.BLOCKED.value,
                summary=failure_reason or f"session probe blocked for task {request.task_id}",
                raw_payload={
                    "status": AgentResultStatus.BLOCKED.value,
                    "blocked_on": "session_probe",
                    "validated": False,
                    **self._tool_execution_payload(context),
                    "session_probe": context.session_probe,
                    "selected_route": context.selected_route,
                    "error_message": failure_reason,
                },
            )

        if context.require_session and not context.session_usable:
            runtime_requests = [
                RuntimeControlRequest(
                    request_type=RuntimeControlType.OPEN_SESSION,
                    source_task_id=request.task_id,
                    lease_seconds=int(request.metadata.get("lease_seconds", 300)),
                    reuse_policy=str(request.metadata.get("reuse_policy", "exclusive")),
                    reason="access validation requires a live runtime session",
                    metadata={
                        "bound_identity": context.bound_identity,
                        "bound_target": context.bound_target or (primary_ref.ref_id if primary_ref else None),
                        "selected_route_id": context.selected_route.get("route_id"),
                        "source_host": context.selected_route.get("source_host"),
                        "via_host": context.selected_route.get("via_host"),
                        "protocol": context.selected_route.get("protocol"),
                    },
                )
            ]
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.BLOCKED.value,
                summary=f"session required before executing {request.task_id}",
                runtime_requests=[item.model_dump(mode="json") for item in runtime_requests],
                raw_payload={
                    "status": AgentResultStatus.BLOCKED.value,
                    "blocked_on": "session",
                    **self._tool_execution_payload(context),
                    "session_probe": context.session_probe,
                    "selected_route": context.selected_route,
                    "runtime_requests": [item.model_dump(mode="json") for item in runtime_requests],
                },
            )

        if context.credential_validation.get("blocked"):
            failure_reason = context.credential_validation.get("failure_reason")
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.BLOCKED.value,
                summary=failure_reason or f"credential validator blocked for task {request.task_id}",
                raw_payload={
                    "status": AgentResultStatus.BLOCKED.value,
                    "blocked_on": "credential_validator",
                    "validated": False,
                    "credential_status": context.credential_status,
                    "credential_validation": context.credential_validation,
                    "error_message": failure_reason,
                },
            )

        if context.require_credential and context.credential_status != CredentialStatus.VALID.value:
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.FAILED.value,
                summary=f"credential validation failed for task {request.task_id}",
                raw_payload={
                    "status": AgentResultStatus.FAILED.value,
                    "validated": False,
                    "credential_status": context.credential_status,
                    "credential_validation": context.credential_validation,
                },
            )

        event_refs = [self._event_ref(ref) for ref in request.target_refs]
        observation = ObservationRecord(
            category="access",
            summary=f"Access worker validated {request.task_label}",
            confidence=context.confidence,
            refs=event_refs,
            payload={
                "session_id": context.selected_session_id,
                "session_probe": context.session_probe,
                **self._tool_execution_payload(context),
                "credential_status": context.credential_status,
                "credential_validation": context.credential_validation,
                "reachable": context.reachability.get("reachable", True),
                "reachability": context.reachability,
                "selected_route": context.selected_route,
            },
        )
        evidence = EvidenceArtifact(
            kind="access_validation",
            summary=f"Access validation evidence for task {request.task_id}",
            payload_ref=f"runtime://outcomes/{request.task_id}/access",
            refs=event_refs,
            metadata={
                "selected_route_id": context.selected_route.get("route_id"),
                "session_probe": context.session_probe,
                **self._tool_execution_payload(context),
                "credential_validation": context.credential_validation,
            },
        )
        fact_write_requests = self._build_fact_writes(
            request=request,
            primary_ref=primary_ref,
            context=context,
            evidence_id=evidence.evidence_id,
            confidence=observation.confidence,
        )
        projection_requests = [
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_TARGETS,
                source_task_id=request.task_id,
                reason="validated access may unlock additional projected actions",
                target_refs=event_refs,
                metadata={
                    "selected_route_id": context.selected_route.get("route_id"),
                    "reachability_via": context.reachability.get("via"),
                },
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
                reason="access validation consumes runtime risk budget",
            )
        ]
        raw_payload = {
            "status": AgentResultStatus.SUCCEEDED.value,
            "session_id": context.selected_session_id,
            "validated": True,
            "session_probe": context.session_probe,
            "credential_status": context.credential_status,
            "credential_validation": context.credential_validation,
            "reachable": context.reachability.get("reachable", True),
            "reachability": context.reachability,
            "selected_route": context.selected_route,
            **self._tool_execution_payload(context),
            "fact_write_requests": [item.model_dump(mode="json") for item in fact_write_requests],
            "projection_requests": [item.model_dump(mode="json") for item in projection_requests],
            "runtime_requests": [item.model_dump(mode="json") for item in runtime_requests],
        }
        return WorkerDomainResult(
            success=True,
            status=AgentResultStatus.SUCCEEDED.value,
            summary=f"access validation completed for task {request.task_id}",
            confidence=context.confidence,
            observations=[observation.model_dump(mode="json")],
            evidence=[evidence.model_dump(mode="json")],
            fact_write_requests=raw_payload["fact_write_requests"],
            projection_requests=raw_payload["projection_requests"],
            runtime_requests=raw_payload["runtime_requests"],
            raw_payload=raw_payload,
        )

    def _run_session_probe_if_available(self, request: AccessValidationRequest) -> ToolExecutionResult | None:
        if self._executor is None:
            return None

        command = request.metadata.get("probe_command")
        adapter = request.metadata.get("execution_adapter")
        if not command or not adapter:
            return None

        agent_id = self._string(request.metadata.get("agent_id") or request.metadata.get("target_agent_ref"))
        args: dict[str, Any] = {}
        if agent_id:
            args["agent_id"] = agent_id
        if isinstance(command, list):
            args["argv"] = [str(part) for part in command]
            command_text = " ".join(str(part) for part in command)
        else:
            command_text = str(command)
        metadata = {"probe": "session"}
        if agent_id:
            metadata["agent_id"] = agent_id
        plan = ToolPlan(
            task_id=request.task_id,
            tool="session_probe",
            adapter=str(adapter),
            command=command_text,
            target_agent_ref=agent_id,
            args=args,
            timeout_seconds=int(request.metadata.get("timeout_seconds") or request.metadata.get("probe_timeout_seconds") or 30),
            payloads=dict(request.metadata.get("payloads", {})),
            metadata=metadata,
        )
        return self._executor.execute(plan)

    def _request_with_tool_execution(
        self,
        request: AccessValidationRequest,
        tool_execution: ToolExecutionResult,
    ) -> AccessValidationRequest:
        metadata = dict(request.metadata)
        payload = tool_execution.model_dump(mode="json")
        metadata["tool_execution"] = payload
        if "session_probe" not in metadata:
            session_id = (
                self._string(metadata.get("session_id"))
                or self._string(metadata.get("agent_id"))
                or self._string(tool_execution.command_id)
                or f"session-probe::{request.task_id}"
            )
            metadata["session_probe"] = {
                "session_id": session_id,
                "status": "active" if tool_execution.success else "failed",
                "usable": tool_execution.success,
                "blocked": not tool_execution.success,
                "failure_reason": None if tool_execution.success else (tool_execution.stderr or "session probe execution failed"),
                "tool_execution": payload,
            }
        return request.model_copy(update={"metadata": metadata}, deep=True)

    @staticmethod
    def _tool_execution_payload(context: AccessExecutionContext) -> dict[str, Any]:
        if isinstance(context.tool_execution, dict):
            return {"tool_execution": dict(context.tool_execution)}
        return {}

    def _build_execution_context(
        self,
        request: AccessValidationRequest,
        *,
        primary_ref: EventGraphRef | None,
    ) -> AccessExecutionContext:
        adapter_request = self._adapter_request(request)
        runtime_state = self._runtime_snapshot(request)
        bound_target = self._string(request.metadata.get("bound_target")) or (primary_ref.ref_id if primary_ref is not None else None)
        bound_identity = self._string(request.metadata.get("bound_identity"))
        selected_route = self._select_route(request, runtime_state=runtime_state, destination_host=bound_target)
        session_probe = self._session_probe.probe(
            request=adapter_request,
            runtime_state=runtime_state,
            bound_target=bound_target,
            bound_identity=bound_identity,
        ).model_dump(mode="json")
        credential_validation = self._credential_validator.validate(
            request=adapter_request,
            runtime_state=runtime_state,
            target_id=bound_target,
        ).model_dump(mode="json")
        reachability = self._reachability_view(request, selected_route=selected_route, session_probe=session_probe)
        return AccessExecutionContext(
            runtime_state=runtime_state,
            session_probe=session_probe,
            credential_validation=credential_validation,
            reachability=reachability,
            selected_session_id=self._string(session_probe.get("session_id")),
            selected_route=selected_route,
            require_session=bool(request.metadata.get("require_session", True)),
            require_credential=bool(request.metadata.get("require_credential", False)),
            bound_identity=bound_identity,
            bound_target=bound_target,
            confidence=float(request.metadata.get("confidence", 0.85)),
            tool_execution=(
                dict(request.metadata["tool_execution"])
                if isinstance(request.metadata.get("tool_execution"), dict)
                else None
            ),
        )

    def _runtime_snapshot(self, request: AccessValidationRequest) -> RuntimeState | None:
        raw = request.metadata.get("runtime_snapshot") or request.context_metadata.get("runtime_snapshot")
        if raw is None:
            return None
        if isinstance(raw, RuntimeState):
            return raw.model_copy(deep=True)
        if isinstance(raw, dict):
            return RuntimeState.model_validate(raw)
        raise ValueError("runtime_snapshot must be a RuntimeState or serialized runtime snapshot")

    def _select_route(
        self,
        request: AccessValidationRequest,
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
        request: AccessValidationRequest,
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
    def _build_fact_writes(
        *,
        request: AccessValidationRequest,
        primary_ref: EventGraphRef | None,
        context: AccessExecutionContext,
        evidence_id: str,
        confidence: float,
    ) -> list[FactWriteRequest]:
        requests: list[FactWriteRequest] = []
        evidence_ref = primary_ref.__class__(graph="kg", ref_id=evidence_id, ref_type="Evidence") if primary_ref is not None else None

        if primary_ref is not None:
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.task_id,
                    subject_ref=primary_ref,
                    attributes={
                        "access_validated": True,
                        "reachable": context.reachability.get("reachable", True),
                        "credential_status": context.credential_status,
                        "session_id": context.selected_session_id,
                        "route_id": context.selected_route.get("route_id"),
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
                        source_task_id=request.task_id,
                        subject_ref=primary_ref,
                        relation_type="SUPPORTED_BY",
                        object_ref=evidence_ref,
                        attributes={"validation": "access"},
                        confidence=confidence,
                        evidence_ids=[evidence_id],
                        summary=f"Access evidence supports {primary_ref.ref_id}",
                    )
                )

        if context.selected_session_id and primary_ref is not None:
            session_ref = primary_ref.__class__(graph="kg", ref_id=context.selected_session_id, ref_type="Session")
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.task_id,
                    subject_ref=session_ref,
                    attributes={
                        "session_state": "available",
                        "session_available": True,
                        "bound_target": primary_ref.ref_id,
                        "selected_route_id": context.selected_route.get("route_id"),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Session available {context.selected_session_id}",
                )
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.task_id,
                    subject_ref=session_ref,
                    relation_type="SESSION_ON",
                    object_ref=primary_ref,
                    attributes={"reachable_via": context.reachability.get("via")},
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Session {context.selected_session_id} established on {primary_ref.ref_id}",
                )
            )

        credential_id = AccessValidationService._string(context.credential_validation.get("credential_id"))
        if credential_id and primary_ref is not None:
            credential_ref = primary_ref.__class__(graph="kg", ref_id=credential_id, ref_type="Credential")
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=request.task_id,
                    subject_ref=credential_ref,
                    attributes={
                        "credential_status": context.credential_status,
                        "validated": context.credential_status == "valid",
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Credential validation state for {credential_id}",
                )
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.task_id,
                    subject_ref=credential_ref,
                    relation_type="AUTHENTICATES_AS",
                    object_ref=primary_ref,
                    attributes={"credential_status": context.credential_status},
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Credential {credential_id} authenticates toward {primary_ref.ref_id}",
                )
            )

        if primary_ref is not None and context.reachability.get("reachable", True):
            source_ref = primary_ref.__class__(
                graph="kg",
                ref_id=AccessValidationService._string(context.reachability.get("source_id"))
                or AccessValidationService._string(context.selected_route.get("route_id"))
                or f"reachability::{request.task_id}",
                ref_type=(
                    "PivotRoute"
                    if context.selected_route.get("route_id")
                    else (AccessValidationService._string(context.reachability.get("source_type")) or "Host")
                ),
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=request.task_id,
                    subject_ref=source_ref,
                    relation_type="CAN_REACH",
                    object_ref=primary_ref,
                    attributes={
                        "reachable_via": context.reachability.get("via") or ("session" if context.selected_session_id else "direct"),
                        "route_id": context.selected_route.get("route_id"),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Reachability validated toward {primary_ref.ref_id}",
                )
            )

        return requests

    def _adapter_request(self, request: AccessValidationRequest) -> Any:
        task_type = SimpleNamespace(value=request.task_type)
        context = SimpleNamespace(
            operation_id=request.operation_id,
            task_id=request.task_id,
            task_type=task_type,
            session_id=request.session_id,
            metadata=request.context_metadata,
        )
        return SimpleNamespace(
            context=context,
            metadata=request.metadata,
            target_refs=request.target_refs,
            input_bindings=request.input_bindings,
        )

    @staticmethod
    def _metadata_view(request: AccessValidationRequest, *keys: str) -> dict[str, Any]:
        for key in keys:
            raw = request.metadata.get(key)
            if isinstance(raw, dict):
                return dict(raw)
            raw = request.context_metadata.get(key)
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
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["AccessValidationRequest", "AccessValidationService"]
