"""Recon worker for host, service, and identity/context discovery tasks."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput, GraphRef, GraphScope
from src.core.models.ag import GraphRef as KGFactRef
from src.core.models.events import (
    AgentExecutionContext,
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    AgentTaskResult,
    CheckpointHint,
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
from src.core.models.runtime import TaskRuntime
from src.core.models.tg import BaseTaskNode, TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.probe_adapters import (
    CustomProbeAdapter,
    MasscanAdapter,
    NmapAdapter,
    ParsedProbeResult,
    ProbeAdapter,
    ProbeAdapterUnavailable,
)
from src.core.workers.tool_runner import ToolExecutionResult, ToolExecutionSpec, ToolRunner


class ReconWorker(BaseWorkerAgent):
    """Concrete worker for environment and context discovery operations."""

    capabilities = frozenset({WorkerCapability.RECON, WorkerCapability.CONTEXT_VALIDATION})
    agent_role = AgentRole.RECON_WORKER
    compatibility_task_types = frozenset(
        {
            TaskType.ASSET_CONFIRMATION,
            TaskType.SERVICE_VALIDATION,
            TaskType.REACHABILITY_VALIDATION,
            TaskType.IDENTITY_CONTEXT_CONFIRMATION,
        }
    )
    supported_task_types = frozenset({"host_discovery", "service_validation", "identity_context_discovery"})

    def __init__(
        self,
        name: str = "recon_worker",
        *,
        tool_runner: ToolRunner | None = None,
        probe_adapters: list[ProbeAdapter] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._tool_runner = tool_runner or ToolRunner()
        adapters = probe_adapters or [NmapAdapter(), MasscanAdapter(), CustomProbeAdapter()]
        self._probe_adapters = {adapter.adapter_name: adapter for adapter in adapters}

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        """Return True when the task is one of the supported recon operations."""

        return task_spec.task_type in self.supported_task_types

    def build_request(
        self,
        task: BaseTaskNode,
        operation_id: str,
        task_runtime: TaskRuntime | None = None,
        session_id: str | None = None,
        assigned_worker_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTaskRequest:
        """Build the compatibility worker request used by the legacy worker tests."""

        if task.task_type not in self.compatibility_task_types:
            raise ValueError(f"{self.agent_role.value} does not support task type {task.task_type.value}")
        runtime = task_runtime or TaskRuntime(
            task_id=task.id,
            tg_node_id=task.id,
            attempt_count=task.attempt_count,
            max_attempts=task.max_attempts,
            deadline=task.deadline,
            resource_keys=set(task.resource_keys),
        )
        context = AgentExecutionContext(
            operation_id=operation_id,
            task_id=task.id,
            tg_node_id=runtime.tg_node_id or task.id,
            task_type=task.task_type,
            attempt_count=runtime.attempt_count,
            max_attempts=runtime.max_attempts,
            assigned_worker_id=assigned_worker_id or runtime.assigned_worker,
            session_id=session_id or runtime.metadata.get("session_id"),
            checkpoint_ref=runtime.checkpoint_ref,
            deadline=runtime.deadline or task.deadline,
            resource_keys=set(task.resource_keys) | set(runtime.resource_keys),
            metadata=dict(runtime.metadata),
        )
        return AgentTaskRequest(
            agent_role=self.agent_role,
            intent=self.default_intent(task.task_type),
            context=context,
            task_label=task.label,
            input_bindings=dict(task.input_bindings),
            target_refs=list(task.target_refs),
            source_refs=list(task.source_refs),
            expected_output_refs=list(task.expected_output_refs),
            metadata=metadata or {},
        )

    def default_intent(self, task_type: TaskType) -> AgentTaskIntent:
        """Return the default compatibility request intent for recon tasks."""

        return AgentTaskIntent.COLLECT_EVIDENCE

    def execute_task(
        self,
        task_spec: WorkerTaskSpec | AgentTaskRequest,
        agent_input: AgentInput | None = None,
    ) -> AgentOutput | AgentTaskResult:
        """Execute either the new agent protocol or the legacy request protocol."""

        if isinstance(task_spec, AgentTaskRequest):
            return self._execute_compat_task(task_spec)
        if agent_input is None:
            raise ValueError("agent_input is required when executing a worker task spec")
        return self._execute_agent_task(task_spec, agent_input)

    def _execute_agent_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        """Execute one recon-style task and return outcome plus raw result."""

        if task_spec.task_type == "host_discovery":
            raw_result = self._execute_host_discovery(task_spec, agent_input)
        elif task_spec.task_type == "service_validation":
            raw_result = self._execute_service_validation(task_spec, agent_input)
        elif task_spec.task_type == "identity_context_discovery":
            raw_result = self._execute_identity_context_discovery(task_spec, agent_input)
        else:
            raise ValueError(f"unsupported recon task type: {task_spec.task_type}")

        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=bool(raw_result.get("success", True)),
            summary=str(raw_result["summary"]),
            raw_result_ref=str(raw_result["payload_ref"]),
            confidence=float(raw_result.get("confidence", 0.65)),
            refs=task_spec.target_refs,
            payload={
                "task_type": task_spec.task_type,
                "executor": raw_result.get("executor", "tool_runner"),
                "result_type": raw_result.get("result_type", task_spec.task_type),
                "tool": raw_result.get("tool", {}),
                "parsed": raw_result.get("parsed", {}),
            },
        )
        return AgentOutput(
            outcomes=[outcome.to_agent_output_fragment()],
            evidence=[raw_result],
            logs=[
                f"worker={self.name}",
                f"task_id={task_spec.task_id}",
                f"task_type={task_spec.task_type}",
                f"target_count={len(task_spec.target_refs)}",
                f"raw_result_ref={raw_result['payload_ref']}",
                str(raw_result["summary"]),
            ],
        )

    def _execute_compat_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Execute the legacy worker request protocol used by compatibility tests."""

        if request.agent_role != self.agent_role:
            return AgentTaskResult(
                request_id=request.request_id,
                agent_role=self.agent_role,
                operation_id=request.context.operation_id,
                task_id=request.context.task_id,
                tg_node_id=request.context.tg_node_id,
                status=AgentResultStatus.FAILED,
                summary=f"{self.agent_role.value} failed while handling task",
                error_message=f"request agent_role {request.agent_role.value} does not match {self.agent_role.value}",
            )
        if request.context.task_type not in self.compatibility_task_types:
            return AgentTaskResult(
                request_id=request.request_id,
                agent_role=self.agent_role,
                operation_id=request.context.operation_id,
                task_id=request.context.task_id,
                tg_node_id=request.context.tg_node_id,
                status=AgentResultStatus.FAILED,
                summary=f"{self.agent_role.value} failed while handling task",
                error_message=f"{self.agent_role.value} does not support task type {request.context.task_type.value}",
            )

        primary_ref = request.target_refs[0] if request.target_refs else None
        execution = self._run_probe(
            task_id=request.context.task_id,
            metadata=request.metadata,
            target_hint=(primary_ref.ref_id if primary_ref else request.context.task_id),
            mode=self._compat_result_type(request.context.task_type),
        )
        normalized = self._normalize_tool_payload(
            execution=execution,
            result_type=self._compat_result_type(request.context.task_type),
            target_hint=(primary_ref.ref_id if primary_ref else request.context.task_id),
            metadata=request.metadata,
        )
        observation = ObservationRecord(
            category="recon",
            summary=str(normalized["summary"]),
            confidence=float(normalized.get("confidence", request.metadata.get("confidence", 0.8))),
            refs=list(request.target_refs),
            payload={
                "task_type": request.context.task_type.value,
                "tool": normalized["tool"],
                "parsed": normalized["parsed"],
            },
        )
        evidence = EvidenceArtifact(
            kind="recon_result",
            summary=str(normalized["summary"]),
            payload_ref=str(normalized["payload_ref"]),
            refs=list(request.target_refs),
            metadata={"tool": normalized["tool"]},
        )
        fact_write_requests = self._fact_writes_from_parsed(
            task_id=request.context.task_id,
            parsed=normalized["parsed"],
            primary_ref=primary_ref,
            evidence_id=evidence.evidence_id,
            confidence=observation.confidence,
            task_type=request.context.task_type.value,
        )
        projection_requests = [
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_LOCAL_FRONTIER,
                source_task_id=request.context.task_id,
                reason="recon may unlock additional local planning actions",
                target_refs=list(request.target_refs),
            )
        ]
        runtime_requests = [
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=request.context.task_id,
                budget_delta=RuntimeBudgetDelta(
                    operations=1,
                    noise=float(request.metadata.get("noise_cost", 0.1)),
                ),
                reason="recon consumes one worker execution budget unit",
            )
        ]
        checkpoint_hints = []
        if request.metadata.get("require_checkpoint"):
            checkpoint_hints.append(
                CheckpointHint(
                    source_task_id=request.context.task_id,
                    summary=f"Checkpoint recommended after recon task {request.context.task_id}",
                    created_after_tasks=[request.context.task_id],
                )
            )
        return AgentTaskResult(
            request_id=request.request_id,
            agent_role=self.agent_role,
            operation_id=request.context.operation_id,
            task_id=request.context.task_id,
            tg_node_id=request.context.tg_node_id,
            status=(
                AgentResultStatus.SUCCEEDED
                if normalized["success"]
                else (AgentResultStatus.BLOCKED if normalized.get("blocked") else AgentResultStatus.FAILED)
            ),
            summary=str(normalized["summary"]),
            observations=[observation],
            evidence=[evidence],
            fact_write_requests=fact_write_requests,
            projection_requests=projection_requests,
            runtime_requests=runtime_requests,
            checkpoint_hints=checkpoint_hints,
            outcome_payload={
                "observed": bool(normalized["success"]),
                "tool": normalized["tool"],
                "parsed": normalized["parsed"],
            },
            error_message=(None if normalized["success"] else normalized["tool"].get("error_message")),
        )

    def _execute_host_discovery(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Execute host discovery via ToolRunner and normalize the result."""

        host_hint = (
            task_spec.input_bindings.get("host")
            or task_spec.input_bindings.get("host_id")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Host")
            or "unknown-host"
        )
        execution = self._run_probe(
            task_id=task_spec.task_id,
            metadata=agent_input.raw_payload,
            target_hint=host_hint,
            mode="host_discovery",
        )
        return self._build_worker_raw_result(
            task_id=task_spec.task_id,
            result_type="host_discovery_result",
            execution=execution,
            refs=task_spec.target_refs,
            metadata=agent_input.raw_payload,
            operation_id=agent_input.context.operation_id,
            target_hint=host_hint,
        )

    def _execute_service_validation(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Execute service validation via ToolRunner and normalize the result."""

        service_hint = (
            task_spec.input_bindings.get("service")
            or task_spec.input_bindings.get("service_id")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Service")
            or "unknown-service"
        )
        port = task_spec.input_bindings.get("port") or task_spec.constraints.get("port")
        execution = self._run_probe(
            task_id=task_spec.task_id,
            metadata=agent_input.raw_payload | {"service_port": port},
            target_hint=service_hint,
            mode="service_validation",
        )
        return self._build_worker_raw_result(
            task_id=task_spec.task_id,
            result_type="service_validation_result",
            execution=execution,
            refs=task_spec.target_refs,
            metadata=agent_input.raw_payload | {"service_port": port},
            operation_id=agent_input.context.operation_id,
            target_hint=service_hint,
        )

    def _execute_identity_context_discovery(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Execute identity/context discovery via ToolRunner and normalize the result."""

        identity_hint = (
            task_spec.input_bindings.get("identity")
            or task_spec.input_bindings.get("identity_id")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Identity")
            or "unknown-identity"
        )
        execution = self._run_probe(
            task_id=task_spec.task_id,
            metadata=agent_input.raw_payload | {"session_id": task_spec.constraints.get("session_id")},
            target_hint=identity_hint,
            mode="identity_context_discovery",
        )
        return self._build_worker_raw_result(
            task_id=task_spec.task_id,
            result_type="identity_context_result",
            execution=execution,
            refs=task_spec.target_refs,
            metadata=agent_input.raw_payload | {"session_id": task_spec.constraints.get("session_id")},
            operation_id=agent_input.context.operation_id,
            target_hint=identity_hint,
        )

    def _run_probe(
        self,
        *,
        task_id: str,
        metadata: dict[str, Any],
        target_hint: str,
        mode: str,
    ) -> dict[str, Any]:
        attempts: list[str] = []
        for adapter in self._candidate_adapters(metadata=metadata, mode=mode):
            try:
                command = adapter.build_command(target_hint=target_hint, mode=mode, metadata=metadata)
            except ProbeAdapterUnavailable as exc:
                attempts.append(f"{adapter.adapter_name}:{exc}")
                continue
            spec = ToolExecutionSpec(
                command=command,
                timeout_sec=int(metadata.get("tool_timeout_sec", 30)),
                retries=int(metadata.get("tool_retries", 0)),
                cwd=metadata.get("tool_cwd"),
                env={str(key): str(value) for key, value in dict(metadata.get("tool_env", {})).items()},
                acceptable_exit_codes=adapter.acceptable_exit_codes(mode=mode, metadata=metadata),
            )
            tool_result = self._tool_runner.run(spec)
            parsed = adapter.parse_output(
                execution_result=tool_result,
                target_hint=target_hint,
                mode=mode,
                metadata=metadata,
            )
            if tool_result.category == "command_not_found" and not metadata.get("probe_adapter") and not metadata.get("tool_command"):
                attempts.append(f"{adapter.adapter_name}:command_not_found")
                continue
            return {"adapter": adapter, "tool_result": tool_result, "parsed": parsed}

        parsed = ParsedProbeResult(
            summary=f"no probe adapter available for {mode} on {target_hint}",
            confidence=0.0,
            reachable=False,
            success=False,
            blocked=True,
            blocked_reason="no_supported_tool",
            evidence={"adapter_attempts": attempts},
            runtime_hints={"blocked_by": "tool_unavailable"},
        )
        return {
            "adapter": None,
            "tool_result": ToolExecutionResult(
                command=[],
                success=False,
                category="command_not_found",
                error_message="; ".join(attempts) or "no probe adapter available",
            ),
            "parsed": parsed,
        }

    def _build_worker_raw_result(
        self,
        *,
        task_id: str,
        result_type: str,
        execution: dict[str, Any],
        refs: list[GraphRef],
        metadata: dict[str, Any],
        operation_id: str,
        target_hint: str,
    ) -> dict[str, Any]:
        normalized = self._normalize_tool_payload(
            execution=execution,
            result_type=result_type,
            target_hint=target_hint,
            metadata=metadata,
        )
        base = self.build_raw_result(
            task_id=task_id,
            result_type=result_type,
            summary=str(normalized["summary"]),
            payload_ref=str(normalized["payload_ref"]),
            refs=refs,
            extra={
                "executor": "tool_runner",
                "operation_id": operation_id,
                "tool": normalized["tool"],
                "parsed": normalized["parsed"],
                "entities": normalized["parsed"].get("entities", []),
                "relations": normalized["parsed"].get("relations", []),
                "runtime_hints": normalized["parsed"].get("runtime_hints", {}),
                "kg_refs": self._filter_refs(refs, GraphScope.KG),
                "ag_refs": self._filter_refs(refs, GraphScope.AG),
            },
        )
        return base | {
            "success": bool(normalized["success"]),
            "confidence": float(normalized["confidence"]),
            "executor": "tool_runner",
            "tool": normalized["tool"],
            "parsed": normalized["parsed"],
        }

    def _normalize_tool_payload(
        self,
        *,
        execution: dict[str, Any],
        result_type: str,
        target_hint: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        tool_result = execution["tool_result"]
        adapter = execution.get("adapter")
        parsed_model: ParsedProbeResult = execution["parsed"]
        parsed = parsed_model.model_dump(mode="json")
        success = parsed_model.success
        summary = str(parsed_model.summary or f"{result_type} completed for {target_hint}")
        if parsed_model.blocked:
            summary = str(parsed_model.summary or f"{result_type} blocked for {target_hint}")
        elif not tool_result.success:
            summary = str(parsed_model.summary or f"{result_type} failed for {target_hint}")
        payload_ref = f"runtime://worker-results/{result_type}/{target_hint}"
        return {
            "success": success,
            "blocked": parsed_model.blocked,
            "summary": summary,
            "confidence": float(parsed_model.confidence or metadata.get("confidence", 0.8)),
            "payload_ref": payload_ref,
            "tool": tool_result.to_payload() | {
                "adapter": (adapter.adapter_name if adapter is not None else "unavailable")
            },
            "parsed": parsed,
        }

    @staticmethod
    def _compat_result_type(task_type: TaskType) -> str:
        mapping = {
            TaskType.ASSET_CONFIRMATION: "host_discovery_result",
            TaskType.SERVICE_VALIDATION: "service_validation_result",
            TaskType.REACHABILITY_VALIDATION: "service_validation_result",
            TaskType.IDENTITY_CONTEXT_CONFIRMATION: "identity_context_result",
        }
        return mapping.get(task_type, "host_discovery_result")

    def _candidate_adapters(self, *, metadata: dict[str, Any], mode: str) -> list[ProbeAdapter]:
        explicit = metadata.get("probe_adapter")
        if explicit is not None:
            adapter = self._probe_adapters.get(str(explicit))
            if adapter is None:
                raise ValueError(f"unknown probe_adapter '{explicit}'")
            return [adapter]
        if isinstance(metadata.get("tool_command"), list) or isinstance(metadata.get("custom_probe_command"), list):
            return [self._probe_adapters["custom"]]
        if mode in {"host_discovery", "host_discovery_result", "service_validation", "service_validation_result"}:
            return [
                self._probe_adapters["nmap"],
                self._probe_adapters["masscan"],
                self._probe_adapters["custom"],
            ]
        return [self._probe_adapters["custom"]]

    def _fact_writes_from_parsed(
        self,
        *,
        task_id: str,
        parsed: dict[str, Any],
        primary_ref: KGFactRef | None,
        evidence_id: str,
        confidence: float,
        task_type: str,
    ) -> list[FactWriteRequest]:
        requests: list[FactWriteRequest] = []
        evidence_ref = KGFactRef(graph="kg", ref_id=evidence_id, ref_type="Evidence")

        seen_entities: set[tuple[str, str]] = set()
        for entity in parsed.get("entities", []):
            if not isinstance(entity, dict):
                continue
            ref = self._entity_ref(entity, primary_ref=primary_ref)
            if ref is None:
                continue
            key = (ref.ref_id, ref.ref_type or "")
            if key in seen_entities:
                continue
            seen_entities.add(key)
            attributes = dict(entity)
            attributes.setdefault("task_type", task_type)
            attributes.setdefault("observed", True)
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=task_id,
                    subject_ref=ref,
                    attributes=attributes,
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Recon entity upsert for {ref.ref_id}",
                )
            )
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=task_id,
                    subject_ref=ref,
                    relation_type="SUPPORTED_BY",
                    object_ref=evidence_ref,
                    attributes={"task_type": task_type},
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Recon evidence supports {ref.ref_id}",
                )
            )

        seen_relations: set[tuple[str, str, str]] = set()
        for relation in parsed.get("relations", []):
            if not isinstance(relation, dict):
                continue
            relation_type = str(relation.get("type") or relation.get("relation_type") or "").upper()
            source_id = self._string(relation.get("source"))
            target_id = self._string(relation.get("target"))
            if not relation_type or not source_id or not target_id:
                continue
            key = (relation_type, source_id, target_id)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.RELATION_UPSERT,
                    source_task_id=task_id,
                    subject_ref=KGFactRef(graph="kg", ref_id=source_id, ref_type=self._relation_endpoint_type(source_id, primary_ref)),
                    relation_type=relation_type,
                    object_ref=KGFactRef(graph="kg", ref_id=target_id, ref_type=self._relation_target_type(relation, target_id)),
                    attributes=dict(relation.get("attributes", {})),
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Recon relation upsert {relation_type} {source_id}->{target_id}",
                )
            )

        if parsed.get("reachable") and primary_ref is not None and not any(
            request.relation_type == "CAN_REACH" for request in requests
        ):
            requests.append(
                FactWriteRequest(
                    kind=FactWriteKind.ENTITY_UPSERT,
                    source_task_id=task_id,
                    subject_ref=primary_ref,
                    attributes={
                        "reachable": True,
                        "task_type": task_type,
                        "runtime_hints": dict(parsed.get("runtime_hints", {})),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Recon reachability state for {primary_ref.ref_id}",
                )
            )

        return requests

    @staticmethod
    def _entity_ref(entity: dict[str, Any], *, primary_ref: KGFactRef | None) -> KGFactRef | None:
        entity_type = str(entity.get("type") or entity.get("entity_type") or "").strip()
        ref_id = (
            ReconWorker._string(entity.get("id"))
            or ReconWorker._string(entity.get("service_id"))
            or ReconWorker._string(entity.get("host_id"))
            or ReconWorker._string(entity.get("identity_id"))
        )
        if not ref_id and primary_ref is not None:
            ref_id = primary_ref.ref_id
            entity_type = entity_type or (primary_ref.ref_type or "")
        if not ref_id:
            return None
        return KGFactRef(graph="kg", ref_id=ref_id, ref_type=(entity_type or None))

    @staticmethod
    def _relation_target_type(relation: dict[str, Any], target_id: str) -> str | None:
        target_type = ReconWorker._string(relation.get("target_type"))
        if target_type is not None:
            return target_type
        lowered = target_id.lower()
        if ":" in target_id or lowered.startswith("svc-"):
            return "Service"
        return None

    @staticmethod
    def _relation_endpoint_type(source_id: str, primary_ref: KGFactRef | None) -> str | None:
        if primary_ref is not None and primary_ref.ref_id == source_id:
            return primary_ref.ref_type
        return "Host" if ":" not in source_id else "Service"

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _filter_refs(refs: list[GraphRef], scope: GraphScope) -> list[dict[str, Any]]:
        """Return refs for a specific graph scope as serialized payloads."""

        return [ref.model_dump(mode="json") for ref in refs if ref.graph == scope]

    @staticmethod
    def _primary_ref_id(refs: list[GraphRef], *, preferred_type: str) -> str | None:
        """Return the first ref id matching the preferred ref type."""

        for ref in refs:
            if (ref.ref_type or "").lower() == preferred_type.lower():
                return ref.ref_id
        return refs[0].ref_id if refs else None


__all__ = ["ReconWorker"]
