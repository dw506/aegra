"""Recon worker for host, service, and identity/context discovery tasks."""

from __future__ import annotations

import re
import shlex
from typing import Any
from urllib.parse import urlparse

from src.core.agents.agent_protocol import AgentInput, AgentOutput, GraphRef, GraphScope
from src.core.execution.adapters.local_shell_adapter import LocalShellAdapter
from src.core.execution.executor import ExecutionExecutor
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult as ExecutionToolResult
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
from src.core.graph.tg_builder import TaskCandidate
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.probe_adapters import (
    CustomProbeAdapter,
    HttpxFingerprintAdapter,
    MasscanAdapter,
    NmapAdapter,
    NucleiSafeTemplateAdapter,
    ParsedProbeResult,
    ProbeAdapter,
    ProbeAdapterUnavailable,
    SSLScanAdapter,
    WhatWebFingerprintAdapter,
)


class _LegacyProbeExecutionResult:
    """Compatibility shape consumed by existing probe parsers."""

    def __init__(
        self,
        *,
        command: list[str],
        attempts: int,
        success: bool,
        category: str,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        duration_sec: float,
        timed_out: bool,
        stdout_truncated: bool,
        stderr_truncated: bool,
        error_message: str | None,
    ) -> None:
        self.command = command
        self.attempts = attempts
        self.success = success
        self.category = category
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration_sec = duration_sec
        self.timed_out = timed_out
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated
        self.error_message = error_message

    def to_payload(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "attempts": self.attempts,
            "success": self.success,
            "category": self.category,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_sec": self.duration_sec,
            "timed_out": self.timed_out,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "error_message": self.error_message,
        }


class ReconWorker(BaseWorkerAgent):
    """Concrete worker for environment and context discovery operations."""

    capabilities = frozenset({WorkerCapability.RECON, WorkerCapability.CONTEXT_VALIDATION})
    agent_role = AgentRole.RECON_WORKER
    compatibility_task_types = frozenset(
        {
            TaskType.ASSET_CONFIRMATION,
            TaskType.SERVICE_VALIDATION,
            TaskType.WEB_ENUMERATION,
            TaskType.REACHABILITY_VALIDATION,
            TaskType.IDENTITY_CONTEXT_CONFIRMATION,
        }
    )
    supported_task_types = frozenset(
        {
            "host_discovery",
            "service_validation",
            "web_enumeration",
            "web_fingerprint",
            "identity_context_discovery",
            TaskType.ASSET_CONFIRMATION.value,
            TaskType.SERVICE_VALIDATION.value,
            TaskType.WEB_ENUMERATION.value,
            TaskType.REACHABILITY_VALIDATION.value,
            TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
        }
    )

    def __init__(
        self,
        name: str = "recon_worker",
        *,
        executor: ExecutionExecutor | None = None,
        tool_runner: Any | None = None,
        probe_adapters: list[ProbeAdapter] | None = None,
    ) -> None:
        super().__init__(name=name)
        _ = tool_runner
        self._executor = executor or ExecutionExecutor([LocalShellAdapter()])
        adapters = probe_adapters or [
            NmapAdapter(),
            MasscanAdapter(),
            HttpxFingerprintAdapter(),
            WhatWebFingerprintAdapter(),
            SSLScanAdapter(),
            NucleiSafeTemplateAdapter(),
            CustomProbeAdapter(),
        ]
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

        if task_spec.task_type in {"host_discovery", TaskType.ASSET_CONFIRMATION.value}:
            raw_result = self._execute_host_discovery(task_spec, agent_input)
        elif task_spec.task_type in {"service_validation", TaskType.SERVICE_VALIDATION.value, TaskType.REACHABILITY_VALIDATION.value}:
            raw_result = self._execute_service_validation(task_spec, agent_input)
        elif task_spec.task_type in {"web_enumeration", "web_fingerprint", TaskType.WEB_ENUMERATION.value}:
            raw_result = self._execute_web_enumeration(task_spec, agent_input)
        elif task_spec.task_type in {"identity_context_discovery", TaskType.IDENTITY_CONTEXT_CONFIRMATION.value}:
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
                "task_candidates": raw_result.get("task_candidates", []),
                "fact_write_requests": raw_result.get("fact_write_requests", []),
                "projection_requests": raw_result.get("projection_requests", []),
                "runtime_requests": raw_result.get("runtime_requests", []),
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
            metadata={"executor": normalized["executor"], "tool": normalized["tool"]},
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
                "executor": normalized["executor"],
                "tool": normalized["tool"],
                "parsed": normalized["parsed"],
            },
            error_message=(None if normalized["success"] else normalized["tool"].get("error_message")),
        )

    def _execute_host_discovery(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Execute host discovery through the execution layer and normalize the result."""

        canonical_host_id = self._primary_ref_id(task_spec.target_refs, preferred_type="Host")
        target_context = self._target_context_metadata(agent_input.raw_payload)
        host_hint = (
            task_spec.input_bindings.get("host")
            or task_spec.input_bindings.get("target_host")
            or task_spec.input_bindings.get("address")
            or task_spec.input_bindings.get("hostname")
            or target_context.get("address")
            or target_context.get("hostname")
            or self._host_from_url(target_context.get("url"))
            or self._primary_ref_label(task_spec.target_refs, preferred_type="Host")
            or task_spec.input_bindings.get("host_id")
            or canonical_host_id
            or "unknown-host"
        )
        metadata = dict(agent_input.raw_payload)
        self._apply_target_context_metadata(metadata, target_context)
        if canonical_host_id:
            metadata.setdefault("canonical_host_id", canonical_host_id)
            metadata.setdefault("host_id", canonical_host_id)
        execution = self._run_probe(
            task_id=task_spec.task_id,
            metadata=metadata,
            target_hint=host_hint,
            mode="host_discovery",
        )
        return self._build_worker_raw_result(
            task_id=task_spec.task_id,
            result_type="host_discovery_result",
            execution=execution,
            refs=task_spec.target_refs,
            metadata=metadata,
            operation_id=agent_input.context.operation_id,
            target_hint=host_hint,
        )

    def _execute_service_validation(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Execute service validation through the execution layer and normalize the result."""

        canonical_host_id = self._primary_ref_id(task_spec.target_refs, preferred_type="Host")
        canonical_service_id = self._primary_ref_id(task_spec.target_refs, preferred_type="Service")
        target_context = self._target_context_metadata(agent_input.raw_payload)
        target_host = (
            task_spec.input_bindings.get("host")
            or task_spec.input_bindings.get("target_host")
            or task_spec.input_bindings.get("address")
            or target_context.get("address")
            or target_context.get("hostname")
            or self._host_from_url(target_context.get("url"))
        )
        service_hint = (
            task_spec.input_bindings.get("service")
            or target_context.get("url")
            or target_host
            or task_spec.input_bindings.get("service_id")
            or canonical_service_id
            or "unknown-service"
        )
        port = task_spec.input_bindings.get("port") or task_spec.constraints.get("port") or target_context.get("port")
        metadata = dict(agent_input.raw_payload) | {"service_port": port}
        self._apply_target_context_metadata(metadata, target_context)
        if canonical_host_id:
            metadata.setdefault("canonical_host_id", canonical_host_id)
            metadata.setdefault("host_id", canonical_host_id)
        if canonical_service_id:
            metadata.setdefault("canonical_service_id", canonical_service_id)
            metadata.setdefault("service_id", canonical_service_id)
        execution = self._run_probe(
            task_id=task_spec.task_id,
            metadata=metadata,
            target_hint=service_hint,
            mode="service_validation",
        )
        return self._build_worker_raw_result(
            task_id=task_spec.task_id,
            result_type="service_validation_result",
            execution=execution,
            refs=task_spec.target_refs,
            metadata=metadata,
            operation_id=agent_input.context.operation_id,
            target_hint=service_hint,
        )

    def _execute_identity_context_discovery(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Execute identity/context discovery through the execution layer and normalize the result."""

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

    def _execute_web_enumeration(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Execute low-risk web fingerprinting/enumeration for a confirmed HTTP service."""

        canonical_host_id = self._primary_ref_id(task_spec.target_refs, preferred_type="Host")
        canonical_service_id = self._primary_ref_id(task_spec.target_refs, preferred_type="Service")
        target_url = (
            task_spec.input_bindings.get("target_url")
            or task_spec.constraints.get("target_url")
            or agent_input.raw_payload.get("target_url")
        )
        service_hint = (
            str(target_url)
            if target_url
            else str(task_spec.input_bindings.get("service_id") or canonical_service_id or "unknown-web-service")
        )
        metadata = dict(agent_input.raw_payload) | {
            "target_url": target_url,
            "port": task_spec.input_bindings.get("port") or task_spec.constraints.get("port"),
            "protocol": task_spec.input_bindings.get("protocol") or task_spec.constraints.get("protocol"),
        }
        if canonical_host_id:
            metadata.setdefault("canonical_host_id", canonical_host_id)
            metadata.setdefault("host_id", canonical_host_id)
        if canonical_service_id:
            metadata.setdefault("canonical_service_id", canonical_service_id)
            metadata.setdefault("service_id", canonical_service_id)
        execution = self._run_probe(
            task_id=task_spec.task_id,
            metadata=metadata,
            target_hint=service_hint,
            mode="web_enumeration",
        )
        raw_result = self._build_worker_raw_result(
            task_id=task_spec.task_id,
            result_type="web_enumeration_result",
            execution=execution,
            refs=task_spec.target_refs,
            metadata=metadata,
            operation_id=agent_input.context.operation_id,
            target_hint=service_hint,
        )
        raw_result["task_candidates"] = self._web_followup_task_candidates(
            task_id=task_spec.task_id,
            parsed=raw_result.get("parsed", {}),
            refs=task_spec.target_refs,
            input_bindings=task_spec.input_bindings,
        )
        return raw_result

    @staticmethod
    def _target_context_metadata(raw_payload: dict[str, Any]) -> dict[str, Any]:
        contexts = raw_payload.get("target_context")
        if not isinstance(contexts, list):
            return {}
        for item in contexts:
            if not isinstance(item, dict):
                continue
            properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
            merged = dict(properties) | {key: value for key, value in item.items() if value is not None}
            if merged.get("address") or merged.get("hostname") or merged.get("url") or merged.get("port"):
                return merged
        return {}

    @staticmethod
    def _host_from_url(value: Any) -> str | None:
        if value is None:
            return None
        parsed = urlparse(str(value))
        return parsed.hostname or None

    @classmethod
    def _apply_target_context_metadata(cls, metadata: dict[str, Any], target_context: dict[str, Any]) -> None:
        if not target_context:
            return
        address = target_context.get("address")
        hostname = target_context.get("hostname")
        target_url = target_context.get("url")
        port = target_context.get("port")
        scheme = target_context.get("scheme")
        host = address or hostname or cls._host_from_url(target_url)
        if host:
            metadata.setdefault("target_host", host)
            metadata.setdefault("address", host)
        if target_url:
            metadata.setdefault("target_url", target_url)
        if port is not None:
            metadata.setdefault("port", port)
            metadata.setdefault("target_port", port)
        if scheme:
            metadata.setdefault("scheme", scheme)

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
            plan = ToolPlan(
                task_id=task_id,
                tool=adapter.adapter_name,
                adapter=self._execution_adapter(metadata),
                command=shlex.join(command),
                target=target_hint,
                args={
                    "argv": command,
                    "cwd": metadata.get("tool_cwd"),
                    "env": {str(key): str(value) for key, value in dict(metadata.get("tool_env", {})).items()},
                    "env_allowlist": list(metadata.get("tool_env_allowlist", []))
                    if isinstance(metadata.get("tool_env_allowlist", []), list)
                    else [],
                    "command_allowlist": list(metadata.get("command_allowlist", []))
                    if isinstance(metadata.get("command_allowlist", []), list)
                    else [],
                    "acceptable_exit_codes": sorted(adapter.acceptable_exit_codes(mode=mode, metadata=metadata)),
                    "stdout_max_bytes": int(metadata.get("stdout_max_bytes", metadata.get("tool_stdout_max_bytes", 262144))),
                    "stderr_max_bytes": int(metadata.get("stderr_max_bytes", metadata.get("tool_stderr_max_bytes", 65536))),
                },
                timeout_seconds=int(metadata.get("tool_timeout_sec", 30)),
                metadata={
                    "task_type": mode,
                    "probe_adapter": adapter.adapter_name,
                    "policy_metadata": {
                        "kind": adapter.adapter_name,
                        "name": adapter.adapter_name,
                        "operation": mode,
                        "tags": metadata.get("tool_tags", ["safe_probe", "fingerprint"]),
                    },
                    "tool_isolation": dict(metadata.get("tool_isolation", {}))
                    if isinstance(metadata.get("tool_isolation"), dict)
                    else {},
                },
            )
            execution_result = self._executor.execute(plan)
            tool_result = self._legacy_tool_result(execution_result)
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
            "tool_result": _LegacyProbeExecutionResult(
                command=[],
                attempts=1,
                success=False,
                category="command_not_found",
                exit_code=None,
                stdout="",
                stderr="",
                duration_sec=0.0,
                timed_out=False,
                stdout_truncated=False,
                stderr_truncated=False,
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
                "executor": "execution_executor",
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
        evidence_id = f"evidence::{task_id}"
        primary_ref = self._primary_kg_fact_ref(refs)
        fact_write_requests = self._fact_writes_from_parsed(
            task_id=task_id,
            parsed=normalized["parsed"],
            primary_ref=primary_ref,
            evidence_id=evidence_id,
            confidence=float(normalized["confidence"]),
            task_type=result_type,
        )
        projection_requests = [
            ProjectionRequest(
                kind=ProjectionRequestKind.REFRESH_LOCAL_FRONTIER,
                source_task_id=task_id,
                reason="recon produced graph facts for local frontier refresh",
                target_refs=[primary_ref] if primary_ref is not None else [],
            ).model_dump(mode="json")
        ]
        runtime_requests = [
            RuntimeControlRequest(
                request_type=RuntimeControlType.CONSUME_BUDGET,
                source_task_id=task_id,
                budget_delta=RuntimeBudgetDelta(operations=1, noise=float(metadata.get("noise_cost", 0.1))),
                reason="recon consumes one worker execution budget unit",
            ).model_dump(mode="json")
        ]
        return base | {
            "evidence_id": evidence_id,
            "success": bool(normalized["success"]),
            "confidence": float(normalized["confidence"]),
            "executor": "execution_executor",
            "tool": normalized["tool"],
            "parsed": normalized["parsed"],
            "fact_write_requests": [request.model_dump(mode="json") for request in fact_write_requests],
            "projection_requests": projection_requests,
            "runtime_requests": runtime_requests,
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
            "executor": "execution_executor",
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
            TaskType.WEB_ENUMERATION: "web_enumeration_result",
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
        if mode in {"web_enumeration", "web_enumeration_result", "web_fingerprint", "web_fingerprint_result"}:
            return [
                adapter
                for name in ("httpx", "whatweb", "custom")
                if (adapter := self._probe_adapters.get(name)) is not None
            ]
        return [self._probe_adapters["custom"]]

    @staticmethod
    def _execution_adapter(metadata: dict[str, Any]) -> str | None:
        value = metadata.get("execution_adapter") or metadata.get("adapter")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _legacy_tool_result(result: ExecutionToolResult) -> _LegacyProbeExecutionResult:
        metadata = dict(result.metadata)
        category = str(metadata.get("category") or ("success" if result.success else "nonzero_exit"))
        error_message = metadata.get("error_message")
        if error_message is not None:
            error_message = str(error_message)
        exit_code = result.exit_code if isinstance(result.exit_code, int) else None
        return _LegacyProbeExecutionResult(
            command=[str(item) for item in metadata.get("command", [])] if isinstance(metadata.get("command"), list) else [],
            attempts=int(metadata.get("attempts", 1)),
            success=result.success,
            category=category,
            exit_code=exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_sec=float(metadata.get("duration_sec", 0.0)),
            timed_out=bool(metadata.get("timed_out", category == "timeout")),
            stdout_truncated=bool(metadata.get("stdout_truncated", False)),
            stderr_truncated=bool(metadata.get("stderr_truncated", False)),
            error_message=error_message,
        )

    @staticmethod
    def _web_followup_task_candidates(
        *,
        task_id: str,
        parsed: dict[str, Any],
        refs: list[GraphRef],
        input_bindings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        service = parsed.get("service") if isinstance(parsed.get("service"), dict) else {}
        hints = parsed.get("runtime_hints") if isinstance(parsed.get("runtime_hints"), dict) else {}
        target_url = service.get("target_url") or hints.get("target_url") or input_bindings.get("target_url")
        service_id = str(service.get("id") or service.get("service_id") or input_bindings.get("service_id") or "")
        if not service_id or not target_url:
            return []
        host_id = str(service.get("host_id") or input_bindings.get("host_id") or "")
        resource_keys = {f"service:{service_id}"}
        if host_id:
            resource_keys.add(f"host:{host_id}")
        target_refs = []
        for ref in refs:
            payload = ref.model_dump(mode="json") if hasattr(ref, "model_dump") else dict(ref)
            if payload.get("graph") not in {"kg", "ag", "tg", "query"}:
                continue
            payload.pop("metadata", None)
            target_refs.append(payload)
        candidates: list[dict[str, Any]] = []
        default_credentials = ReconWorker._default_credentials_from_web_result(parsed)
        if default_credentials:
            username, password = default_credentials
            credential_candidate = TaskCandidate(
                source_action_id=f"default-credentials-idor::{task_id}::{service_id}",
                task_type=TaskType.VULNERABILITY_VALIDATION,
                input_bindings={
                    "host_id": host_id,
                    "service_id": service_id,
                    "port": service.get("port") or input_bindings.get("port"),
                    "protocol": service.get("protocol") or input_bindings.get("protocol") or "http",
                    "target_url": target_url,
                    "service_name": service.get("service_name") or "http",
                    "validator_id": "default-credentials-idor",
                    "vulnerability_id": "vuln-xbow-validation",
                    "vulnerability_name": "XBOW benchmark vulnerability: idor, default_credentials",
                    "username": username,
                    "password": password,
                },
                target_refs=target_refs,
                estimated_cost=0.28,
                estimated_risk=0.08,
                estimated_noise=0.08,
                goal_relevance=0.98,
                resource_keys=resource_keys,
                approval_required=False,
                parallelizable=False,
                tags={"default_credentials", "idor", "safe_http_client"},
            )
            candidates.append(credential_candidate.model_dump(mode="json"))
            return candidates

        candidate = TaskCandidate(
            source_action_id=f"web-fingerprint::{task_id}::{service_id}",
            task_type=TaskType.VULNERABILITY_VALIDATION,
            input_bindings={
                "host_id": host_id,
                "service_id": service_id,
                "port": service.get("port") or input_bindings.get("port"),
                "protocol": service.get("protocol") or input_bindings.get("protocol") or "http",
                "target_url": target_url,
                "service_name": service.get("service_name") or "http",
                "validator_id": "http-fingerprint",
                "vulnerability_id": f"vuln::http-fingerprint::{service_id}",
                "vulnerability_name": "HTTP Fingerprint",
                "http_method": "GET",
            },
            target_refs=target_refs,
            estimated_cost=0.18,
            estimated_risk=0.04,
            estimated_noise=0.05,
            goal_relevance=0.8,
            resource_keys=resource_keys,
            approval_required=False,
            parallelizable=True,
            tags={"web_fingerprint", "http_fingerprint"},
        )
        candidates.append(candidate.model_dump(mode="json"))
        return candidates

    @staticmethod
    def _default_credentials_from_web_result(parsed: dict[str, Any]) -> tuple[str, str] | None:
        evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), dict) else {}
        text_parts = [
            evidence.get("body_excerpt"),
            evidence.get("stdout"),
            parsed.get("raw_output"),
        ]
        tool = evidence.get("tool") if isinstance(evidence.get("tool"), dict) else {}
        text_parts.append(tool.get("stdout"))
        text = "\n".join(str(item) for item in text_parts if item)
        if not text:
            return None

        patterns = [
            r"(?:account|credential|creds?|login)[^A-Za-z0-9]{0,40}\(([A-Za-z0-9_.@-]{1,64}):([^\s:)<]{1,64})\)",
            r"\b([A-Za-z0-9_.@-]{1,64}):([A-Za-z0-9_.@-]{1,64})\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                username, password = match.group(1).strip(), match.group(2).strip()
                if username.lower() in {"http", "https", "cache-control"}:
                    continue
                return username, password
        return None

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
    def _primary_kg_fact_ref(refs: list[GraphRef]) -> KGFactRef | None:
        """Return the first KG host/service ref as a fact-write compatible ref."""

        for preferred_type in ("Host", "Service"):
            for ref in refs:
                if ref.graph == GraphScope.KG and (ref.ref_type or "").lower() == preferred_type.lower():
                    return KGFactRef(
                        graph="kg",
                        ref_id=ref.ref_id,
                        ref_type=ref.ref_type,
                        label=getattr(ref, "label", None),
                    )
        for ref in refs:
            if ref.graph == GraphScope.KG:
                return KGFactRef(
                    graph="kg",
                    ref_id=ref.ref_id,
                    ref_type=ref.ref_type,
                    label=getattr(ref, "label", None),
                )
        return None

    @staticmethod
    def _primary_ref_id(refs: list[GraphRef], *, preferred_type: str) -> str | None:
        """Return the first ref id matching the preferred ref type."""

        for ref in refs:
            if (ref.ref_type or "").lower() == preferred_type.lower():
                return ref.ref_id
        return refs[0].ref_id if refs else None

    @staticmethod
    def _primary_ref_label(refs: list[GraphRef], *, preferred_type: str) -> str | None:
        """Return the label for the first ref matching the preferred ref type."""

        for ref in refs:
            label = getattr(ref, "label", None)
            if (ref.ref_type or "").lower() == preferred_type.lower() and label:
                return str(label)
        for ref in refs:
            label = getattr(ref, "label", None)
            if label:
                return str(label)
        return None


__all__ = ["ReconWorker"]
