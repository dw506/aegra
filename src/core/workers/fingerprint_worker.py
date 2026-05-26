"""Fingerprint worker that normalizes recon output and emits vulnerability candidates."""

from __future__ import annotations

import re
from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import GraphRef as KGFactRef
from src.core.models.events import (
    AgentExecutionContext,
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    AgentTaskResult,
    EvidenceArtifact,
    FactWriteKind,
    FactWriteRequest,
    ObservationRecord,
    ProjectionRequest,
    ProjectionRequestKind,
)
from src.core.models.fingerprint import ComponentFingerprint, ServiceFingerprint, TechnologyStackFingerprint
from src.core.models.runtime import TaskRuntime
from src.core.models.tg import BaseTaskNode, TaskType
from src.core.models.vulnerability_candidate import VulnerabilityCandidate
from src.core.vuln_candidates.matcher import CandidateMatcher
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec


class FingerprintWorker(BaseWorkerAgent):
    """Normalize recon evidence into service fingerprints and candidate validation tasks."""

    capabilities = frozenset({WorkerCapability.FINGERPRINT, WorkerCapability.CONTEXT_VALIDATION})
    agent_role = AgentRole.FINGERPRINT_WORKER
    compatibility_task_types = frozenset({TaskType.SERVICE_VALIDATION})
    supported_task_types = frozenset({"fingerprint", "service_fingerprint"})

    def __init__(self, name: str = "fingerprint_worker", *, matcher: CandidateMatcher | None = None) -> None:
        super().__init__(name=name)
        self._matcher = matcher or CandidateMatcher()

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
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
        runtime = task_runtime or TaskRuntime(task_id=task.id, tg_node_id=task.id)
        return AgentTaskRequest(
            agent_role=self.agent_role,
            intent=AgentTaskIntent.COLLECT_EVIDENCE,
            context=AgentExecutionContext(
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
            ),
            task_label=task.label,
            input_bindings=dict(task.input_bindings),
            target_refs=list(task.target_refs),
            source_refs=list(task.source_refs),
            expected_output_refs=list(task.expected_output_refs),
            metadata=metadata or {},
        )

    def execute_task(
        self,
        task_spec: WorkerTaskSpec | AgentTaskRequest,
        agent_input: AgentInput | None = None,
    ) -> AgentOutput | AgentTaskResult:
        if isinstance(task_spec, AgentTaskRequest):
            return self._execute_request(task_spec)
        if agent_input is None:
            raise ValueError("agent_input is required when executing a worker task spec")
        request = AgentTaskRequest(
            agent_role=self.agent_role,
            intent=AgentTaskIntent.COLLECT_EVIDENCE,
            context=AgentExecutionContext(
                operation_id=agent_input.context.operation_id,
                task_id=task_spec.task_id,
                tg_node_id=str(agent_input.raw_payload.get("tg_node_id") or task_spec.task_id),
                task_type=TaskType.SERVICE_VALIDATION,
                resource_keys=set(task_spec.resource_keys),
            ),
            task_label=str(agent_input.raw_payload.get("task_label") or "Normalize service fingerprint"),
            input_bindings=dict(task_spec.input_bindings),
            target_refs=[
                KGFactRef.model_validate(_strip_graph_ref_metadata(ref.model_dump(mode="json")))
                for ref in task_spec.target_refs
            ],
            metadata=dict(agent_input.raw_payload) | dict(task_spec.constraints),
        )
        result = self._execute_request(request)
        return AgentOutput(
            observations=[item.model_dump(mode="json") for item in result.observations],
            evidence=[
                {
                    "task_id": result.task_id,
                    "source_agent": self.name,
                    "result_type": item.kind,
                    "summary": item.summary,
                    "payload_ref": item.payload_ref,
                    "refs": [ref.model_dump(mode="json") for ref in item.refs],
                    "extra": dict(item.metadata),
                }
                for item in result.evidence
            ],
            outcomes=[
                self.build_outcome(
                    task_id=task_spec.task_id,
                    outcome_type="service_fingerprint",
                    success=result.status == AgentResultStatus.SUCCEEDED,
                    summary=result.summary,
                    raw_result_ref=(result.evidence[0].payload_ref if result.evidence else None),
                    confidence=float(result.outcome_payload.get("confidence", 0.0)),
                    refs=task_spec.target_refs,
                    payload=result.outcome_payload,
                ).to_agent_output_fragment()
            ],
            logs=[result.summary],
            errors=([result.error_message] if result.error_message else []),
        )

    def _execute_request(self, request: AgentTaskRequest) -> AgentTaskResult:
        if request.agent_role != self.agent_role:
            return self._failure_result(request, f"request agent_role {request.agent_role.value} does not match {self.agent_role.value}")
        try:
            fingerprints = self.extract_fingerprints(request)
            candidates = [candidate for fingerprint in fingerprints for candidate in self._matcher.match(fingerprint)]
        except Exception as exc:
            return self._failure_result(request, f"fingerprint extraction failed: {exc}")

        task_candidates = [candidate.to_task_candidate() for candidate in candidates]
        refs = list(request.target_refs)
        confidence = max([fingerprint.confidence for fingerprint in fingerprints], default=0.0)
        summary = f"generated {len(fingerprints)} service fingerprint(s) and {len(candidates)} vulnerability candidate(s)"
        evidence = EvidenceArtifact(
            kind="service_fingerprint",
            summary=summary,
            payload_ref=f"runtime://worker-results/fingerprint/{request.context.task_id}",
            refs=refs,
            metadata={
                "fingerprints": [fingerprint.model_dump(mode="json") for fingerprint in fingerprints],
                "vulnerability_candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "task_candidates": [candidate.model_dump(mode="json") for candidate in task_candidates],
            },
        )
        observation = ObservationRecord(
            category="service_fingerprint",
            summary=summary,
            confidence=confidence,
            refs=refs,
            payload=evidence.metadata,
        )
        fact_writes = self._fact_writes(
            request=request,
            evidence_id=evidence.evidence_id,
            fingerprints=fingerprints,
            candidates=candidates,
            confidence=confidence,
        )
        return AgentTaskResult(
            request_id=request.request_id,
            agent_role=self.agent_role,
            operation_id=request.context.operation_id,
            task_id=request.context.task_id,
            tg_node_id=request.context.tg_node_id,
            status=AgentResultStatus.SUCCEEDED,
            summary=summary,
            observations=[observation],
            evidence=[evidence],
            fact_write_requests=fact_writes,
            projection_requests=[
                ProjectionRequest(
                    kind=ProjectionRequestKind.REFRESH_LOCAL_FRONTIER,
                    source_task_id=request.context.task_id,
                    reason="fingerprint candidates may add vulnerability validation tasks",
                    target_refs=refs,
                )
            ],
            task_candidate_proposals=[
                proposal
                for candidate in task_candidates
                for proposal in []
            ],
            outcome_payload={
                "outcome_type": "service_fingerprint",
                "confidence": confidence,
                "fingerprints": [fingerprint.model_dump(mode="json") for fingerprint in fingerprints],
                "vulnerability_candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "task_candidates": [candidate.model_dump(mode="json") for candidate in task_candidates],
            },
            metadata={"vulnerability_candidates": [candidate.model_dump(mode="json") for candidate in candidates]},
        )

    def extract_fingerprints(self, request: AgentTaskRequest) -> list[ServiceFingerprint]:
        parsed_items = self._parsed_items(request.metadata)
        if not parsed_items:
            parsed_items = [{"service": dict(request.input_bindings)}]
        fingerprints: list[ServiceFingerprint] = []
        for parsed in parsed_items:
            services = self._service_payloads(parsed, fallback=dict(request.input_bindings))
            for service in services:
                fingerprints.append(self._fingerprint_from_service(service, parsed=parsed, request=request))
        return fingerprints

    @classmethod
    def _parsed_items(cls, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for key in ("parsed", "recon_parsed"):
            if isinstance(metadata.get(key), dict):
                result.append(dict(metadata[key]))
        for key in ("recon_result", "agent_task_result"):
            value = metadata.get(key)
            if isinstance(value, dict):
                parsed = value.get("parsed") or value.get("outcome_payload", {}).get("parsed")
                if isinstance(parsed, dict):
                    result.append(dict(parsed))
        for item in metadata.get("evidence", []) if isinstance(metadata.get("evidence"), list) else []:
            if not isinstance(item, dict):
                continue
            extra = item.get("extra") if isinstance(item.get("extra"), dict) else item.get("metadata")
            if isinstance(extra, dict) and isinstance(extra.get("parsed"), dict):
                result.append(dict(extra["parsed"]))
        return result

    @staticmethod
    def _service_payloads(parsed: dict[str, Any], *, fallback: dict[str, Any]) -> list[dict[str, Any]]:
        services: list[dict[str, Any]] = []
        if isinstance(parsed.get("service"), dict) and parsed["service"]:
            services.append(dict(parsed["service"]))
        for item in parsed.get("entities", []) if isinstance(parsed.get("entities"), list) else []:
            if isinstance(item, dict) and str(item.get("type") or item.get("entity_type") or "").lower() == "service":
                services.append(dict(item))
        return services or [fallback]

    def _fingerprint_from_service(
        self,
        service: dict[str, Any],
        *,
        parsed: dict[str, Any],
        request: AgentTaskRequest,
    ) -> ServiceFingerprint:
        host = self._string(service.get("host") or service.get("host_id") or request.input_bindings.get("host_id"))
        port = self._int(service.get("port") or request.input_bindings.get("port"))
        protocol = self._string(service.get("protocol") or request.input_bindings.get("protocol"))
        service_name = self._string(service.get("service_name") or service.get("name") or request.input_bindings.get("service_name"))
        service_id = self._string(service.get("id") or service.get("service_id") or request.input_bindings.get("service_id"))
        if service_id is None:
            service_id = f"{host or 'unknown'}:{port or 'unknown'}/{protocol or 'tcp'}"
        headers = self._headers(service.get("headers") or parsed.get("headers") or service.get("http_headers"))
        body_signals = self._string_list(service.get("body_signals") or parsed.get("body_signals"))
        banner = self._string(service.get("banner"))
        if banner:
            body_signals.append(banner)
        http_title = self._string(service.get("http_title") or parsed.get("http_title") or self._title_from_body_signals(body_signals))
        product = self._string(service.get("product") or self._product_from_banner(banner))
        version = self._string(service.get("version") or self._version_from_banner(banner))
        cpe = self._string_list(service.get("cpe") or service.get("cpes") or parsed.get("cpe"))
        components = self._components(service=service, product=product, version=version, cpe=cpe, headers=headers, body_signals=body_signals)
        tags = sorted({item.lower() for item in [service_name, product, *body_signals, *headers.values()] if item})
        confidence = float(service.get("confidence") or parsed.get("confidence") or request.metadata.get("confidence") or 0.7)
        return ServiceFingerprint(
            service_id=service_id,
            host=host,
            port=port,
            protocol=protocol,
            service_name=service_name,
            product=product,
            version=version,
            http_title=http_title,
            headers=headers,
            body_signals=body_signals,
            cpe=cpe,
            stack=TechnologyStackFingerprint(components=components, tags=tags, confidence=confidence),
            confidence=confidence,
            metadata={
                "target_url": request.metadata.get("target_url") or service.get("target_url"),
                "source_task_id": request.context.task_id,
                "raw_service": service,
            },
        )

    @staticmethod
    def _components(
        *,
        service: dict[str, Any],
        product: str | None,
        version: str | None,
        cpe: list[str],
        headers: dict[str, str],
        body_signals: list[str],
    ) -> list[ComponentFingerprint]:
        components: list[ComponentFingerprint] = []
        if product or service.get("service_name"):
            components.append(
                ComponentFingerprint(
                    name=str(product or service.get("service_name")),
                    product=product,
                    version=version,
                    cpe=(cpe[0] if cpe else None),
                    confidence=float(service.get("confidence") or 0.7),
                    evidence={"headers": headers, "body_signals": body_signals},
                )
            )
        for key, value in headers.items():
            if key.lower() in {"server", "x-powered-by"}:
                components.append(
                    ComponentFingerprint(
                        name=value,
                        product=value.split("/")[0].strip() or value,
                        version=(value.split("/", 1)[1].split()[0] if "/" in value else None),
                        confidence=0.65,
                        evidence={"header": key},
                    )
                )
        return components

    def _fact_writes(
        self,
        *,
        request: AgentTaskRequest,
        evidence_id: str,
        fingerprints: list[ServiceFingerprint],
        candidates: list[VulnerabilityCandidate],
        confidence: float,
    ) -> list[FactWriteRequest]:
        writes: list[FactWriteRequest] = []
        candidates_by_service: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            candidates_by_service.setdefault(candidate.service_id, []).append(candidate.model_dump(mode="json"))
        for fingerprint in fingerprints:
            service_ref = KGFactRef(graph="kg", ref_id=fingerprint.service_id, ref_type="Service", label=fingerprint.service_id)
            writes.append(
                FactWriteRequest(
                    kind=FactWriteKind.ASSERTION,
                    source_task_id=request.context.task_id,
                    subject_ref=service_ref,
                    attributes={
                        "service_fingerprint": fingerprint.model_dump(mode="json"),
                        "vulnerability_candidates": candidates_by_service.get(fingerprint.service_id, []),
                    },
                    confidence=confidence,
                    evidence_ids=[evidence_id],
                    summary=f"Service fingerprint and vulnerability candidates for {fingerprint.service_id}",
                )
            )
        return writes

    def _failure_result(self, request: AgentTaskRequest, message: str) -> AgentTaskResult:
        return AgentTaskResult(
            request_id=request.request_id,
            agent_role=self.agent_role,
            operation_id=request.context.operation_id,
            task_id=request.context.task_id,
            tg_node_id=request.context.tg_node_id,
            status=AgentResultStatus.FAILED,
            summary=message,
            error_message=message,
        )

    @staticmethod
    def _headers(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items()}

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [str(item) for item in items if str(item).strip()]

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _product_from_banner(banner: str | None) -> str | None:
        if not banner:
            return None
        return banner.split("/", 1)[0].split()[0]

    @staticmethod
    def _version_from_banner(banner: str | None) -> str | None:
        if not banner or "/" not in banner:
            return None
        return banner.split("/", 1)[1].split()[0]

    @staticmethod
    def _title_from_body_signals(signals: list[str]) -> str | None:
        text = " ".join(signals)
        match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None


__all__ = ["FingerprintWorker"]


def _strip_graph_ref_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.pop("metadata", None)
    return payload
