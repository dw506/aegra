"""Controlled port scan primary worker."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.recon_worker import ReconWorker


class PortScanWorker(BaseWorkerAgent):
    """Primary worker for bounded host port discovery."""

    capabilities = frozenset({WorkerCapability.PORT_SCAN, WorkerCapability.RECON})
    supported_task_types = frozenset({TaskType.PORT_SCAN.value, "port_scan", "scan_ports"})

    def __init__(self, name: str = "port_scan_worker", *, recon_worker: ReconWorker | None = None) -> None:
        super().__init__(name=name)
        self._recon = recon_worker or ReconWorker(name=f"{name}_recon")

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        bindings = dict(task_spec.input_bindings)
        target = (
            bindings.get("target_host")
            or bindings.get("target_address")
            or bindings.get("host")
            or bindings.get("host_id")
            or bindings.get("target_cidr")
        )
        if not target:
            return self._blocked(task_spec, "port scan requires target_host, target_address, host_id, or target_cidr")

        ports = bindings.get("ports") or bindings.get("port")
        scan_spec = task_spec.model_copy(
            update={
                "task_type": "service_validation" if ports else "host_discovery",
                "input_bindings": {
                    **bindings,
                    "host": str(target),
                    "target_host": str(target),
                    "service": str(target),
                    **({"port": ports} if ports is not None else {}),
                },
            }
        )
        payload = dict(agent_input.raw_payload)
        payload.setdefault("tool_tags", ["safe_probe", "port_scan"])
        if ports is not None:
            payload.setdefault("port", ports)
            payload.setdefault("service_port", ports)
        output = self._recon.execute_task(
            scan_spec,
            agent_input.model_copy(update={"raw_payload": payload}),
        )
        return self._retag_output(output, task_spec=task_spec, outcome_type="port_scan")

    def _blocked(self, task_spec: WorkerTaskSpec, summary: str) -> AgentOutput:
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type="port_scan",
            success=False,
            summary=summary,
            confidence=0.0,
            refs=task_spec.target_refs,
            payload={"blocked": True, "blocked_on": "target"},
        )
        return AgentOutput(outcomes=[outcome.to_agent_output_fragment()], errors=[summary], logs=[summary])

    def _retag_output(self, output: AgentOutput, *, task_spec: WorkerTaskSpec, outcome_type: str) -> AgentOutput:
        for outcome in output.outcomes:
            outcome["source_agent"] = self.name
            outcome["task_id"] = task_spec.task_id
            outcome["outcome_type"] = outcome_type
            payload = outcome.setdefault("payload", {})
            if isinstance(payload, dict):
                payload["task_type"] = TaskType.PORT_SCAN.value
        for evidence in output.evidence:
            evidence["source_agent"] = self.name
            evidence["task_id"] = task_spec.task_id
            evidence["result_type"] = "port_scan_result"
            extra = evidence.setdefault("extra", {})
            if isinstance(extra, dict):
                parsed = extra.get("parsed") if isinstance(extra.get("parsed"), dict) else {}
                services = [
                    item
                    for item in parsed.get("entities", [])
                    if isinstance(item, dict) and str(item.get("type", "")).lower() == "service"
                ]
                extra["task_candidates"] = self._fingerprint_candidates(task_spec=task_spec, services=services)
        output.logs.insert(0, f"worker={self.name}")
        return output

    @staticmethod
    def _fingerprint_candidates(*, task_spec: WorkerTaskSpec, services: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for service in services:
            service_id = str(service.get("id") or service.get("service_id") or "")
            if not service_id:
                continue
            host_id = str(service.get("host_id") or task_spec.input_bindings.get("host_id") or "")
            candidates.append(
                {
                    "source_action_id": f"port-scan::{task_spec.task_id}::{service_id}",
                    "task_type": TaskType.INTERNAL_SERVICE_FINGERPRINT.value,
                    "input_bindings": {
                        "host_id": host_id,
                        "service_id": service_id,
                        "port": service.get("port"),
                        "protocol": service.get("protocol"),
                        "service_name": service.get("service_name"),
                    },
                    "resource_keys": [key for key in (f"host:{host_id}" if host_id else "", f"service:{service_id}") if key],
                    "tags": ["fingerprint", "internal_service"],
                }
            )
        return candidates


__all__ = ["PortScanWorker"]
