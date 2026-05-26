"""Internal service fingerprint primary worker."""

from __future__ import annotations

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.fingerprint_worker import FingerprintWorker


class InternalServiceFingerprintWorker(BaseWorkerAgent):
    """Fingerprint services reachable through an internal route or session."""

    capabilities = frozenset({WorkerCapability.INTERNAL_SERVICE_FINGERPRINT, WorkerCapability.FINGERPRINT})
    supported_task_types = frozenset(
        {TaskType.INTERNAL_SERVICE_FINGERPRINT.value, "internal_service_fingerprint", "internal_fingerprint"}
    )

    def __init__(self, name: str = "internal_service_fingerprint_worker", fingerprint_worker: FingerprintWorker | None = None) -> None:
        super().__init__(name=name)
        self._fingerprint = fingerprint_worker or FingerprintWorker(name=f"{name}_fingerprint")

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        route_id = (
            task_spec.input_bindings.get("route_id")
            or task_spec.input_bindings.get("selected_route_id")
            or task_spec.constraints.get("route_id")
        )
        selected_route = task_spec.input_bindings.get("selected_route")
        session_id = task_spec.input_bindings.get("session_id") or task_spec.constraints.get("session_id")
        if not route_id and not selected_route and not session_id:
            summary = "internal service fingerprint requires route_id, selected_route, or session_id"
            outcome = self.build_outcome(
                task_id=task_spec.task_id,
                outcome_type="internal_service_fingerprint",
                success=False,
                summary=summary,
                confidence=0.0,
                refs=task_spec.target_refs,
                payload={"blocked": True, "blocked_on": "route"},
            )
            return AgentOutput(outcomes=[outcome.to_agent_output_fragment()], errors=[summary], logs=[summary])

        fp_spec = task_spec.model_copy(update={"task_type": "service_fingerprint"})
        output = self._fingerprint.execute_task(fp_spec, agent_input)
        for outcome in output.outcomes:
            outcome["source_agent"] = self.name
            outcome["task_id"] = task_spec.task_id
            outcome["outcome_type"] = "internal_service_fingerprint"
            payload = outcome.setdefault("payload", {})
            if isinstance(payload, dict):
                payload["task_type"] = TaskType.INTERNAL_SERVICE_FINGERPRINT.value
                payload["route_id"] = route_id
                payload["session_id"] = session_id
        for evidence in output.evidence:
            evidence["source_agent"] = self.name
            evidence["task_id"] = task_spec.task_id
            evidence["result_type"] = "internal_service_fingerprint"
        output.logs.insert(0, f"worker={self.name}")
        return output


__all__ = ["InternalServiceFingerprintWorker"]
