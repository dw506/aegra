"""Lateral reachability validation primary worker."""

from __future__ import annotations

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.services.pivot_validation_service import PivotValidationRequest, PivotValidationService


class LateralReachabilityWorker(BaseWorkerAgent):
    """Validate host-to-host reachability through a selected route or session."""

    capabilities = frozenset({WorkerCapability.LATERAL_REACHABILITY_VALIDATION, WorkerCapability.ACCESS_VALIDATION})
    supported_task_types = frozenset(
        {TaskType.LATERAL_REACHABILITY_VALIDATION.value, "lateral_reachability_validation", "validate_lateral_reachability"}
    )

    def __init__(self, name: str = "lateral_reachability_worker", service: PivotValidationService | None = None) -> None:
        super().__init__(name=name)
        self._service = service or PivotValidationService()

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        bindings = dict(task_spec.input_bindings)
        selected_route = dict(bindings.get("selected_route")) if isinstance(bindings.get("selected_route"), dict) else {}
        selected_route.setdefault("route_id", bindings.get("route_id") or task_spec.constraints.get("route_id"))
        selected_route.setdefault("source_host", bindings.get("source_host_id") or bindings.get("source_host"))
        selected_route.setdefault("destination_host", bindings.get("target_host_id") or bindings.get("destination_host") or bindings.get("host_id"))
        selected_route.setdefault("session_id", bindings.get("session_id") or task_spec.constraints.get("session_id"))
        selected_route.setdefault("protocol", bindings.get("protocol"))
        if bindings.get("port") is not None:
            selected_route.setdefault("allowed_ports", [bindings.get("port")])
        request = PivotValidationRequest.from_task_spec(
            task_spec=task_spec.model_copy(
                update={
                    "task_type": "lateral_reachability_validation",
                    "input_bindings": {**bindings, "selected_route": selected_route},
                }
            ),
            agent_input=agent_input,
        )
        result = self._service.validate(request)
        evidence_ref = result.evidence[0].get("payload_ref") if result.evidence else None
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type="lateral_reachability_validation",
            success=result.success,
            summary=result.summary,
            raw_result_ref=str(evidence_ref) if evidence_ref else None,
            confidence=result.confidence,
            refs=task_spec.target_refs,
            payload=result.raw_payload | {"task_type": TaskType.LATERAL_REACHABILITY_VALIDATION.value},
        )
        return AgentOutput(
            evidence=result.evidence,
            outcomes=[outcome.to_agent_output_fragment()],
            logs=[f"worker={self.name}", result.summary],
            errors=[] if result.success else [result.summary],
        )


__all__ = ["LateralReachabilityWorker"]
