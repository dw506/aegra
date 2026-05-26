"""Pivot route validation worker."""

from __future__ import annotations

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.services.pivot_validation_service import PivotValidationRequest, PivotValidationService


class PivotValidationWorker(BaseWorkerAgent):
    """Primary worker for pivot route probes and health checks."""

    capabilities = frozenset({WorkerCapability.ACCESS_VALIDATION, WorkerCapability.CONTEXT_VALIDATION})
    supported_task_types = frozenset({"pivot_route_validation", "pivot_probe", "pivot_health_check"})

    def __init__(
        self,
        name: str = "pivot_validation_worker",
        service: PivotValidationService | None = None,
    ) -> None:
        super().__init__(name=name)
        self._service = service or PivotValidationService()

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        request = PivotValidationRequest.from_task_spec(task_spec=task_spec, agent_input=agent_input)
        result = self._service.validate(request)
        evidence_ref = result.evidence[0].get("payload_ref") if result.evidence else None
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=result.success,
            summary=result.summary,
            raw_result_ref=str(evidence_ref) if evidence_ref else None,
            confidence=result.confidence,
            refs=task_spec.target_refs,
            payload=result.raw_payload,
        )
        return AgentOutput(
            outcomes=[outcome.to_agent_output_fragment()],
            evidence=result.evidence,
            logs=[
                f"worker={self.name}",
                f"task_id={task_spec.task_id}",
                f"task_type={task_spec.task_type}",
                f"status={result.status}",
                result.summary,
            ],
        )


__all__ = ["PivotValidationWorker"]
