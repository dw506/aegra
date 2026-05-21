"""Goal validation worker for goal-state confirmation tasks."""

from __future__ import annotations

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.services.goal_validation_service import GoalValidationRequest, GoalValidationService


class GoalValidationWorker(BaseWorkerAgent):
    """Primary worker for goal reached and target-state confirmation."""

    capabilities = frozenset(
        {
            WorkerCapability.GOAL_VALIDATION,
            WorkerCapability.CONTEXT_VALIDATION,
        }
    )
    supported_task_types = frozenset(
        {
            TaskType.GOAL_CONDITION_VALIDATION.value,
            "goal_reached_verification",
            "target_state_confirmation",
        }
    )

    def __init__(
        self,
        name: str = "goal_validation_worker",
        service: GoalValidationService | None = None,
    ) -> None:
        super().__init__(name=name)
        self._service = service or GoalValidationService()

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        """Return True when the task is one of the supported goal operations."""

        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        """Execute one goal-validation task through the domain service."""

        domain_request = GoalValidationRequest.from_task_spec(
            task_spec=task_spec,
            agent_input=agent_input,
        )
        result = self._service.validate(domain_request)
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
            observations=result.observations,
            evidence=result.evidence,
            logs=[
                f"worker={self.name}",
                f"task_id={task_spec.task_id}",
                f"task_type={task_spec.task_type}",
                f"status={result.status}",
                result.summary,
            ],
        )


__all__ = ["GoalValidationWorker"]
