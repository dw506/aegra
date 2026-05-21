"""LEGACY wrapper for goal-condition validation."""

from __future__ import annotations

from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    AgentTaskResult,
    CheckpointHint,
    CriticSignal,
    EvidenceArtifact,
    FactWriteRequest,
    ObservationRecord,
    ProjectionRequest,
    ReplanHint,
    RuntimeControlRequest,
)
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorker
from src.core.workers.goal_validator import GoalValidator
from src.core.workers.services.goal_validation_service import GoalValidationRequest, GoalValidationService
from src.core.workers.services.result_builders import WorkerDomainResult


class GoalWorker(BaseWorker):
    """Compatibility worker that adapts legacy requests to the goal service."""

    agent_role = AgentRole.GOAL_WORKER
    supported_task_types = frozenset({TaskType.GOAL_CONDITION_VALIDATION})
    capabilities = frozenset({"validate_goal", "emit_replan_hints", "propose_goal_assertions"})

    def __init__(
        self,
        *,
        validator: GoalValidator | None = None,
        service: GoalValidationService | None = None,
    ) -> None:
        self._service = service or GoalValidationService(validator=validator)

    def default_intent(self, task_type: TaskType) -> AgentTaskIntent:
        return AgentTaskIntent.VALIDATE_GOAL

    def handle_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Validate a goal condition through `GoalValidationService`."""

        domain_request = GoalValidationRequest.from_legacy_request(request)
        result = self._service.validate(domain_request)
        return self._to_legacy_result(request, result)

    def _to_legacy_result(self, request: AgentTaskRequest, result: WorkerDomainResult) -> AgentTaskResult:
        raw = dict(result.raw_payload)
        return self._result(
            request,
            status=AgentResultStatus(result.status),
            summary=result.summary,
            error_message=raw.get("error_message"),
            observations=[ObservationRecord.model_validate(item) for item in result.observations],
            evidence=[EvidenceArtifact.model_validate(item) for item in result.evidence],
            fact_write_requests=[FactWriteRequest.model_validate(item) for item in result.fact_write_requests],
            projection_requests=[ProjectionRequest.model_validate(item) for item in result.projection_requests],
            runtime_requests=[RuntimeControlRequest.model_validate(item) for item in result.runtime_requests],
            checkpoint_hints=[CheckpointHint.model_validate(item) for item in raw.get("checkpoint_hints", [])],
            critic_signals=[CriticSignal.model_validate(item) for item in result.critic_signals],
            replan_hints=[ReplanHint.model_validate(item) for item in result.replan_hints],
            outcome_payload={
                "goal_satisfied": bool(raw.get("goal_satisfied", False)),
                "goal_evaluation": dict(raw.get("goal_evaluation", {})),
            },
        )


__all__ = ["GoalWorker"]
