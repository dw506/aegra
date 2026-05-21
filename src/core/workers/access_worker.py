"""LEGACY compatibility wrapper.

New execution pipeline should use AccessValidationWorker /
PrivilegeValidationWorker. This class remains for old AgentTaskRequest /
AgentTaskResult compatibility.
"""

from __future__ import annotations

from src.core.models.events import (
    AgentResultStatus,
    AgentRole,
    AgentTaskIntent,
    AgentTaskRequest,
    AgentTaskResult,
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
from src.core.workers.services.access_validation_service import AccessValidationRequest, AccessValidationService
from src.core.workers.services.privilege_validation_service import (
    PrivilegeValidationRequest,
    PrivilegeValidationService,
)
from src.core.workers.services.result_builders import WorkerDomainResult
from src.core.workers.tool_runner import ToolRunner


class AccessWorker(BaseWorker):
    """Compatibility worker that adapts legacy requests to validation services."""

    agent_role = AgentRole.ACCESS_WORKER
    supported_task_types = frozenset(
        {
            TaskType.IDENTITY_CONTEXT_CONFIRMATION,
            TaskType.PRIVILEGE_CONFIGURATION_VALIDATION,
        }
    )
    capabilities = frozenset({"validate_access", "request_sessions", "propose_access_facts"})

    def __init__(
        self,
        *,
        tool_runner: ToolRunner | None = None,
        access_service: AccessValidationService | None = None,
        privilege_service: PrivilegeValidationService | None = None,
    ) -> None:
        self._access_service = access_service or AccessValidationService(tool_runner=tool_runner)
        self._privilege_service = privilege_service or PrivilegeValidationService()

    def default_intent(self, task_type: TaskType) -> AgentTaskIntent:
        return AgentTaskIntent.VALIDATE_ACCESS

    def handle_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        """Validate a legacy access or privilege task through its domain service."""

        if request.context.task_type == TaskType.PRIVILEGE_CONFIGURATION_VALIDATION:
            result = self._privilege_service.validate(PrivilegeValidationRequest.from_legacy_request(request))
        else:
            result = self._access_service.validate(AccessValidationRequest.from_legacy_request(request))
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
            critic_signals=[CriticSignal.model_validate(item) for item in result.critic_signals],
            replan_hints=[ReplanHint.model_validate(item) for item in result.replan_hints],
            outcome_payload={
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "status",
                    "error_message",
                    "fact_write_requests",
                    "projection_requests",
                    "runtime_requests",
                    "critic_signals",
                    "replan_hints",
                }
            },
        )


__all__ = ["AccessWorker"]
