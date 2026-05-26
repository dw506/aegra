"""Credential validation primary worker."""

from __future__ import annotations

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.services.access_validation_service import AccessValidationRequest, AccessValidationService


class CredentialValidationWorker(BaseWorkerAgent):
    """Validate one credential against one scoped target service."""

    capabilities = frozenset({WorkerCapability.CREDENTIAL_VALIDATION, WorkerCapability.ACCESS_VALIDATION})
    supported_task_types = frozenset({TaskType.CREDENTIAL_VALIDATION.value, "credential_validation", "validate_credential"})

    def __init__(self, name: str = "credential_validation_worker", service: AccessValidationService | None = None) -> None:
        super().__init__(name=name)
        self._service = service or AccessValidationService()

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        raw_metadata = agent_input.raw_payload.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.update(
            {
                "require_credential": True,
                "require_session": False,
                "credential_validation": self._credential_metadata(task_spec),
                "bound_target": self._bound_target(task_spec),
            }
        )
        request = AccessValidationRequest.from_task_spec(
            task_spec=task_spec.model_copy(
                update={
                    "task_type": "credential_validation",
                }
            ),
            agent_input=agent_input.model_copy(
                update={
                    "raw_payload": {
                        **agent_input.raw_payload,
                        "metadata": metadata,
                    }
                }
            ),
        )
        result = self._service.validate(request)
        evidence_ref = result.evidence[0].get("payload_ref") if result.evidence else None
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type="credential_validation",
            success=result.success,
            summary=result.summary,
            raw_result_ref=str(evidence_ref) if evidence_ref else None,
            confidence=result.confidence,
            refs=task_spec.target_refs,
            payload=result.raw_payload,
        )
        return AgentOutput(
            observations=result.observations,
            evidence=result.evidence,
            outcomes=[outcome.to_agent_output_fragment()],
            logs=[f"worker={self.name}", result.summary],
            errors=[] if result.success else [result.summary],
        )

    @staticmethod
    def _credential_metadata(task_spec: WorkerTaskSpec) -> dict[str, object]:
        bindings = task_spec.input_bindings
        return {
            "credential_id": bindings.get("credential_id"),
            "principal": bindings.get("principal") or bindings.get("username"),
            "kind": bindings.get("credential_kind") or bindings.get("kind"),
            "secret_ref": bindings.get("secret_ref"),
            "status": bindings.get("credential_status", "valid" if bindings.get("credential_id") else "unknown"),
            "target_id": CredentialValidationWorker._bound_target(task_spec),
        }

    @staticmethod
    def _bound_target(task_spec: WorkerTaskSpec) -> str | None:
        return (
            CredentialValidationWorker._string(task_spec.input_bindings.get("service_id"))
            or CredentialValidationWorker._string(task_spec.input_bindings.get("host_id"))
            or next((ref.ref_id for ref in task_spec.target_refs if ref.ref_type in {"Service", "Host"}), None)
        )

    @staticmethod
    def _string(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["CredentialValidationWorker"]
