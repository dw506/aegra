"""Credential reuse validation primary worker."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.credential_validation_worker import CredentialValidationWorker


class CredentialReuseWorker(BaseWorkerAgent):
    """Check whether a known credential can be reused on scoped targets."""

    capabilities = frozenset(
        {WorkerCapability.CREDENTIAL_REUSE_VALIDATION, WorkerCapability.CREDENTIAL_VALIDATION}
    )
    supported_task_types = frozenset(
        {TaskType.CREDENTIAL_REUSE_VALIDATION.value, "credential_reuse_validation", "check_credential_reuse"}
    )

    def __init__(
        self,
        name: str = "credential_reuse_worker",
        credential_worker: CredentialValidationWorker | None = None,
    ) -> None:
        super().__init__(name=name)
        self._credential_worker = credential_worker or CredentialValidationWorker(name=f"{name}_credential_validation")

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        provided = task_spec.input_bindings.get("reuse_results") or agent_input.raw_payload.get("reuse_results")
        if isinstance(provided, list):
            results = [dict(item) for item in provided if isinstance(item, dict)]
            return self._output_from_results(task_spec, results=results)

        validation_spec = task_spec.model_copy(update={"task_type": TaskType.CREDENTIAL_VALIDATION.value})
        output = self._credential_worker.execute_task(validation_spec, agent_input)
        reusable = bool(output.outcomes and output.outcomes[0].get("success"))
        for outcome in output.outcomes:
            outcome["source_agent"] = self.name
            outcome["task_id"] = task_spec.task_id
            outcome["outcome_type"] = "credential_reuse_validation"
            payload = outcome.setdefault("payload", {})
            if isinstance(payload, dict):
                payload["reusable"] = reusable
                payload["sensitive"] = True
                payload["task_type"] = TaskType.CREDENTIAL_REUSE_VALIDATION.value
        for evidence in output.evidence:
            evidence["source_agent"] = self.name
            evidence["task_id"] = task_spec.task_id
            evidence["result_type"] = "credential_reuse_result"
        output.logs.insert(0, f"worker={self.name}")
        return output

    def _output_from_results(self, task_spec: WorkerTaskSpec, *, results: list[dict[str, Any]]) -> AgentOutput:
        reusable = any(bool(item.get("reusable") or item.get("valid")) for item in results)
        summary = f"credential reuse checked on {len(results)} target(s); reusable={reusable}"
        payload = {
            "task_type": TaskType.CREDENTIAL_REUSE_VALIDATION.value,
            "reusable": reusable,
            "sensitive": True,
            "targets": results,
            "task_candidates": self._followup_candidates(task_spec, results),
        }
        evidence = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="credential_reuse_result",
            summary=summary,
            payload_ref=f"runtime://worker-results/credential-reuse/{task_spec.task_id}",
            refs=task_spec.target_refs,
            extra={"parsed": payload},
        )
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type="credential_reuse_validation",
            success=True,
            summary=summary,
            raw_result_ref=evidence["payload_ref"],
            confidence=0.8 if reusable else 0.55,
            refs=task_spec.target_refs,
            payload=payload,
        )
        return AgentOutput(outcomes=[outcome.to_agent_output_fragment()], evidence=[evidence], logs=[f"worker={self.name}", summary])

    @staticmethod
    def _followup_candidates(task_spec: WorkerTaskSpec, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        credential_id = task_spec.input_bindings.get("credential_id")
        candidates: list[dict[str, Any]] = []
        for item in results:
            if not bool(item.get("reusable") or item.get("valid")):
                continue
            host_id = item.get("host_id") or item.get("target_host_id") or task_spec.input_bindings.get("host_id")
            if not host_id:
                continue
            candidates.append(
                {
                    "source_action_id": f"credential-reuse::{task_spec.task_id}::{host_id}",
                    "task_type": TaskType.LATERAL_REACHABILITY_VALIDATION.value,
                    "input_bindings": {
                        "credential_id": credential_id,
                        "target_host_id": host_id,
                        "service_id": item.get("service_id"),
                    },
                    "tags": ["lateral_reachability", "credential_reuse"],
                }
            )
        return candidates


__all__ = ["CredentialReuseWorker"]
