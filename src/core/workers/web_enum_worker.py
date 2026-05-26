"""Controlled web enumeration primary worker."""

from __future__ import annotations

from src.core.agents.agent_protocol import AgentInput, AgentOutput
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec
from src.core.workers.recon_worker import ReconWorker


class WebEnumerationWorker(BaseWorkerAgent):
    """Primary worker for low-risk HTTP service enumeration."""

    capabilities = frozenset({WorkerCapability.WEB_ENUMERATION, WorkerCapability.RECON})
    supported_task_types = frozenset({TaskType.WEB_ENUMERATION.value, "web_enumeration", "web_fingerprint"})

    def __init__(self, name: str = "web_enumeration_worker", *, recon_worker: ReconWorker | None = None) -> None:
        super().__init__(name=name)
        self._recon = recon_worker or ReconWorker(name=f"{name}_recon")

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        enum_spec = task_spec.model_copy(update={"task_type": "web_enumeration"})
        payload = dict(agent_input.raw_payload)
        payload.setdefault("tool_tags", ["safe_probe", "fingerprint", "web_enumeration"])
        output = self._recon.execute_task(enum_spec, agent_input.model_copy(update={"raw_payload": payload}))
        for outcome in output.outcomes:
            outcome["source_agent"] = self.name
            outcome["task_id"] = task_spec.task_id
            outcome["outcome_type"] = "web_enumeration"
        for evidence in output.evidence:
            evidence["source_agent"] = self.name
            evidence["task_id"] = task_spec.task_id
            evidence["result_type"] = "web_enumeration_result"
        output.logs.insert(0, f"worker={self.name}")
        return output


__all__ = ["WebEnumerationWorker"]
