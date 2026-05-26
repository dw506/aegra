"""Registry for primary `BaseWorkerAgent` workers."""

from __future__ import annotations

from src.core.workers.access_validation_worker import AccessValidationWorker
from src.core.workers.base import BaseWorkerAgent, WorkerTaskSpec
from src.core.workers.credential_reuse_worker import CredentialReuseWorker
from src.core.workers.credential_validation_worker import CredentialValidationWorker
from src.core.workers.fingerprint_worker import FingerprintWorker
from src.core.workers.goal_validation_worker import GoalValidationWorker
from src.core.workers.internal_service_fingerprint_worker import InternalServiceFingerprintWorker
from src.core.workers.lateral_reachability_worker import LateralReachabilityWorker
from src.core.workers.port_scan_worker import PortScanWorker
from src.core.workers.pivot_validation_worker import PivotValidationWorker
from src.core.workers.privilege_validation_worker import PrivilegeValidationWorker
from src.core.workers.recon_worker import ReconWorker
from src.core.workers.vulnerability_validation_worker import GenericVulnerabilityValidationWorker
from src.core.workers.web_discovery_worker import WebDiscoveryWorker
from src.core.workers.web_enum_worker import WebEnumerationWorker


class WorkerNotFoundError(LookupError):
    """Raised when no primary worker supports a task spec."""


class WorkerRegistry:
    """Registry for the new worker protocol.

    This registry only accepts `BaseWorkerAgent` implementations. Legacy
    `BaseWorker` classes remain available to compatibility tests and adapters
    but are not part of the primary worker selection path.
    """

    def __init__(self, workers: list[BaseWorkerAgent] | None = None) -> None:
        self._workers: dict[str, BaseWorkerAgent] = {}
        for worker in workers or []:
            self.register(worker)

    @classmethod
    def default(cls) -> "WorkerRegistry":
        return cls(
            [
                PortScanWorker(),
                InternalServiceFingerprintWorker(),
                WebEnumerationWorker(),
                WebDiscoveryWorker(),
                CredentialValidationWorker(),
                CredentialReuseWorker(),
                LateralReachabilityWorker(),
                AccessValidationWorker(),
                PivotValidationWorker(),
                GoalValidationWorker(),
                PrivilegeValidationWorker(),
                ReconWorker(),
                FingerprintWorker(),
                GenericVulnerabilityValidationWorker(),
            ]
        )

    def register(self, worker: BaseWorkerAgent) -> BaseWorkerAgent:
        if not isinstance(worker, BaseWorkerAgent):
            raise TypeError("WorkerRegistry only accepts BaseWorkerAgent workers")
        self._workers[worker.name] = worker
        return worker

    def select(self, task_spec: WorkerTaskSpec) -> BaseWorkerAgent:
        for worker in self._workers.values():
            if worker.supports_task(task_spec):
                return worker
        raise WorkerNotFoundError(f"no worker supports task type {task_spec.task_type}")

    def list_all(self) -> list[BaseWorkerAgent]:
        return [self._workers[name] for name in sorted(self._workers)]


__all__ = ["WorkerNotFoundError", "WorkerRegistry"]
