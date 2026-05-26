from __future__ import annotations

import pytest

from src.core.models.tg import TaskType
from src.core.workers.access_validation_worker import AccessValidationWorker
from src.core.workers.access_worker import AccessWorker
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.credential_reuse_worker import CredentialReuseWorker
from src.core.workers.credential_validation_worker import CredentialValidationWorker
from src.core.workers.fingerprint_worker import FingerprintWorker
from src.core.workers.goal_validation_worker import GoalValidationWorker
from src.core.workers.goal_worker import GoalWorker
from src.core.workers.internal_service_fingerprint_worker import InternalServiceFingerprintWorker
from src.core.workers.lateral_reachability_worker import LateralReachabilityWorker
from src.core.workers.port_scan_worker import PortScanWorker
from src.core.workers.pivot_validation_worker import PivotValidationWorker
from src.core.workers.privilege_validation_worker import PrivilegeValidationWorker
from src.core.workers.recon_worker import ReconWorker
from src.core.workers.registry import WorkerRegistry
from src.core.workers.vulnerability_validation_worker import GenericVulnerabilityValidationWorker
from src.core.workers.web_discovery_worker import WebDiscoveryWorker
from src.core.workers.web_enum_worker import WebEnumerationWorker


def test_worker_registry_default_registers_only_primary_workers() -> None:
    workers = WorkerRegistry.default().list_all()

    assert {type(worker) for worker in workers} == {
        AccessValidationWorker,
        CredentialReuseWorker,
        CredentialValidationWorker,
        FingerprintWorker,
        GenericVulnerabilityValidationWorker,
        GoalValidationWorker,
        InternalServiceFingerprintWorker,
        LateralReachabilityWorker,
        PortScanWorker,
        PivotValidationWorker,
        PrivilegeValidationWorker,
        ReconWorker,
        WebDiscoveryWorker,
        WebEnumerationWorker,
    }
    assert {worker.name for worker in workers} == {
        "access_validation_worker",
        "credential_reuse_worker",
        "credential_validation_worker",
        "fingerprint_worker",
        "generic_vulnerability_validation_worker",
        "goal_validation_worker",
        "internal_service_fingerprint_worker",
        "lateral_reachability_worker",
        "port_scan_worker",
        "pivot_validation_worker",
        "privilege_validation_worker",
        "recon_worker",
        "web_discovery_worker",
        "web_enumeration_worker",
    }


def test_worker_registry_rejects_legacy_worker() -> None:
    registry = WorkerRegistry()

    with pytest.raises(TypeError, match="BaseWorkerAgent"):
        registry.register(AccessWorker())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="BaseWorkerAgent"):
        registry.register(GoalWorker())  # type: ignore[arg-type]


def test_worker_registry_selects_access_validation_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION.value,
        )
    )

    assert isinstance(worker, AccessValidationWorker)


def test_worker_registry_selects_port_scan_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.PORT_SCAN.value,
            input_bindings={"target_host": "10.0.0.5", "ports": "22,80"},
        )
    )

    assert isinstance(worker, PortScanWorker)


def test_worker_registry_selects_web_workers() -> None:
    registry = WorkerRegistry.default()

    web_enum = registry.select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.WEB_ENUMERATION.value,
            input_bindings={"target_url": "http://127.0.0.1:8080/"},
        )
    )
    web_discovery = registry.select(
        WorkerTaskSpec(
            task_id="task-2",
            task_type=TaskType.WEB_DISCOVERY.value,
            input_bindings={"target_url": "http://127.0.0.1:8080/"},
        )
    )

    assert isinstance(web_enum, WebEnumerationWorker)
    assert isinstance(web_discovery, WebDiscoveryWorker)


def test_worker_registry_selects_credential_workers() -> None:
    registry = WorkerRegistry.default()

    credential_validation = registry.select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.CREDENTIAL_VALIDATION.value,
            input_bindings={"credential_id": "cred-1", "service_id": "svc-1"},
        )
    )
    credential_reuse = registry.select(
        WorkerTaskSpec(
            task_id="task-2",
            task_type=TaskType.CREDENTIAL_REUSE_VALIDATION.value,
            input_bindings={"credential_id": "cred-1", "service_id": "svc-2"},
        )
    )

    assert isinstance(credential_validation, CredentialValidationWorker)
    assert isinstance(credential_reuse, CredentialReuseWorker)


def test_worker_registry_selects_lateral_and_internal_fingerprint_workers() -> None:
    registry = WorkerRegistry.default()

    lateral = registry.select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.LATERAL_REACHABILITY_VALIDATION.value,
            input_bindings={"source_host_id": "host-1", "target_host_id": "host-2", "route_id": "route-1"},
        )
    )
    fingerprint = registry.select(
        WorkerTaskSpec(
            task_id="task-2",
            task_type=TaskType.INTERNAL_SERVICE_FINGERPRINT.value,
            input_bindings={"service_id": "svc-1", "route_id": "route-1"},
        )
    )

    assert isinstance(lateral, LateralReachabilityWorker)
    assert isinstance(fingerprint, InternalServiceFingerprintWorker)


def test_worker_registry_selects_pivot_validation_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type="pivot_route_validation",
            input_bindings={"route_id": "route-1"},
        )
    )

    assert isinstance(worker, PivotValidationWorker)


def test_worker_registry_selects_goal_validation_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.GOAL_CONDITION_VALIDATION.value,
        )
    )

    assert isinstance(worker, GoalValidationWorker)


def test_worker_registry_selects_privilege_validation_worker() -> None:
    worker = WorkerRegistry.default().select(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.PRIVILEGE_CONFIGURATION_VALIDATION.value,
        )
    )

    assert isinstance(worker, PrivilegeValidationWorker)
