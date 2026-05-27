from __future__ import annotations

import sys

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult
from src.core.models.tg import TaskType
from src.core.workers.credential_reuse_worker import CredentialReuseWorker
from src.core.workers.credential_validation_worker import CredentialValidationWorker
from src.core.workers.internal_service_fingerprint_worker import InternalServiceFingerprintWorker
from src.core.workers.lateral_reachability_worker import LateralReachabilityWorker
from src.core.workers.port_scan_worker import PortScanWorker
from src.core.workers.web_discovery_worker import WebDiscoveryWorker
from src.core.workers.base import WorkerTaskSpec


class FakeHttpExecutor:
    def __init__(self) -> None:
        self.plans: list[ToolPlan] = []

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        self.plans.append(plan)
        return ToolExecutionResult(
            adapter="http_request",
            tool="http_request",
            success=True,
            exit_code=200,
            stdout='{"status_code": 200, "content_type": "application/json", "reachable": true}',
            metadata={"status_code": 200, "content_type": "application/json", "reachable": True},
        )


def _input(raw_payload: dict | None = None, refs: list[GraphRef] | None = None) -> AgentInput:
    return AgentInput(
        graph_refs=refs or [],
        task_ref="task-1",
        context=AgentContext(operation_id="op-1"),
        raw_payload=raw_payload or {},
    )


def _service_ref() -> GraphRef:
    return GraphRef(graph=GraphScope.KG, ref_id="svc-1", ref_type="Service")


def test_port_scan_worker_emits_structured_result_from_custom_probe() -> None:
    worker = PortScanWorker()
    spec = WorkerTaskSpec(
        task_id="task-1",
        task_type=TaskType.PORT_SCAN.value,
        input_bindings={"target_host": "host-1", "ports": "80"},
        target_refs=[GraphRef(graph=GraphScope.KG, ref_id="host-1", ref_type="Host")],
    )
    output = worker.execute_task(
        spec,
        _input(
            {
                "tool_command": [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "print(json.dumps({'summary':'port open','success':True,'reachable':True,"
                        "'entities':[{'id':'svc-1','type':'Service','host_id':'host-1','port':80,'protocol':'tcp'}],"
                        "'relations':[{'type':'HOSTS','source':'host-1','target':'svc-1'}]}))"
                    ),
                ]
            },
            refs=spec.target_refs,
        ),
    )

    assert output.outcomes[0]["outcome_type"] == "port_scan"
    assert output.evidence[0]["result_type"] == "port_scan_result"
    assert output.evidence[0]["extra"]["task_candidates"][0]["task_type"] == TaskType.INTERNAL_SERVICE_FINGERPRINT.value


def test_web_discovery_worker_accepts_preparsed_same_origin_results() -> None:
    executor = FakeHttpExecutor()
    worker = WebDiscoveryWorker(executor=executor)
    spec = WorkerTaskSpec(
        task_id="task-1",
        task_type=TaskType.WEB_DISCOVERY.value,
        input_bindings={
            "target_url": "http://127.0.0.1:8080/",
            "discovery_results": [{"path": "/api/status", "status_code": 200, "reachable": True}],
        },
        target_refs=[_service_ref()],
    )

    output = worker.execute_task(spec, _input(refs=spec.target_refs))

    assert output.outcomes[0]["success"] is True
    assert output.outcomes[0]["payload"]["endpoint_count"] == 1
    assert output.evidence[0]["extra"]["parsed"]["entities"][0]["type"] == "WebEndpoint"
    assert executor.plans == []


def test_web_discovery_worker_probes_paths_through_execution_executor() -> None:
    executor = FakeHttpExecutor()
    worker = WebDiscoveryWorker(executor=executor)
    spec = WorkerTaskSpec(
        task_id="task-1",
        task_type=TaskType.WEB_DISCOVERY.value,
        input_bindings={"target_url": "http://127.0.0.1:8080/", "paths": ["/api/status"]},
        target_refs=[_service_ref()],
    )

    output = worker.execute_task(spec, _input(refs=spec.target_refs))

    assert executor.plans
    assert executor.plans[0].tool == "http_request"
    assert executor.plans[0].target == "http://127.0.0.1:8080/api/status"
    assert output.outcomes[0]["payload"]["endpoints"][0]["status_code"] == 200


def test_credential_validation_worker_uses_access_validation_boundary() -> None:
    worker = CredentialValidationWorker()
    spec = WorkerTaskSpec(
        task_id="task-1",
        task_type=TaskType.CREDENTIAL_VALIDATION.value,
        input_bindings={"credential_id": "cred-1", "service_id": "svc-1", "username": "alice"},
        target_refs=[_service_ref()],
    )

    output = worker.execute_task(spec, _input(refs=spec.target_refs))

    assert output.outcomes[0]["outcome_type"] == "credential_validation"
    assert output.outcomes[0]["success"] is True
    assert output.outcomes[0]["payload"]["credential_status"] == "valid"


def test_credential_reuse_worker_emits_lateral_followup_candidate() -> None:
    worker = CredentialReuseWorker()
    spec = WorkerTaskSpec(
        task_id="task-1",
        task_type=TaskType.CREDENTIAL_REUSE_VALIDATION.value,
        input_bindings={
            "credential_id": "cred-1",
            "reuse_results": [{"host_id": "host-2", "service_id": "svc-2", "reusable": True}],
        },
    )

    output = worker.execute_task(spec, _input())

    assert output.outcomes[0]["payload"]["reusable"] is True
    assert output.outcomes[0]["payload"]["task_candidates"][0]["task_type"] == TaskType.LATERAL_REACHABILITY_VALIDATION.value


def test_lateral_reachability_worker_reuses_pivot_validation_service_shape() -> None:
    worker = LateralReachabilityWorker()
    spec = WorkerTaskSpec(
        task_id="task-1",
        task_type=TaskType.LATERAL_REACHABILITY_VALIDATION.value,
        input_bindings={"source_host_id": "host-1", "target_host_id": "host-2", "route_id": "route-1"},
    )

    output = worker.execute_task(spec, _input())

    assert output.outcomes[0]["outcome_type"] == "lateral_reachability_validation"
    assert output.outcomes[0]["payload"]["reachability"]["route_id"] == "route-1"


def test_internal_service_fingerprint_requires_route_context() -> None:
    worker = InternalServiceFingerprintWorker()
    blocked = worker.execute_task(
        WorkerTaskSpec(
            task_id="task-1",
            task_type=TaskType.INTERNAL_SERVICE_FINGERPRINT.value,
            input_bindings={"service_id": "svc-1"},
            target_refs=[_service_ref()],
        ),
        _input(refs=[_service_ref()]),
    )
    assert blocked.outcomes[0]["success"] is False

    output = worker.execute_task(
        WorkerTaskSpec(
            task_id="task-2",
            task_type=TaskType.INTERNAL_SERVICE_FINGERPRINT.value,
            input_bindings={"service_id": "svc-1", "route_id": "route-1", "service_name": "http"},
            target_refs=[_service_ref()],
        ),
        _input(refs=[_service_ref()]),
    )
    assert output.outcomes[0]["outcome_type"] == "internal_service_fingerprint"
