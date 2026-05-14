from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.models.events import AgentRole
from src.core.models.tg import TaskNode, TaskType
from src.core.workers.fingerprint_worker import FingerprintWorker


def test_fingerprint_worker_extracts_http_service_fingerprint_from_recon_parsed_result() -> None:
    task = TaskNode(
        id="task-fp",
        label="Fingerprint service",
        task_type=TaskType.SERVICE_VALIDATION,
        source_action_id="action-fp",
        input_bindings={"host_id": "127.0.0.1", "port": 8080, "service_id": "127.0.0.1:8080/tcp"},
        target_refs=[GraphRef(graph="kg", ref_id="127.0.0.1:8080/tcp", ref_type="Service")],
        resource_keys={"host:127.0.0.1"},
    )
    worker = FingerprintWorker()
    request = worker.build_request(
        task=task,
        operation_id="op-fp",
        metadata={
            "target_url": "http://127.0.0.1:8080/manager/html",
            "parsed": {
                "confidence": 0.9,
                "service": {
                    "id": "127.0.0.1:8080/tcp",
                    "host_id": "127.0.0.1",
                    "port": 8080,
                    "protocol": "http",
                    "service_name": "http",
                    "product": "Apache Tomcat",
                    "version": "9.0",
                    "headers": {"Server": "Apache-Coyote/1.1"},
                    "http_title": "Tomcat Manager",
                    "body_signals": ["Tomcat Manager", "/manager/html"],
                    "cpe": ["cpe:/a:apache:tomcat:9.0"],
                },
            },
        },
    )

    result = worker.execute_task(request)

    assert result.agent_role == AgentRole.FINGERPRINT_WORKER
    assert result.status.value == "succeeded"
    fingerprint = result.outcome_payload["fingerprints"][0]
    assert fingerprint["protocol"] == "http"
    assert fingerprint["service_name"] == "http"
    assert fingerprint["product"] == "Apache Tomcat"
    assert fingerprint["version"] == "9.0"
    assert fingerprint["http_title"] == "Tomcat Manager"
    assert fingerprint["headers"]["Server"] == "Apache-Coyote/1.1"
    assert fingerprint["body_signals"] == ["Tomcat Manager", "/manager/html"]
    assert fingerprint["cpe"] == ["cpe:/a:apache:tomcat:9.0"]
    assert result.outcome_payload["vulnerability_candidates"][0]["matched_rule_id"] == "tomcat-manager-exposed"
    assert result.fact_write_requests[0].attributes["vulnerability_candidates"]
