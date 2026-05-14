from __future__ import annotations

import pytest

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.models.finding import Finding, RiskScore

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover
    TestClient = None
    _TESTCLIENT_IMPORT_ERROR = exc
else:  # pragma: no cover
    _TESTCLIENT_IMPORT_ERROR = None


def _require_test_client():
    if app_api.FastAPI is None:
        pytest.skip(app_api.FASTAPI_UNAVAILABLE_MESSAGE)
    if TestClient is None:
        pytest.skip(f"FastAPI TestClient unavailable: {_TESTCLIENT_IMPORT_ERROR}")
    return TestClient


def _client(tmp_path):
    client_cls = _require_test_client()
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.create_operation("op-findings-api")
    state = orchestrator.get_operation_state("op-findings-api")
    state.execution.metadata["control_plane"] = {"audit_redaction_enabled": True}
    state.execution.metadata["evidence_artifacts"] = [
        {
            "evidence_id": "evidence-1",
            "kind": "validation",
            "summary": "validated with token=secret-token",
            "payload_ref": "runtime://worker-results/evidence-1",
            "task_ref": "task-1",
            "tool_output_ref": "runtime://worker-results/evidence-1",
            "metadata": {"password": "pw-1"},
            "refs": [],
            "created_at": "2026-05-12T00:00:00+00:00",
        }
    ]
    state.execution.metadata["findings"] = [
        Finding(
            finding_id="finding-1",
            title="Validated vuln",
            affected_asset_refs=["svc-1"],
            service_ref="svc-1",
            vulnerability_ref="vuln-1",
            evidence_refs=["evidence-1"],
            validation_status="validated",
            severity="high",
            confidence=0.9,
            false_positive_risk=0.1,
            remediation="patch",
            risk_score=RiskScore(score=80, severity="high"),
        ).model_dump(mode="json")
    ]
    orchestrator.runtime_store.save_state(state)
    return client_cls(app_api.create_app(orchestrator=orchestrator, settings=settings))


def test_findings_api_exposes_findings_evidence_graph_and_reports(tmp_path) -> None:
    client = _client(tmp_path)

    findings = client.get("/operations/op-findings-api/findings")
    evidence = client.get("/operations/op-findings-api/evidence")
    graph = client.get("/operations/op-findings-api/graph")
    json_report = client.get("/operations/op-findings-api/report?format=json")
    csv_report = client.get("/operations/op-findings-api/report?format=csv")
    md_report = client.get("/operations/op-findings-api/report?format=md")

    assert findings.status_code == 200
    assert findings.json()[0]["finding_id"] == "finding-1"
    assert evidence.status_code == 200
    assert evidence.json()[0]["metadata"]["password"] == "[REDACTED]"
    assert graph.status_code == 200
    assert any(node["type"] == "Finding" for node in graph.json()["nodes"])
    assert json_report.status_code == 200
    assert csv_report.status_code == 200
    assert "finding-1" in csv_report.text
    assert md_report.status_code == 200
    assert "Validated vuln" in md_report.text
