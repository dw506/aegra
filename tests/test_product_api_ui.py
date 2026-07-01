from __future__ import annotations

from pathlib import Path

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
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        runtime_policy={"authorized_hosts": ["127.0.0.1"]},
    )
    orchestrator = AppOrchestrator(settings=settings)
    return client_cls(app_api.create_app(orchestrator=orchestrator, settings=settings)), orchestrator


def test_product_api_contract_workspace_assets_and_tasks(tmp_path) -> None:
    client, _ = _client(tmp_path)

    workspace = client.post("/workspaces", json={"id": "ws-product", "name": "Product Workspace"})
    assert workspace.status_code == 200
    assert workspace.json()["operation_id"] == "ws-product"

    asset = client.post(
        "/workspaces/ws-product/assets",
        json={"address": "127.0.0.1", "hostname": "localhost", "port": 8080, "protocol": "http"},
    )
    assert asset.status_code == 200
    assert asset.json()["authorized"] is True

    assert client.get("/workspaces").json()[0]["asset_count"] == 1
    assert client.get("/workspaces/ws-product/assets").json()[0]["links"]["audit"]


def test_frontend_smoke_serves_required_pages(tmp_path) -> None:
    if not (Path(__file__).resolve().parents[1] / "web" / "dashboard" / "dist").exists():
        pytest.skip("web dashboard dist is not built")
    client, _ = _client(tmp_path)

    response = client.get("/ui/")

    assert response.status_code == 200
    html = response.text
    assert '<div id="root"></div>' in html
    assert "/ui/assets/" in html
    assert "Aegra Graph Dashboard" in html


def test_graph_rendering_contract_has_host_service_vulnerability_evidence_chain(tmp_path) -> None:
    client, orchestrator = _client(tmp_path)
    orchestrator.create_operation("op-graph")
    state = orchestrator.get_operation_state("op-graph")
    state.execution.metadata["evidence_artifacts"] = [
        {
            "evidence_id": "evidence-1",
            "kind": "validation",
            "summary": "safe validation evidence",
            "payload_ref": "runtime://worker-results/evidence-1",
            "execution_ref": "execution-1",
            "refs": [],
            "created_at": "2026-05-12T00:00:00+00:00",
        }
    ]
    state.execution.metadata["findings"] = [
        Finding(
            finding_id="finding-1",
            title="Validated vuln",
            affected_asset_refs=["host-1"],
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

    response = client.get("/operations/op-graph/graph")

    assert response.status_code == 200
    graph = response.json()
    assert graph["nodes"]
    assert graph["edges"]
    node_types = {node["type"] for node in graph["nodes"]}
    assert {"Host", "Service", "Vulnerability", "Evidence"}.issubset(node_types)
    edge_pairs = {(edge["source"], edge["target"], edge["type"]) for edge in graph["edges"]}
    assert ("host-1", "svc-1", "EXPOSES") in edge_pairs
    assert ("svc-1", "vuln-1", "HAS_VULNERABILITY") in edge_pairs
    assert ("vuln-1", "evidence-1", "SUPPORTED_BY") in edge_pairs
