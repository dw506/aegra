from __future__ import annotations

import pytest

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.attack_process import AttackStepNode
from src.core.models.kg import Host
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.visualization.graph_event import GraphOperation, VisualGraphChange, VisualGraphDelta
from src.core.visualization.graph_publisher import graph_delta_publisher
from src.core.visualization.graph_serializer import build_visual_snapshot, graph_payload_to_delta
from src.core.visualization.unified_visualization import build_unified_visualization

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover - depends on optional HTTP deps
    TestClient = None
    _TESTCLIENT_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on optional HTTP deps
    _TESTCLIENT_IMPORT_ERROR = None


def _require_test_client():
    if app_api.FastAPI is None:
        pytest.skip(app_api.FASTAPI_UNAVAILABLE_MESSAGE)
    if TestClient is None:
        pytest.skip(f"FastAPI TestClient unavailable: {_TESTCLIENT_IMPORT_ERROR}")
    return TestClient


def test_visual_snapshot_serializes_kg_ag_and_runtime_by_default() -> None:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="10.20.0.30", address="10.20.0.30"))

    ag = AttackGraph()
    ag.add_node(
        AttackStepNode(
            id="state-1",
            label="Host known",
            operation_id="op-vis",
            capability="recon",
        )
    )

    runtime = RuntimeState(operation_id="op-vis", execution=OperationRuntime(operation_id="op-vis"))
    snapshot = build_visual_snapshot(
        operation_id="op-vis",
        kg_payload=kg.to_dict(),
        ag_payload=ag.to_dict(),
        runtime_state=runtime,
    )

    assert snapshot.type == "graph_snapshot"
    assert snapshot.graphs["kg"].nodes[0].id == "host-1"
    assert snapshot.graphs["ag"].nodes[0].type == "ATTACK_STEP"
    assert snapshot.graphs["runtime"].nodes[0].type == "OperationRuntime"
    assert "tg" not in snapshot.graphs


def test_graph_payload_to_delta_upserts_nodes_and_edges() -> None:
    payload = {
        "version": 7,
        "nodes": [{"id": "n1", "label": "Node 1", "type": "Host", "status": "observed"}],
        "edges": [{"id": "e1", "source": "n1", "target": "n1", "type": "SELF", "label": "self"}],
    }

    delta = graph_payload_to_delta(operation_id="op-vis", graph="kg", payload=payload)

    assert delta.version == 7
    assert [change.operation for change in delta.changes] == [
        GraphOperation.UPSERT_NODE,
        GraphOperation.UPSERT_EDGE,
    ]


def test_unified_visualization_adapts_generic_fields_without_environment_assumptions() -> None:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-alpha", label="host-alpha", hostname="host-alpha"))
    runtime = RuntimeState(operation_id="op-unified", execution=OperationRuntime(operation_id="op-unified"))
    ag = AttackGraph()
    ag.add_node(
        AttackStepNode(
            id="state-alpha",
            label="Recon completed",
            operation_id="op-unified",
            agent_name="execution_agent",
            cycle_index=1,
            status="success",
            summary="Host was discovered",
            capability="recon",
            properties={"target": "host-alpha"},
        )
    )

    payload = build_unified_visualization(
        operation_id="op-unified",
        kg_payload=kg.to_dict(),
        ag_payload=ag.to_dict(),
        runtime_state=runtime,
    )

    assert payload["operation"]["id"] == "op-unified"
    assert payload["kg"]["nodes"][0]["display_name"] == "host-alpha"
    assert payload["overview"]["asset_count"] == 1
    assert payload["timeline"][0]["id"] == "timeline::state-alpha"


def test_unified_visualization_contract_includes_current_dashboard_interfaces() -> None:
    runtime = RuntimeState(operation_id="op-interface", execution=OperationRuntime(operation_id="op-interface"))
    runtime.execution.metadata["findings"] = [
        {
            "finding_id": "finding-1",
            "title": "Open HTTP service",
            "summary": "HTTP was observed on the target.",
            "severity": "info",
            "confidence": 0.8,
            "evidence_ids": ["evidence-1"],
            "target": "host-alpha",
        }
    ]
    kg_payload = {
        "nodes": [
            {"id": "host-alpha", "type": "Host", "address": "host-alpha", "status": "observed"},
            {
                "id": "service-http",
                "type": "Service",
                "status": "observed",
                "properties": {
                    "host": "host-alpha",
                    "service_name": "http",
                    "port": 80,
                    "protocol": "tcp",
                    "product": "generic-http",
                    "version": "1.0",
                    "risk_tags": ["http_service_detected"],
                    "evidence_ids": ["evidence-1"],
                },
            },
            {"id": "evidence-1", "type": "Evidence", "summary": "Probe output", "status": "observed"},
        ],
        "edges": [{"id": "edge-1", "source": "host-alpha", "target": "service-http", "type": "EXPOSES"}],
    }
    ag_payload = {
        "nodes": [
            {
                "id": "step-1",
                "type": "ATTACK_STEP",
                "properties": {
                    "cycle_index": 1,
                    "agent_name": "execution_agent",
                    "capability": "recon",
                    "objective": "Discover exposed services.",
                    "status": "success",
                    "tool_name": "safe_probe",
                    "target": "host-alpha",
                    "result_summary": "HTTP service discovered.",
                    "evidence_ids": ["evidence-1"],
                    "finding_ids": ["finding-1"],
                    "risk_tags": ["http_service_detected", "weak_config_candidate"],
                    "trace_id": "trace-1",
                },
            },
        ],
        "edges": [],
    }

    payload = build_unified_visualization(
        operation_id="op-interface",
        kg_payload=kg_payload,
        ag_payload=ag_payload,
        runtime_state=runtime,
    )

    assert payload["tool_trace"][0]["id"] == "trace-1"
    assert payload["findings"][0]["evidence_ids"] == ["evidence-1"]
    assert payload["service_matrix"][0]["services"][0]["name"] == "http"
    assert payload["service_matrix"][0]["services"][0]["product"] == "generic-http"
    assert payload["service_matrix"][0]["services"][0]["version"] == "1.0"
    assert payload["service_matrix"][0]["open_ports"] == 1
    assert payload["service_matrix"][0]["vulnerability_tags"] == ["http_service_detected", "weak_config_candidate"]
    assert payload["overview"]["open_ports"] == 1
    assert payload["overview"]["identified_services"] == 1
    assert payload["overview"]["vulnerability_tag_count"] == 2
    assert payload["overview"]["last_cycle_summary"]["cycle_index"] == 1
    assert payload["risk_tags"] == ["http_service_detected", "weak_config_candidate"]
    assert payload["kg_groups"]["assets"][0]["id"] == "host-alpha"
    assert payload["timeline"][0]["result_summary"]


def test_visual_snapshot_api_and_websocket(tmp_path) -> None:
    client_cls = _require_test_client()
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.create_operation("op-vis")
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="10.20.0.30", address="10.20.0.30"))
    orchestrator.graph_memory_store.save_kg("op-vis", kg)
    client = client_cls(app_api.create_app(orchestrator=orchestrator, settings=settings))

    response = client.get("/operations/op-vis/visual-graphs/snapshot")

    assert response.status_code == 200
    assert response.json()["graphs"]["kg"]["nodes"][0]["id"] == "host-1"
    visualization_response = client.get("/operations/op-vis/visualization")
    assert visualization_response.status_code == 200
    assert visualization_response.json()["kg"]["nodes"][0]["display_name"] == "10.20.0.30"
    trace_response = client.get("/operations/op-vis/trace")
    assert trace_response.status_code == 200
    assert trace_response.json()["operation_id"] == "op-vis"
    assert "operation-trace.txt" in trace_response.json()["path"]
    assert client.get("/operations/missing/visual-graphs/snapshot").status_code == 404

    with client.websocket_connect("/operations/op-vis/visual-graphs/ws") as websocket:
        assert websocket.receive_json()["type"] == "graph_snapshot"
        graph_delta_publisher.publish_nowait(
            VisualGraphDelta(
                operation_id="op-vis",
                graph="kg",
                version=2,
                changes=[
                    VisualGraphChange(
                        operation=GraphOperation.UPSERT_NODE,
                        entity_id="host-2",
                        entity_type="Host",
                        label="10.20.0.31",
                    )
                ],
            )
        )
        message = websocket.receive_json()
        assert message["type"] == "graph_delta"
        assert message["changes"][0]["entity_id"] == "host-2"
