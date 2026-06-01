from __future__ import annotations

import pytest

from src.app import api as app_api
from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph, StateNode, StateNodeType
from src.core.models.kg import Host
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.visualization.graph_event import GraphOperation, VisualGraphChange, VisualGraphDelta
from src.core.visualization.graph_publisher import graph_delta_publisher
from src.core.visualization.graph_serializer import build_visual_snapshot, graph_payload_to_delta

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
        StateNode(
            id="state-1",
            label="Host known",
            node_type=StateNodeType.HOST_KNOWN,
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
    assert snapshot.graphs["ag"].nodes[0].type == "HOST_KNOWN"
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
