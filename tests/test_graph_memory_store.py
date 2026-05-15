from __future__ import annotations

import json

import pytest

from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import (
    ActionNode,
    ActionNodeType,
    AttackGraph,
    GraphRef,
    StateNode,
    StateNodeType,
    TruthStatus,
)
from src.core.models.kg import Host
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskType


def test_graph_memory_store_roundtrips_graphs_and_runtime(tmp_path) -> None:
    store = GraphMemoryStore(tmp_path / "runtime-store")
    operation_id = "op-1"

    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", hostname="gateway.local"))
    store.save_kg(operation_id, kg)

    ag = AttackGraph()
    ag.add_node(
        StateNode(
            id="state-1",
            label="Host known",
            node_type=StateNodeType.HOST_KNOWN,
            truth_status=TruthStatus.ACTIVE,
            subject_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        )
    )
    ag.add_node(
        ActionNode(
            id="action-1",
            label="Enumerate host",
            action_type=ActionNodeType.ENUMERATE_HOST,
        )
    )
    store.save_ag(operation_id, ag)

    tg = TaskGraph()
    tg.add_node(
        BaseTaskNode(
            id="task-1",
            label="Asset confirmation",
            task_type=TaskType.ASSET_CONFIRMATION,
            target_refs=[GraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        )
    )
    store.save_tg(operation_id, tg)

    runtime = RuntimeState(
        operation_id=operation_id,
        execution=OperationRuntime(operation_id=operation_id),
    )
    store.save_runtime(operation_id, runtime)

    assert store.load_kg(operation_id).get_node("host-1").label == "Gateway"
    assert store.load_ag(operation_id).get_node("state-1").label == "Host known"
    assert store.load_tg(operation_id).get_node("task-1").label == "Asset confirmation"
    assert store.load_runtime(operation_id).operation_id == operation_id


def test_graph_memory_store_returns_empty_missing_graphs(tmp_path) -> None:
    store = GraphMemoryStore(tmp_path)

    assert store.load_kg("missing").list_nodes() == []
    assert store.load_ag("missing").list_nodes() == []
    assert store.load_tg("missing").list_nodes() == []
    assert store.load_runtime("missing") is None


def test_graph_memory_store_saves_cycle_snapshot(tmp_path) -> None:
    store = GraphMemoryStore(tmp_path)
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway"))
    store.save_kg("op-1", kg)

    snapshot_dir = store.save_snapshot("op-1", 3)

    assert snapshot_dir.name == "cycle-000003"
    assert snapshot_dir.joinpath("kg.json").exists()
    manifest = json.loads(snapshot_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "operation_id": "op-1",
        "cycle_index": 3,
        "files": ["kg.json"],
    }


def test_graph_memory_store_rejects_path_traversal(tmp_path) -> None:
    store = GraphMemoryStore(tmp_path)

    with pytest.raises(ValueError):
        store.save_kg("..\\outside", KnowledgeGraph())

    with pytest.raises(ValueError):
        store.load_kg("../outside")
