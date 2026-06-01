from __future__ import annotations

import json

from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.kg import Host
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskType


def test_graph_memory_store_default_snapshot_excludes_legacy_tg(tmp_path) -> None:
    store = GraphMemoryStore(tmp_path)
    operation_id = "op-two-graph"

    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway"))
    store.save_kg(operation_id, kg)
    store.save_ag(operation_id, AttackGraph())
    store.save_runtime(
        operation_id,
        RuntimeState(operation_id=operation_id, execution=OperationRuntime(operation_id=operation_id)),
    )

    tg = TaskGraph()
    tg.add_node(BaseTaskNode(id="task-1", label="Legacy task", task_type=TaskType.ASSET_CONFIRMATION))
    store.save_tg(operation_id, tg)

    snapshot_dir = store.save_snapshot(operation_id, 1)
    manifest = json.loads(snapshot_dir.joinpath("manifest.json").read_text(encoding="utf-8"))

    assert snapshot_dir.joinpath("kg.json").exists()
    assert snapshot_dir.joinpath("ag.json").exists()
    assert snapshot_dir.joinpath("runtime.json").exists()
    assert not snapshot_dir.joinpath("tg.json").exists()
    assert manifest["files"] == ["kg.json", "ag.json", "runtime.json"]


def test_graph_memory_store_snapshot_can_include_legacy_tg(tmp_path) -> None:
    store = GraphMemoryStore(tmp_path)
    operation_id = "op-legacy-tg"

    store.save_kg(operation_id, KnowledgeGraph())
    store.save_ag(operation_id, AttackGraph())
    store.save_runtime(
        operation_id,
        RuntimeState(operation_id=operation_id, execution=OperationRuntime(operation_id=operation_id)),
    )
    tg = TaskGraph()
    tg.add_node(BaseTaskNode(id="task-1", label="Legacy task", task_type=TaskType.ASSET_CONFIRMATION))
    store.save_tg(operation_id, tg)

    snapshot_dir = store.save_snapshot(operation_id, 1, include_legacy_tg=True)
    manifest = json.loads(snapshot_dir.joinpath("manifest.json").read_text(encoding="utf-8"))

    assert snapshot_dir.joinpath("tg.json").exists()
    assert manifest["files"] == ["kg.json", "ag.json", "tg.json", "runtime.json"]
