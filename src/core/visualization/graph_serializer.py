"""Serialize Aegra graph/runtime state into visualization read models."""

from __future__ import annotations

from typing import Any

from src.core.models.runtime import RuntimeState
from src.core.visualization.graph_event import (
    GraphName,
    GraphOperation,
    VisualEdge,
    VisualGraphChange,
    VisualGraphDelta,
    VisualGraphSnapshot,
    VisualGraphState,
    VisualNode,
)


def build_visual_snapshot(
    *,
    operation_id: str,
    kg_payload: dict[str, Any] | None = None,
    ag_payload: dict[str, Any] | None = None,
    tg_payload: dict[str, Any] | None = None,
    runtime_state: RuntimeState | dict[str, Any] | None = None,
    legacy_tg: bool = False,
) -> VisualGraphSnapshot:
    graphs: dict[GraphName, VisualGraphState] = {
        "kg": graph_payload_to_state("kg", kg_payload or {}),
        "ag": graph_payload_to_state("ag", ag_payload or {}),
        "runtime": runtime_to_state(runtime_state),
    }
    if legacy_tg:
        graphs["tg"] = graph_payload_to_state("tg", tg_payload or {})
    return VisualGraphSnapshot(
        operation_id=operation_id,
        graphs=graphs,
    )


def graph_payload_to_delta(
    *,
    operation_id: str,
    graph: GraphName,
    payload: dict[str, Any] | None,
) -> VisualGraphDelta:
    state = graph_payload_to_state(graph, payload or {})
    changes = [
        VisualGraphChange(
            operation=GraphOperation.UPSERT_NODE,
            entity_id=node.id,
            entity_type=node.type,
            label=node.label,
            status=node.status,
            properties=node.properties,
        )
        for node in state.nodes
    ]
    changes.extend(
        VisualGraphChange(
            operation=GraphOperation.UPSERT_EDGE,
            entity_id=edge.id,
            source=edge.source,
            target=edge.target,
            edge_type=edge.type,
            label=edge.label,
            properties=edge.properties,
        )
        for edge in state.edges
    )
    return VisualGraphDelta(
        operation_id=operation_id,
        graph=graph,
        version=state.version,
        changes=changes,
    )


def runtime_to_delta(*, operation_id: str, runtime_state: RuntimeState | dict[str, Any] | None) -> VisualGraphDelta:
    state = runtime_to_state(runtime_state)
    changes = [
        VisualGraphChange(
            operation=GraphOperation.UPSERT_NODE,
            entity_id=node.id,
            entity_type=node.type,
            label=node.label,
            status=node.status,
            properties=node.properties,
        )
        for node in state.nodes
    ]
    changes.extend(
        VisualGraphChange(
            operation=GraphOperation.UPSERT_EDGE,
            entity_id=edge.id,
            source=edge.source,
            target=edge.target,
            edge_type=edge.type,
            label=edge.label,
            properties=edge.properties,
        )
        for edge in state.edges
    )
    return VisualGraphDelta(operation_id=operation_id, graph="runtime", version=state.version, changes=changes)


def graph_payload_to_state(graph: GraphName, payload: dict[str, Any]) -> VisualGraphState:
    version = _graph_version(payload)
    nodes = [_visual_node(graph, item) for item in payload.get("nodes", []) if isinstance(item, dict)]
    edges = [_visual_edge(graph, item) for item in payload.get("edges", []) if isinstance(item, dict)]
    return VisualGraphState(version=version, nodes=nodes, edges=edges)


def runtime_to_state(runtime_state: RuntimeState | dict[str, Any] | None) -> VisualGraphState:
    payload = (
        runtime_state.model_dump(mode="json")
        if isinstance(runtime_state, RuntimeState)
        else dict(runtime_state or {})
    )
    operation_id = str(payload.get("operation_id") or "runtime")
    execution = _mapping(payload.get("execution"))
    nodes: list[VisualNode] = [
        VisualNode(
            id=f"runtime:{operation_id}",
            label=operation_id,
            type="OperationRuntime",
            graph="runtime",
            status=str(payload.get("operation_status") or execution.get("status") or ""),
            properties={
                "operation_id": operation_id,
                "last_updated": payload.get("last_updated"),
                "summary": execution.get("summary"),
            },
        )
    ]
    edges: list[VisualEdge] = []

    for task_id, task in _items(execution.get("tasks")):
        status = str(task.get("status") or "")
        node_id = f"runtime-task:{task_id}"
        nodes.append(
            VisualNode(
                id=node_id,
                label=str(task.get("tg_node_id") or task_id),
                type="TaskRuntime",
                graph="runtime",
                status=status,
                properties=task,
            )
        )
        edges.append(_runtime_edge(f"runtime-edge:{operation_id}:task:{task_id}", f"runtime:{operation_id}", node_id, "has_task"))

    for worker_id, worker in _items(payload.get("workers")):
        node_id = f"worker:{worker_id}"
        nodes.append(
            VisualNode(
                id=node_id,
                label=str(worker.get("agent_name") or worker_id),
                type="WorkerRuntime",
                graph="runtime",
                status=str(worker.get("status") or ""),
                properties=worker,
            )
        )
        edges.append(_runtime_edge(f"runtime-edge:{operation_id}:worker:{worker_id}", f"runtime:{operation_id}", node_id, "has_worker"))

    metadata = _mapping(execution.get("metadata"))
    for collection_name, node_type in (
        ("sessions", "SessionRuntime"),
        ("credentials", "CredentialRuntime"),
        ("locks", "ResourceLock"),
        ("pivot_routes", "PivotRouteRuntime"),
        ("routes", "PivotRouteRuntime"),
    ):
        for item_id, item in _items(payload.get(collection_name) or execution.get(collection_name) or metadata.get(collection_name)):
            node_id = f"{collection_name}:{item_id}"
            nodes.append(
                VisualNode(
                    id=node_id,
                    label=str(item.get("label") or item.get("session_id") or item.get("route_id") or item_id),
                    type=node_type,
                    graph="runtime",
                    status=str(item.get("status") or ""),
                    properties=item,
                )
            )
            edges.append(_runtime_edge(f"runtime-edge:{operation_id}:{collection_name}:{item_id}", f"runtime:{operation_id}", node_id, collection_name))

    return VisualGraphState(version=_runtime_version(payload), nodes=nodes, edges=edges)


def _visual_node(graph: GraphName, item: dict[str, Any]) -> VisualNode:
    entity_type = _string_value(item.get("type") or item.get("node_type") or item.get("task_type") or item.get("kind"))
    status = _string_value(item.get("status") or item.get("truth_status") or item.get("activation_status"))
    properties = dict(item)
    return VisualNode(
        id=str(item.get("id")),
        label=str(item.get("label") or item.get("id")),
        type=entity_type,
        graph=graph,
        status=status,
        properties=properties,
    )


def _visual_edge(graph: GraphName, item: dict[str, Any]) -> VisualEdge:
    edge_type = _string_value(item.get("type") or item.get("edge_type") or item.get("dependency_type"))
    return VisualEdge(
        id=str(item.get("id")),
        source=str(item.get("source")),
        target=str(item.get("target")),
        label=str(item.get("label") or edge_type or ""),
        type=edge_type,
        graph=graph,
        properties=dict(item),
    )


def _runtime_edge(edge_id: str, source: str, target: str, edge_type: str) -> VisualEdge:
    return VisualEdge(
        id=edge_id,
        source=source,
        target=target,
        label=edge_type,
        type=edge_type,
        graph="runtime",
        properties={},
    )


def _graph_version(payload: dict[str, Any]) -> int:
    metadata = _mapping(payload.get("metadata"))
    return int(payload.get("version") or metadata.get("version") or 0)


def _runtime_version(payload: dict[str, Any]) -> int:
    metadata = _mapping(_mapping(payload.get("execution")).get("metadata"))
    graph_memory = _mapping(metadata.get("graph_memory"))
    versions = [
        graph_memory.get("kg_version"),
        graph_memory.get("ag_version"),
        graph_memory.get("tg_version"),
        len(_mapping(_mapping(payload.get("execution")).get("tasks"))),
        len(payload.get("pending_events") or []),
    ]
    return sum(int(value or 0) for value in versions)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _items(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        return [(str(key), dict(item)) for key, item in value.items() if isinstance(item, dict)]
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                item_id = str(item.get("id") or item.get("task_id") or item.get("session_id") or item.get("route_id") or index)
                result.append((item_id, dict(item)))
        return result
    return []


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "build_visual_snapshot",
    "graph_payload_to_delta",
    "graph_payload_to_state",
    "runtime_to_delta",
    "runtime_to_state",
]
