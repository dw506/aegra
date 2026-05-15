"""Task Graph merge utilities.

AG projection can rebuild a fresh TG frontier every cycle. This module keeps
that rebuild incremental by overlaying new structural information onto the
existing graph while preserving task lifecycle state.
"""

from __future__ import annotations

from typing import Any

from src.core.models.tg import BaseTaskEdge, BaseTaskNode, TaskGraph, TaskGraphNode


def merge_task_graphs(existing: TaskGraph | None, incoming: TaskGraph) -> TaskGraph:
    """Merge a newly generated TG snapshot with the existing operation TG."""

    if existing is None or not existing.list_nodes():
        incoming.refresh_blocked_states()
        return incoming

    merged = TaskGraph()

    for node in incoming.list_nodes():
        merged.add_node(_merged_node(existing=existing, incoming=node))

    for node in existing.list_nodes():
        if node.id not in merged._nodes:
            merged.add_node(node.model_copy(deep=True))

    _add_edges(merged, incoming)
    _add_edges(merged, existing)

    merged.refresh_blocked_states()
    merged.set_metadata(
        source_ag_version=incoming.source_ag_version
        if incoming.source_ag_version is not None
        else existing.source_ag_version,
        frontier_version=incoming.frontier_version
        if incoming.frontier_version is not None
        else existing.frontier_version,
        metadata=_merged_metadata(existing=existing, incoming=incoming),
        version=max(existing.version, incoming.version, merged.version) + 1,
    )
    return merged


def _merged_node(*, existing: TaskGraph, incoming: TaskGraphNode) -> TaskGraphNode:
    if incoming.id not in existing._nodes:
        return incoming.model_copy(deep=True)

    existing_node = existing.get_node(incoming.id)
    merged_node = incoming.model_copy(deep=True)
    if isinstance(existing_node, BaseTaskNode) and isinstance(merged_node, BaseTaskNode):
        _preserve_task_lifecycle(existing_node=existing_node, merged_node=merged_node)
    return merged_node


def _preserve_task_lifecycle(*, existing_node: BaseTaskNode, merged_node: BaseTaskNode) -> None:
    merged_node.status = existing_node.status
    merged_node.reason = existing_node.reason
    merged_node.attempt_count = existing_node.attempt_count
    merged_node.attempts = existing_node.attempts
    merged_node.last_outcome_ref = existing_node.last_outcome_ref
    merged_node.created_at = existing_node.created_at
    merged_node.updated_at = existing_node.updated_at


def _add_edges(target: TaskGraph, source: TaskGraph) -> None:
    for edge in source.list_edges():
        if edge.id in target._edges:
            continue
        if edge.source not in target._nodes or edge.target not in target._nodes:
            continue
        target.add_edge(BaseTaskEdge.model_validate(edge.model_dump(mode="json")))


def _merged_metadata(*, existing: TaskGraph, incoming: TaskGraph) -> dict[str, Any]:
    existing_metadata = _custom_metadata(existing)
    incoming_metadata = _custom_metadata(incoming)
    return {
        **existing_metadata,
        **incoming_metadata,
        "merge_strategy": "incremental",
        "merged_from_tg_versions": [existing.version, incoming.version],
    }


def _custom_metadata(graph: TaskGraph) -> dict[str, Any]:
    metadata = dict(graph.to_dict().get("metadata") or {})
    for key in ("version", "source_ag_version", "frontier_version"):
        metadata.pop(key, None)
    return metadata


__all__ = ["merge_task_graphs"]
