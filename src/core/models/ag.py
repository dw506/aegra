"""Attack Graph models and container implementation."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Literal, TypeAlias

from src.core.models.attack_process import (
    AgentExecutionNode,
    AttackProcessEdge,
    AttackProcessEdgeType,
    AttackCycleNode,
    AttackProcessNode,
    AttackProcessNodeType,
    BlockedReasonNode,
    GoalCheckNode,
    HandoffSuggestionNode,
    PlannerDecisionNode,
    StageResultNode,
    StopDecisionNode,
    ToolCallNode,
)
from src.core.models.graph_common import GraphRef, stable_node_id


AGNode: TypeAlias = AttackProcessNode
AGEdge: TypeAlias = AttackProcessEdge


class AttackGraph:
    """In-memory Attack Graph with small indexes for planning."""

    def __init__(self) -> None:
        self._nodes: dict[str, AGNode] = {}
        self._edges: dict[str, AGEdge] = {}
        self._node_type_index: dict[str, set[str]] = defaultdict(set)
        self._outgoing_index: dict[str, set[str]] = defaultdict(set)
        self._incoming_index: dict[str, set[str]] = defaultdict(set)
        self._subject_ref_index: dict[str, set[str]] = defaultdict(set)
        self._version: int = 0
        self._source_kg_version: int | None = None
        self._projection_batch_id: str | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def version(self) -> int:
        """Return the AG logical version."""

        return self._version

    @property
    def source_kg_version(self) -> int | None:
        """Return the KG version this AG snapshot was derived from."""

        return self._source_kg_version

    @property
    def projection_batch_id(self) -> str | None:
        """Return the projection batch identifier associated with this AG snapshot."""

        return self._projection_batch_id

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a shallow copy of AG metadata for serialization and audit."""

        return dict(self._metadata)

    def set_projection_metadata(
        self,
        *,
        source_kg_version: int | None,
        projection_batch_id: str | None,
        metadata: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> None:
        """Attach projection metadata to the current AG snapshot.

        AG 的版本并不试图模拟数据库事务版本，而是作为一个轻量可追踪版本号，
        用于把当前 AG 快照与来源 KG 版本、投影批次和下游规划关联起来。
        """

        self._source_kg_version = source_kg_version
        self._projection_batch_id = projection_batch_id
        if metadata is not None:
            self._metadata = dict(metadata)
        if version is not None:
            self._version = max(version, 0)

    def add_node(self, node: AGNode) -> AGNode:
        """Add a node and index its type and subject refs."""

        if node.id in self._nodes:
            raise ValueError(f"node '{node.id}' already exists")
        self._nodes[node.id] = node
        self._node_type_index[self._node_type_key(node)].add(node.id)
        for ref in self._refs_for_index(node):
            self._subject_ref_index[ref.key()].add(node.id)
        self._version += 1
        return node

    def add_edge(self, edge: AGEdge) -> AGEdge:
        """Add an edge between existing nodes."""

        if edge.id in self._edges:
            raise ValueError(f"edge '{edge.id}' already exists")
        if edge.source not in self._nodes:
            raise ValueError(f"edge source '{edge.source}' does not exist")
        if edge.target not in self._nodes:
            raise ValueError(f"edge target '{edge.target}' does not exist")
        self._edges[edge.id] = edge
        self._outgoing_index[edge.source].add(edge.id)
        self._incoming_index[edge.target].add(edge.id)
        self._version += 1
        return edge

    def add_process_node(self, node: AttackProcessNode | None = None, **kwargs: Any) -> AttackProcessNode:
        """Create or add an attack-process node."""

        if node is not None and kwargs:
            raise ValueError("pass either a process node or process node fields, not both")
        if node is None:
            node = parse_ag_node(kwargs)
        if not isinstance(node, AttackProcessNode):
            raise ValueError("process node helper only accepts attack-process nodes")
        return self.add_node(node)

    def add_process_edge(self, edge: AttackProcessEdge | None = None, **kwargs: Any) -> AttackProcessEdge:
        """Create or add an attack-process edge."""

        if edge is not None and kwargs:
            raise ValueError("pass either a process edge or process edge fields, not both")
        if edge is None:
            edge = parse_ag_edge(kwargs)
        if not isinstance(edge, AttackProcessEdge):
            raise ValueError("process edge helper only accepts attack-process edges")
        return self.add_edge(edge)

    def get_node(self, node_id: str) -> AGNode:
        """Return a node by ID."""

        return self._nodes[node_id]

    def get_edge(self, edge_id: str) -> AGEdge:
        """Return an edge by ID."""

        return self._edges[edge_id]

    def remove_node(self, node_id: str) -> AGNode:
        """Remove a node and its incident edges."""

        node = self._nodes[node_id]
        incident_edges = set(self._outgoing_index.get(node_id, set()))
        incident_edges.update(self._incoming_index.get(node_id, set()))
        for edge_id in list(incident_edges):
            self.remove_edge(edge_id)
        del self._nodes[node_id]
        self._node_type_index[self._node_type_key(node)].discard(node_id)
        for ref in self._refs_for_index(node):
            self._subject_ref_index[ref.key()].discard(node_id)
            if not self._subject_ref_index[ref.key()]:
                del self._subject_ref_index[ref.key()]
        self._version += 1
        return node

    def remove_edge(self, edge_id: str) -> AGEdge:
        """Remove an edge by ID."""

        edge = self._edges[edge_id]
        self._outgoing_index[edge.source].discard(edge_id)
        self._incoming_index[edge.target].discard(edge_id)
        del self._edges[edge_id]
        self._version += 1
        return edge

    def list_nodes(self, node_type: str | Enum | None = None) -> list[AGNode]:
        """List nodes, optionally filtered by their effective type."""

        if node_type is None:
            nodes = self._nodes.values()
        else:
            key = node_type.value if isinstance(node_type, Enum) else node_type
            nodes = (self._nodes[node_id] for node_id in self._node_type_index.get(key, set()))
        return sorted(nodes, key=lambda item: item.id)

    def list_edges(self, edge_type: Enum | str | None = None) -> list[AGEdge]:
        """List all edges, optionally filtered by type."""

        if edge_type is None:
            edges = self._edges.values()
        else:
            key = edge_type.value if isinstance(edge_type, Enum) else edge_type
            edges = (edge for edge in self._edges.values() if edge.edge_type.value == key)
        return sorted(edges, key=lambda item: item.id)

    def neighbors(
        self,
        node_id: str,
        edge_type: Enum | str | None = None,
        direction: Literal["in", "out", "both"] = "both",
    ) -> list[AGNode]:
        """Return neighboring nodes from incoming, outgoing or both edges."""

        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be one of: in, out, both")
        edge_ids: set[str] = set()
        if direction in {"out", "both"}:
            edge_ids.update(self._outgoing_index.get(node_id, set()))
        if direction in {"in", "both"}:
            edge_ids.update(self._incoming_index.get(node_id, set()))
        if edge_type is not None:
            edge_key = edge_type.value if isinstance(edge_type, Enum) else edge_type
            edge_ids = {edge_id for edge_id in edge_ids if self._edges[edge_id].edge_type.value == edge_key}

        result: dict[str, AGNode] = {}
        for edge_id in edge_ids:
            edge = self._edges[edge_id]
            other_id = edge.target if edge.source == node_id else edge.source
            result[other_id] = self._nodes[other_id]
        return sorted(result.values(), key=lambda item: item.id)

    def find_process_nodes(
        self,
        operation_id: str | None = None,
        cycle_index: int | None = None,
        agent_name: str | None = None,
        node_type: Enum | str | None = None,
    ) -> list[AttackProcessNode]:
        """Return attack-process nodes filtered by process metadata."""

        node_type_key = node_type.value if isinstance(node_type, Enum) else node_type
        candidates = self.list_nodes(node_type_key) if node_type_key is not None else self.list_nodes()
        nodes = [node for node in candidates if isinstance(node, AttackProcessNode)]
        if operation_id is not None:
            nodes = [node for node in nodes if node.operation_id == operation_id]
        if cycle_index is not None:
            nodes = [node for node in nodes if node.cycle_index == cycle_index]
        if agent_name is not None:
            nodes = [node for node in nodes if node.agent_name == agent_name]
        return sorted(nodes, key=lambda item: (item.created_at, item.id))

    def recent_process_nodes(self, limit: int = 20) -> list[AttackProcessNode]:
        """Return the most recently created attack-process nodes."""

        nodes = [node for node in self.list_nodes() if isinstance(node, AttackProcessNode)]
        return sorted(nodes, key=lambda item: (item.created_at, item.id), reverse=True)[: max(limit, 0)]

    def by_subject_ref(self, ref: GraphRef) -> list[AGNode]:
        """Return AG nodes indexed by a subject-like graph ref."""

        return sorted(
            (self._nodes[node_id] for node_id in self._subject_ref_index.get(ref.key(), set())),
            key=lambda item: item.id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a JSON-safe dictionary."""

        return {
            "metadata": {
                "version": self._version,
                "source_kg_version": self._source_kg_version,
                "projection_batch_id": self._projection_batch_id,
                **self._metadata,
            },
            "nodes": [node.model_dump(mode="json") for node in self.list_nodes()],
            "edges": [edge.model_dump(mode="json") for edge in self.list_edges()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttackGraph":
        """Restore a graph from serialized data.

        Loading is tolerant of legacy/unknown records: nodes or edges that no
        longer have a model (e.g. retired projection nodes from an older AG
        snapshot) are skipped instead of failing the whole restore, so existing
        operations can still resume after the model changes.
        """

        graph = cls()
        skipped_node_ids: set[str] = set()
        for node_data in payload.get("nodes", []):
            try:
                node = parse_ag_node(node_data)
            except Exception:
                node_id = node_data.get("id") if isinstance(node_data, dict) else None
                if node_id is not None:
                    skipped_node_ids.add(str(node_id))
                continue
            try:
                graph.add_node(node)
            except ValueError:
                continue
        for edge_data in payload.get("edges", []):
            try:
                edge = parse_ag_edge(edge_data)
            except Exception:
                continue
            if edge.source in skipped_node_ids or edge.target in skipped_node_ids:
                continue
            try:
                graph.add_edge(edge)
            except (ValueError, KeyError):
                continue
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, dict):
            graph.set_projection_metadata(
                source_kg_version=metadata.get("source_kg_version"),
                projection_batch_id=metadata.get("projection_batch_id"),
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key not in {"version", "source_kg_version", "projection_batch_id"}
                },
                version=metadata.get("version"),
            )
        return graph

    @staticmethod
    def _node_type_key(node: AGNode) -> str:
        return node.node_type.value

    @staticmethod
    def _refs_for_index(node: AGNode) -> list[GraphRef]:
        refs: list[GraphRef] = []
        if isinstance(node, AttackProcessNode):
            refs.extend(node.refs)
        if hasattr(node, "source_refs"):
            refs.extend(node.source_refs)
        unique: dict[str, GraphRef] = {ref.key(): ref for ref in refs}
        return list(unique.values())


def parse_ag_node(data: dict[str, Any]) -> AGNode:
    """Instantiate a typed AG node from serialized data."""

    kind = data.get("kind")
    node_type = data.get("node_type")
    node_type_key = node_type.value if isinstance(node_type, Enum) else node_type
    process_node_types = {item.value for item in AttackProcessNodeType}
    if kind == "process" or node_type_key in process_node_types:
        node_data = dict(data)
        node_data.pop("kind", None)
        process_node_map = {
            AttackProcessNodeType.ATTACK_CYCLE.value: AttackCycleNode,
            AttackProcessNodeType.PLANNER_DECISION.value: PlannerDecisionNode,
            AttackProcessNodeType.AGENT_EXECUTION.value: AgentExecutionNode,
            AttackProcessNodeType.TOOL_CALL.value: ToolCallNode,
            AttackProcessNodeType.STAGE_RESULT.value: StageResultNode,
            AttackProcessNodeType.HANDOFF_SUGGESTION.value: HandoffSuggestionNode,
            AttackProcessNodeType.BLOCKED_REASON.value: BlockedReasonNode,
            AttackProcessNodeType.GOAL_CHECK.value: GoalCheckNode,
            AttackProcessNodeType.STOP_DECISION.value: StopDecisionNode,
        }
        model = process_node_map.get(node_type_key, AttackProcessNode)
        return model.model_validate(node_data)

    raise ValueError(f"unknown AG node kind or node_type: {kind or node_type_key}")


def parse_ag_edge(data: dict[str, Any]) -> AGEdge:
    """Instantiate a typed AG edge from serialized data."""

    edge_type = data.get("edge_type")
    edge_type_key = edge_type.value if isinstance(edge_type, Enum) else edge_type
    if edge_type_key in {item.value for item in AttackProcessEdgeType}:
        return AttackProcessEdge.model_validate(data)

    raise ValueError(f"unknown AG edge_type: {edge_type_key}")
