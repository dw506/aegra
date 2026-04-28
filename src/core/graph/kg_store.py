"""In-memory knowledge graph implementation."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from src.core.models.kg import (
    EDGE_MODEL_BY_TYPE,
    BaseEdge,
    BaseNode,
    Evidence,
    GraphChange,
    GraphDelta,
    GraphEntityRef,
    Observation,
    parse_edge,
    parse_node,
)
from src.core.models.kg_enums import ChangeOperation, EdgeType, EntityStatus, NodeType
from src.core.models.kg_exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    ValidationConstraintError,
)
from src.core.models.kg_query import QueryFilter
from src.core.models.kg_types import Direction, JsonDict


class KnowledgeGraph:
    """Mutable in-memory knowledge graph with indexes and traceability helpers."""

    def __init__(self) -> None:
        self._nodes: dict[str, BaseNode] = {}
        self._edges: dict[str, BaseEdge] = {}
        self._node_type_index: dict[str, set[str]] = defaultdict(set)
        self._edge_type_index: dict[str, set[str]] = defaultdict(set)
        self._outgoing_index: dict[str, set[str]] = defaultdict(set)
        self._incoming_index: dict[str, set[str]] = defaultdict(set)
        self._source_ref_index: dict[str, set[str]] = defaultdict(set)
        self._delta = GraphDelta()
        self._version: int = 0
        self._last_patch_batch_id: str | None = None

    @property
    def delta(self) -> GraphDelta:
        """Return the recorded graph change log."""

        return self._delta

    @property
    def version(self) -> int:
        """Return the logical KG version."""

        return self._version

    @property
    def last_patch_batch_id(self) -> str | None:
        """Return the last successfully applied patch batch id."""

        return self._last_patch_batch_id

    def apply_patch_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply one StateWriter-style patch batch to the KG.

        这里故意只接受序列化后的字典载荷，避免 KG store 反向依赖 agent 层类型。
        真正的协议 owner 仍是 StateWriter；store 只负责执行结构变更并推进版本。
        """

        state_deltas = request.get("state_deltas") or []
        metadata = request.get("metadata") or {}
        batch_id = request.get("patch_batch_id") or metadata.get("patch_batch_id")
        base_version = request.get("base_kg_version")
        if base_version is not None and int(base_version) != self._version:
            raise ValidationConstraintError(
                f"patch batch expects KG version {base_version}, current version is {self._version}"
            )

        applied_entity_ids: list[str] = []
        applied_relation_ids: list[str] = []
        applied_delta_ids: list[str] = []

        for delta in state_deltas:
            if not isinstance(delta, dict):
                continue
            patch = delta.get("patch")
            if not isinstance(patch, dict):
                continue
            patch_kind = (
                delta.get("payload", {}).get("patch_kind")
                if isinstance(delta.get("payload"), dict)
                else None
            ) or patch.get("entity_kind")
            if patch_kind == "entity" or patch.get("entity_kind") == "node":
                entity = self._apply_entity_patch(patch)
                applied_entity_ids.append(entity.id)
            elif patch_kind == "relation" or patch.get("entity_kind") == "edge":
                relation = self._apply_relation_patch(patch)
                applied_relation_ids.append(relation.id)
            if delta.get("id"):
                applied_delta_ids.append(str(delta["id"]))

        self._last_patch_batch_id = str(batch_id) if batch_id else self._last_patch_batch_id
        self._delta.last_patch_batch_id = self._last_patch_batch_id
        return {
            "patch_batch_id": self._last_patch_batch_id,
            "kg_version": self._version,
            "applied_delta_ids": applied_delta_ids,
            "applied_entity_ids": applied_entity_ids,
            "applied_relation_ids": applied_relation_ids,
        }

    def add_node(self, node: BaseNode) -> BaseNode:
        """Add a new node to the graph."""

        if node.id in self._nodes:
            raise DuplicateEntityError(f"node '{node.id}' already exists")
        self._nodes[node.id] = node
        self._node_type_index[node.type.value].add(node.id)
        self._index_source_refs("node", node.id, node.source_refs)
        self._record_change(ChangeOperation.CREATE, node.to_ref("node"), None, node)
        return node

    def update_node(self, node_id: str, patch: dict[str, Any]) -> BaseNode:
        """Patch an existing node and revalidate its model."""

        current = self.get_node(node_id)
        if "id" in patch and patch["id"] != node_id:
            raise ValidationConstraintError("node ID cannot be changed")
        if "type" in patch and self._node_type_key(patch["type"]) != current.type.value:
            raise ValidationConstraintError("node type cannot be changed")

        payload = current.model_dump()
        payload.update(patch)
        updated = current.__class__.model_validate(payload)

        self._unindex_source_refs("node", current.id, current.source_refs)
        self._nodes[node_id] = updated
        self._index_source_refs("node", updated.id, updated.source_refs)
        self._record_change(ChangeOperation.UPDATE, updated.to_ref("node"), current, updated)
        return updated

    def get_node(self, node_id: str) -> BaseNode:
        """Return a node by ID."""

        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise EntityNotFoundError(f"node '{node_id}' was not found") from exc

    def remove_node(self, node_id: str) -> BaseNode:
        """Remove a node and all incident edges."""

        node = self.get_node(node_id)
        incident_edge_ids = set(self._outgoing_index.get(node_id, set()))
        incident_edge_ids.update(self._incoming_index.get(node_id, set()))
        for edge_id in list(incident_edge_ids):
            self.remove_edge(edge_id)

        self._node_type_index[node.type.value].discard(node_id)
        self._unindex_source_refs("node", node.id, node.source_refs)
        del self._nodes[node_id]
        self._record_change(ChangeOperation.DELETE, node.to_ref("node"), node, None)
        return node

    def add_edge(self, edge: BaseEdge) -> BaseEdge:
        """Add a validated edge between existing nodes."""

        if edge.id in self._edges:
            raise DuplicateEntityError(f"edge '{edge.id}' already exists")
        self._validate_edge_constraints(edge)
        self._edges[edge.id] = edge
        self._edge_type_index[edge.type.value].add(edge.id)
        self._outgoing_index[edge.source].add(edge.id)
        self._incoming_index[edge.target].add(edge.id)
        self._index_source_refs("edge", edge.id, edge.source_refs)
        self._record_change(ChangeOperation.CREATE, edge.to_ref("edge"), None, edge)
        return edge

    def get_edge(self, edge_id: str) -> BaseEdge:
        """Return an edge by ID."""

        try:
            return self._edges[edge_id]
        except KeyError as exc:
            raise EntityNotFoundError(f"edge '{edge_id}' was not found") from exc

    def remove_edge(self, edge_id: str) -> BaseEdge:
        """Remove an edge by ID."""

        edge = self.get_edge(edge_id)
        self._edge_type_index[edge.type.value].discard(edge_id)
        self._outgoing_index[edge.source].discard(edge_id)
        self._incoming_index[edge.target].discard(edge_id)
        self._unindex_source_refs("edge", edge.id, edge.source_refs)
        del self._edges[edge_id]
        self._record_change(ChangeOperation.DELETE, edge.to_ref("edge"), edge, None)
        return edge

    def list_nodes(
        self,
        type: str | NodeType | None = None,
        status: str | EntityStatus | None = None,
        tags: set[str] | list[str] | None = None,
    ) -> list[BaseNode]:
        """List nodes filtered by type, status and tags."""

        node_ids = (
            set(self._nodes)
            if type is None
            else set(self._node_type_index.get(self._node_type_key(type), set()))
        )
        requested_tags = set(tags or [])
        result: list[BaseNode] = []
        for node_id in node_ids:
            node = self._nodes[node_id]
            if status is not None and node.status.value != self._status_key(status):
                continue
            if not requested_tags.issubset(node.tags):
                continue
            result.append(node)
        return sorted(result, key=lambda item: item.id)

    def list_edges(
        self,
        type: str | EdgeType | None = None,
        source: str | None = None,
        target: str | None = None,
    ) -> list[BaseEdge]:
        """List edges filtered by type or endpoints."""

        edge_ids = (
            set(self._edges)
            if type is None
            else set(self._edge_type_index.get(self._edge_type_key(type), set()))
        )
        if source is not None:
            edge_ids &= set(self._outgoing_index.get(source, set()))
        if target is not None:
            edge_ids &= set(self._incoming_index.get(target, set()))
        return sorted((self._edges[edge_id] for edge_id in edge_ids), key=lambda item: item.id)

    def upsert_observation(self, observation: Observation | JsonDict) -> Observation:
        """Insert or update an observation node."""

        payload = observation.model_dump() if isinstance(observation, Observation) else dict(observation)
        candidate = Observation.model_validate(payload)
        if candidate.id in self._nodes:
            patch = candidate.model_dump()
            patch["first_seen"] = self.get_node(candidate.id).first_seen
            return self.update_node(candidate.id, patch)  # type: ignore[return-value]
        return self.add_node(candidate)  # type: ignore[return-value]

    def upsert_evidence(self, evidence: Evidence | JsonDict) -> Evidence:
        """Insert or update an evidence node."""

        payload = evidence.model_dump() if isinstance(evidence, Evidence) else dict(evidence)
        candidate = Evidence.model_validate(payload)
        if candidate.id in self._nodes:
            patch = candidate.model_dump()
            patch["first_seen"] = self.get_node(candidate.id).first_seen
            return self.update_node(candidate.id, patch)  # type: ignore[return-value]
        return self.add_node(candidate)  # type: ignore[return-value]

    def link_supported_by(
        self,
        subject_ref: GraphEntityRef | str,
        evidence_ref: GraphEntityRef | str,
    ) -> BaseEdge:
        """Create or return a SUPPORTING edge from a subject node to evidence."""

        subject_id = self._normalize_ref(subject_ref)
        evidence_id = self._normalize_ref(evidence_ref)
        evidence = self.get_node(evidence_id)
        if not isinstance(evidence, Evidence):
            raise ValidationConstraintError("SUPPORTED_BY target must be an Evidence node")

        edge_id = f"supported_by::{subject_id}::{evidence_id}"
        if edge_id in self._edges:
            return self._edges[edge_id]

        edge_cls = EDGE_MODEL_BY_TYPE[EdgeType.SUPPORTED_BY]
        edge = edge_cls(
            id=edge_id,
            type=EdgeType.SUPPORTED_BY,
            label="supported_by",
            source=subject_id,
            target=evidence_id,
        )
        return self.add_edge(edge)

    def neighbors(
        self,
        node_id: str,
        edge_type: str | EdgeType | None = None,
        direction: Direction = "both",
    ) -> list[BaseNode]:
        """Return neighboring nodes from incoming, outgoing or both directions."""

        self.get_node(node_id)
        if direction not in {"in", "out", "both"}:
            raise ValidationConstraintError("direction must be one of: in, out, both")

        edge_ids: set[str] = set()
        if direction in {"out", "both"}:
            edge_ids.update(self._outgoing_index.get(node_id, set()))
        if direction in {"in", "both"}:
            edge_ids.update(self._incoming_index.get(node_id, set()))
        if edge_type is not None:
            edge_ids = {
                edge_id
                for edge_id in edge_ids
                if self._edges[edge_id].type.value == self._edge_type_key(edge_type)
            }

        result: dict[str, BaseNode] = {}
        for edge_id in edge_ids:
            edge = self._edges[edge_id]
            other_id = edge.target if edge.source == node_id else edge.source
            result[other_id] = self._nodes[other_id]
        return sorted(result.values(), key=lambda item: item.id)

    def subgraph(self, node_ids: set[str] | list[str]) -> "KnowledgeGraph":
        """Return a new graph containing only the selected nodes and edges."""

        selected = set(node_ids)
        graph = KnowledgeGraph()
        for node_id in selected:
            graph.add_node(self.get_node(node_id))
        for edge in self._edges.values():
            if edge.source in selected and edge.target in selected:
                graph.add_edge(edge)
        return graph

    def to_dict(self) -> JsonDict:
        """Serialize the graph into a plain dictionary."""

        return {
            "version": self._version,
            "last_patch_batch_id": self._last_patch_batch_id,
            "nodes": [node.model_dump(mode="json") for node in self.list_nodes()],
            "edges": [edge.model_dump(mode="json") for edge in self.list_edges()],
            "delta": self._delta.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, payload: JsonDict) -> "KnowledgeGraph":
        """Restore a graph from serialized data."""

        graph = cls()
        for node_data in payload.get("nodes", []):
            node = parse_node(node_data)
            graph._nodes[node.id] = node
            graph._node_type_index[node.type.value].add(node.id)
            graph._index_source_refs("node", node.id, node.source_refs)
        for edge_data in payload.get("edges", []):
            edge = parse_edge(edge_data)
            graph._validate_edge_constraints(edge)
            graph._edges[edge.id] = edge
            graph._edge_type_index[edge.type.value].add(edge.id)
            graph._outgoing_index[edge.source].add(edge.id)
            graph._incoming_index[edge.target].add(edge.id)
            graph._index_source_refs("edge", edge.id, edge.source_refs)
        if "delta" in payload:
            graph._delta = GraphDelta.model_validate(payload["delta"])
        graph._version = int(payload.get("version", graph._delta.version))
        graph._last_patch_batch_id = payload.get("last_patch_batch_id") or graph._delta.last_patch_batch_id
        return graph

    def get_supporting_evidence(self, entity_id: str) -> list[Evidence]:
        """Return evidence nodes reachable from an entity via support chains."""

        self.get_node(entity_id)
        evidence_map: dict[str, Evidence] = {}
        for path in self.get_support_chain(entity_id):
            for ref in path:
                if ref.entity_kind == "node" and ref.entity_id in self._nodes:
                    node = self._nodes[ref.entity_id]
                    if isinstance(node, Evidence):
                        evidence_map[node.id] = node
        return sorted(evidence_map.values(), key=lambda item: item.id)

    def get_support_chain(self, entity_id: str, max_depth: int = 3) -> list[list[GraphEntityRef]]:
        """Return support paths following SUPPORTED_BY and DERIVED_FROM edges."""

        self.get_node(entity_id)
        if max_depth < 1:
            return []

        results: list[list[GraphEntityRef]] = []
        allowed_types = {EdgeType.SUPPORTED_BY, EdgeType.DERIVED_FROM}
        queue: deque[tuple[str, list[GraphEntityRef], int]] = deque(
            [(entity_id, [self._nodes[entity_id].to_ref("node")], 0)]
        )

        while queue:
            current_id, path, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.list_edges(source=current_id):
                if edge.type not in allowed_types:
                    continue
                next_node = self._nodes[edge.target]
                next_path = [*path, next_node.to_ref("node")]
                if isinstance(next_node, (Evidence, Observation)):
                    results.append(next_path)
                if next_node.id not in {ref.entity_id for ref in path}:
                    queue.append((next_node.id, next_path, depth + 1))
        return results

    def get_observations_for_entity(self, entity_id: str) -> list[Observation]:
        """Return observations directly or indirectly linked to an entity."""

        self.get_node(entity_id)
        observations: dict[str, Observation] = {}
        for edge in self.list_edges(type=EdgeType.OBSERVED_ON, target=entity_id):
            node = self._nodes[edge.source]
            if isinstance(node, Observation):
                observations[node.id] = node
        for path in self.get_support_chain(entity_id, max_depth=4):
            for ref in path:
                if ref.entity_kind == "node" and ref.entity_id in self._nodes:
                    node = self._nodes[ref.entity_id]
                    if isinstance(node, Observation):
                        observations[node.id] = node
        return sorted(observations.values(), key=lambda item: item.id)

    def get_entities_supported_by_evidence(self, evidence_id: str) -> list[BaseNode]:
        """Return entities that directly point to an evidence node."""

        evidence = self.get_node(evidence_id)
        if not isinstance(evidence, Evidence):
            raise ValidationConstraintError("evidence_id must reference an Evidence node")

        result: dict[str, BaseNode] = {}
        for edge in self.list_edges(type=EdgeType.SUPPORTED_BY, target=evidence_id):
            result[edge.source] = self._nodes[edge.source]
        return sorted(result.values(), key=lambda item: item.id)

    def get_hosts(self) -> list[BaseNode]:
        """Return all host nodes."""

        return self.list_nodes(type=NodeType.HOST)

    def get_services_for_host(self, host_id: str) -> list[BaseNode]:
        """Return services hosted on the given host."""

        self._ensure_node_type(host_id, NodeType.HOST)
        return self._collect_targets(host_id, EdgeType.HOSTS)

    def get_identities_on_host(self, host_id: str) -> list[BaseNode]:
        """Return identities present on the given host."""

        self._ensure_node_type(host_id, NodeType.HOST)
        return self._collect_sources(host_id, EdgeType.IDENTITY_PRESENT_ON)

    def get_sessions_on_host(self, host_id: str) -> list[BaseNode]:
        """Return sessions active on the given host."""

        self._ensure_node_type(host_id, NodeType.HOST)
        return self._collect_sources(host_id, EdgeType.SESSION_ON)

    def get_privilege_states_for_host(self, host_id: str) -> list[BaseNode]:
        """Return privilege states that apply to the given host."""

        self._ensure_node_type(host_id, NodeType.HOST)
        return self._collect_sources(host_id, EdgeType.APPLIES_TO_HOST)

    def get_goal_related_entities(self, goal_id: str) -> list[BaseNode]:
        """Return entities connected to a goal through planning-relevant edges."""

        self._ensure_node_type(goal_id, NodeType.GOAL)
        related_edge_types = {EdgeType.TARGETS, EdgeType.RELATED_TO, EdgeType.CONTAINS}
        result: dict[str, BaseNode] = {}
        for edge in self.list_edges():
            if edge.type not in related_edge_types:
                continue
            if edge.source == goal_id:
                result[edge.target] = self._nodes[edge.target]
            elif edge.target == goal_id:
                result[edge.source] = self._nodes[edge.source]
        return sorted(result.values(), key=lambda item: item.id)

    def get_reachable_targets(self, source_id: str) -> list[BaseNode]:
        """Return nodes directly reachable from the given source entity."""

        self.get_node(source_id)
        return self._collect_targets(source_id, EdgeType.CAN_REACH)

    def get_entities_by_confidence(self, min_confidence: float) -> list[BaseNode]:
        """Return nodes with confidence at or above the given threshold."""

        query = QueryFilter(min_confidence=min_confidence)
        return sorted(
            [node for node in self._nodes.values() if query.matches_confidence(node.confidence)],
            key=lambda item: item.id,
        )

    def _collect_targets(self, source_id: str, edge_type: EdgeType) -> list[BaseNode]:
        result = {
            edge.target: self._nodes[edge.target]
            for edge in self.list_edges(type=edge_type, source=source_id)
        }
        return sorted(result.values(), key=lambda item: item.id)

    def _collect_sources(self, target_id: str, edge_type: EdgeType) -> list[BaseNode]:
        result = {
            edge.source: self._nodes[edge.source]
            for edge in self.list_edges(type=edge_type, target=target_id)
        }
        return sorted(result.values(), key=lambda item: item.id)

    def _ensure_node_type(self, node_id: str, expected_type: NodeType) -> None:
        node = self.get_node(node_id)
        if node.type != expected_type:
            raise ValidationConstraintError(
                f"node '{node_id}' must be of type '{expected_type.value}'"
            )

    def _validate_edge_constraints(self, edge: BaseEdge) -> None:
        if edge.source not in self._nodes:
            raise ValidationConstraintError(f"edge source '{edge.source}' does not exist")
        if edge.target not in self._nodes:
            raise ValidationConstraintError(f"edge target '{edge.target}' does not exist")
        if edge.type == EdgeType.SUPPORTED_BY and self._nodes[edge.target].type != NodeType.EVIDENCE:
            raise ValidationConstraintError("SUPPORTED_BY target must be Evidence")
        if edge.type == EdgeType.OBSERVED_ON and self._nodes[edge.source].type != NodeType.OBSERVATION:
            raise ValidationConstraintError("OBSERVED_ON source must be Observation")

    def _record_change(
        self,
        operation: ChangeOperation,
        entity_ref: GraphEntityRef,
        before: BaseNode | BaseEdge | None,
        after: BaseNode | BaseEdge | None,
    ) -> None:
        self._version += 1
        self._delta.changes.append(
            GraphChange(
                operation=operation,
                entity_ref=entity_ref,
                before=before.model_dump(mode="json") if before is not None else None,
                after=after.model_dump(mode="json") if after is not None else None,
            )
        )
        self._delta.version = self._version
        self._delta.change_count = len(self._delta.changes)
        self._delta.last_patch_batch_id = self._last_patch_batch_id
        self._delta.last_changed_at = self._delta.changes[-1].changed_at

    def _apply_entity_patch(self, patch: dict[str, Any]) -> BaseNode:
        """Apply one entity upsert patch emitted by StateWriter."""

        entity_id = str(patch["entity_id"])
        entity_type = NodeType(str(patch["entity_type"]))
        payload = {
            "id": entity_id,
            "type": entity_type.value,
            "label": patch.get("label") or entity_id,
            "source_refs": self._normalize_source_refs(patch.get("source_refs") or []),
            **{
                key: value
                for key, value in (patch.get("attributes") or {}).items()
                if key not in {"properties"}
            },
            "properties": dict((patch.get("attributes") or {}).get("properties") or {}),
        }
        if entity_id in self._nodes:
            return self.update_node(entity_id, payload)
        return self.add_node(parse_node(payload))

    def _apply_relation_patch(self, patch: dict[str, Any]) -> BaseEdge:
        """Apply one relation upsert patch emitted by StateWriter."""

        relation_id = str(patch["relation_id"])
        relation_type = EdgeType(str(patch["relation_type"]))
        payload = {
            "id": relation_id,
            "type": relation_type.value,
            "label": patch.get("label") or relation_type.value.lower(),
            "source": str(patch["source"]),
            "target": str(patch["target"]),
            "source_refs": self._normalize_source_refs(patch.get("source_refs") or []),
            **{
                key: value
                for key, value in (patch.get("attributes") or {}).items()
                if key not in {"properties"}
            },
            "properties": dict((patch.get("attributes") or {}).get("properties") or {}),
        }
        if relation_id in self._edges:
            current = self.get_edge(relation_id)
            merged_payload = current.model_dump()
            merged_payload.update(payload)
            updated = current.__class__.model_validate(merged_payload)
            self._validate_edge_constraints(updated)
            self._edge_type_index[current.type.value].discard(current.id)
            self._outgoing_index[current.source].discard(current.id)
            self._incoming_index[current.target].discard(current.id)
            self._unindex_source_refs("edge", current.id, current.source_refs)
            self._edges[relation_id] = updated
            self._edge_type_index[updated.type.value].add(updated.id)
            self._outgoing_index[updated.source].add(updated.id)
            self._incoming_index[updated.target].add(updated.id)
            self._index_source_refs("edge", updated.id, updated.source_refs)
            self._record_change(ChangeOperation.UPDATE, updated.to_ref("edge"), current, updated)
            return updated
        return self.add_edge(parse_edge(payload))

    def _index_source_refs(
        self,
        entity_kind: str,
        entity_id: str,
        source_refs: list[GraphEntityRef],
    ) -> None:
        storage_key = self._storage_key(entity_kind, entity_id)
        for ref in source_refs:
            self._source_ref_index[ref.key()].add(storage_key)

    def _unindex_source_refs(
        self,
        entity_kind: str,
        entity_id: str,
        source_refs: list[GraphEntityRef],
    ) -> None:
        storage_key = self._storage_key(entity_kind, entity_id)
        for ref in source_refs:
            ref_key = ref.key()
            self._source_ref_index[ref_key].discard(storage_key)
            if not self._source_ref_index[ref_key]:
                del self._source_ref_index[ref_key]

    def _normalize_ref(self, value: GraphEntityRef | str) -> str:
        if isinstance(value, GraphEntityRef):
            if value.entity_kind != "node":
                raise ValidationConstraintError("only node references can be used for edge linkage")
            return value.entity_id
        return value

    @staticmethod
    def _normalize_source_refs(raw_refs: list[Any]) -> list[GraphEntityRef]:
        """把 agent 层 GraphRef 形式规范化为 KG 自己的 GraphEntityRef。"""

        normalized: list[GraphEntityRef] = []
        for raw in raw_refs:
            if isinstance(raw, GraphEntityRef):
                normalized.append(raw)
                continue
            if not isinstance(raw, dict):
                continue
            if "entity_id" in raw:
                normalized.append(GraphEntityRef.model_validate(raw))
                continue
            ref_id = raw.get("ref_id")
            if not ref_id:
                continue
            ref_type = raw.get("ref_type")
            entity_kind = "edge" if str(ref_type or "").isupper() else "node"
            normalized.append(
                GraphEntityRef(
                    entity_id=str(ref_id),
                    entity_kind=entity_kind,
                    entity_type=str(ref_type) if ref_type is not None else None,
                    label=raw.get("label"),
                )
            )
        return normalized

    @staticmethod
    def _storage_key(entity_kind: str, entity_id: str) -> str:
        return f"{entity_kind}:{entity_id}"

    @staticmethod
    def _node_type_key(value: str | NodeType) -> str:
        return value.value if isinstance(value, NodeType) else value

    @staticmethod
    def _edge_type_key(value: str | EdgeType) -> str:
        return value.value if isinstance(value, EdgeType) else value

    @staticmethod
    def _status_key(value: str | EntityStatus) -> str:
        return value.value if isinstance(value, EntityStatus) else value
