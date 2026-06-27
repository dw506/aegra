"""In-memory knowledge graph implementation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.models.kg import (
    EDGE_MODEL_BY_TYPE,
    BaseEdge,
    BaseNode,
    GraphChange,
    GraphDelta,
    GraphEntityRef,
    NODE_MODEL_BY_TYPE,
    parse_edge,
    parse_node,
)
from src.core.models.kg_enums import ChangeOperation, EdgeType, EntityStatus, NodeType
from src.core.models.kg_exceptions import (
    DuplicateEntityError,
    EntityNotFoundError,
    ValidationConstraintError,
)
from src.core.models.kg_types import JsonDict


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
        """Apply one serialized state-delta patch batch to the KG.

        这里故意只接受序列化后的字典载荷（由 ResultApplier 构造），避免 KG store
        反向依赖 agent 层类型；store 只负责执行结构变更并推进版本。
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
        failed_delta_ids: list[str] = []
        errors: list[dict[str, Any]] = []

        for index, delta in enumerate(state_deltas):
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
            delta_id = str(delta.get("id") or patch.get("entity_id") or patch.get("relation_id") or f"delta-{index}")
            # 单条 delta 容错：一条坏 patch 不应拖垮整批写入（否则整个 stage 的写图全部丢失）。
            try:
                if patch_kind == "entity" or patch.get("entity_kind") == "node":
                    entity = self._apply_entity_patch(patch)
                    applied_entity_ids.append(entity.id)
                elif patch_kind == "relation" or patch.get("entity_kind") == "edge":
                    relation = self._apply_relation_patch(patch)
                    applied_relation_ids.append(relation.id)
                else:
                    continue
            except (
                ValidationConstraintError,
                DuplicateEntityError,
                EntityNotFoundError,
                ValueError,
                KeyError,
            ) as exc:
                failed_delta_ids.append(delta_id)
                errors.append(
                    {
                        "delta_id": delta_id,
                        "patch_kind": patch_kind or patch.get("entity_kind"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
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
            "failed_delta_ids": failed_delta_ids,
            "errors": errors,
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
        # properties 走合并而非整体替换：upsert 的精简 patch 不应抹掉之前 stage
        # 写入的丰富 properties；新键覆盖同名旧键，缺失的旧键保留。
        merged_properties = dict(payload.get("properties") or {})
        if isinstance(patch.get("properties"), dict):
            merged_properties.update(patch["properties"])
        payload.update(patch)
        if merged_properties:
            payload["properties"] = merged_properties
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
        """Apply one entity upsert patch from a state-delta batch."""

        entity_id = str(patch["entity_id"])
        entity_type = self._normalize_node_type(str(patch["entity_type"]))
        node_cls = self._node_model(entity_type)
        model_attrs, property_attrs = self._split_patch_attributes(
            node_cls,
            patch.get("attributes") or {},
            reserved={"id", "type", "label", "source_refs", "properties"},
        )
        payload = {
            "id": entity_id,
            "type": entity_type.value,
            "label": patch.get("label") or entity_id,
            "source_refs": self._normalize_source_refs(patch.get("source_refs") or []),
            **model_attrs,
            "properties": {
                **dict((patch.get("attributes") or {}).get("properties") or {}),
                **property_attrs,
            },
        }
        if entity_id in self._nodes:
            return self.update_node(entity_id, payload)
        return self.add_node(parse_node(payload))

    def _apply_relation_patch(self, patch: dict[str, Any]) -> BaseEdge:
        """Apply one relation upsert patch from a state-delta batch."""

        relation_id = str(patch["relation_id"])
        relation_type = self._normalize_edge_type(str(patch["relation_type"]))
        edge_cls = EDGE_MODEL_BY_TYPE[relation_type]
        model_attrs, property_attrs = self._split_patch_attributes(
            edge_cls,
            patch.get("attributes") or {},
            reserved={"id", "type", "label", "source", "target", "source_refs", "properties"},
        )
        payload = {
            "id": relation_id,
            "type": relation_type.value,
            "label": patch.get("label") or relation_type.value.lower(),
            "source": str(patch["source"]),
            "target": str(patch["target"]),
            "source_refs": self._normalize_source_refs(patch.get("source_refs") or []),
            **model_attrs,
            "properties": {
                **dict((patch.get("attributes") or {}).get("properties") or {}),
                **property_attrs,
            },
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

    @staticmethod
    def _node_model(node_type: NodeType) -> type[BaseNode]:
        return NODE_MODEL_BY_TYPE[node_type]

    @staticmethod
    def _split_patch_attributes(
        model_cls: type[BaseNode] | type[BaseEdge],
        attributes: dict[str, Any],
        *,
        reserved: set[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_attrs: dict[str, Any] = {}
        property_attrs: dict[str, Any] = {}
        allowed_fields = set(model_cls.model_fields)
        for key, value in attributes.items():
            if key == "properties" or key in reserved:
                continue
            if key in allowed_fields:
                if key == "status" and not KnowledgeGraph._is_entity_status(value):
                    property_attrs[key] = value
                    continue
                model_attrs[key] = value
            else:
                property_attrs[key] = value
        return model_attrs, property_attrs

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
    def _normalize_node_type(value: str) -> NodeType:
        # NB: VulnerabilityCandidate is intentionally NOT aliased — it has a
        # first-class node model (NodeType.VULNERABILITY_CANDIDATE) and the success
        # contract checks for that exact type. The remaining aliases map external
        # shapes that have no dedicated NodeType to the closest first-class kind.
        aliases = {
            "Fingerprint": NodeType.OBSERVATION,
            "WebEndpoint": NodeType.SERVICE,
            "CandidateRejected": NodeType.FINDING,
            "NeedMoreEvidence": NodeType.FINDING,
            "ValidationPlan": NodeType.OBSERVATION,
            "ValidationProfile": NodeType.OBSERVATION,
        }
        alias = aliases.get(value)
        return alias if alias is not None else NodeType(value)

    @staticmethod
    def _normalize_edge_type(value: str) -> EdgeType:
        aliases = {
            "HOSTS_SERVICE": EdgeType.HOSTS,
            "HOST_HAS_SERVICE": EdgeType.HOSTS,
            "SERVICE_ON_HOST": EdgeType.HOSTS,
            "HAS_SERVICE": EdgeType.HOSTS,
            "HAS_FINGERPRINT": EdgeType.RELATED_TO,
            "FINGERPRINTS": EdgeType.RELATED_TO,
            "HAS_VULN_CANDIDATE": EdgeType.RELATED_TO,
            "HAS_VULNERABILITY_CANDIDATE": EdgeType.RELATED_TO,
            "SUPPORTED_BY_EVIDENCE": EdgeType.SUPPORTED_BY,
        }
        alias = aliases.get(value)
        return alias if alias is not None else EdgeType(value)

    @staticmethod
    def _edge_type_key(value: str | EdgeType) -> str:
        return value.value if isinstance(value, EdgeType) else value

    @staticmethod
    def _status_key(value: str | EntityStatus) -> str:
        return value.value if isinstance(value, EntityStatus) else value

    @staticmethod
    def _is_entity_status(value: Any) -> bool:
        if isinstance(value, EntityStatus):
            return True
        try:
            EntityStatus(str(value))
        except ValueError:
            return False
        return True
