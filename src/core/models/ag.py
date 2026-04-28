"""Attack Graph models and container implementation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def stable_node_id(prefix: str, payload: dict[str, Any]) -> str:
    """Build a deterministic ID from a small structured payload."""

    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}::{digest}"


class StateNodeType(str, Enum):
    """Planner-visible state categories projected from KG facts."""

    HOST_KNOWN = "HOST_KNOWN"
    HOST_VALIDATED = "HOST_VALIDATED"
    SERVICE_KNOWN = "SERVICE_KNOWN"
    SERVICE_CONFIRMED = "SERVICE_CONFIRMED"
    PATH_CANDIDATE = "PATH_CANDIDATE"
    REACHABILITY_VALIDATED = "REACHABILITY_VALIDATED"
    IDENTITY_KNOWN = "IDENTITY_KNOWN"
    IDENTITY_AVAILABLE_ON_HOST = "IDENTITY_AVAILABLE_ON_HOST"
    CREDENTIAL_USABLE = "CREDENTIAL_USABLE"
    CREDENTIAL_REUSABLE_ON_HOST = "CREDENTIAL_REUSABLE_ON_HOST"
    MANAGED_SESSION_AVAILABLE = "MANAGED_SESSION_AVAILABLE"
    SESSION_ACTIVE_ON_HOST = "SESSION_ACTIVE_ON_HOST"
    IDENTITY_CONTEXT_KNOWN = "IDENTITY_CONTEXT_KNOWN"
    PRIVILEGE_VALIDATED = "PRIVILEGE_VALIDATED"
    PRIVILEGE_SOURCE_KNOWN = "PRIVILEGE_SOURCE_KNOWN"
    PIVOT_HOST_AVAILABLE = "PIVOT_HOST_AVAILABLE"
    LATERAL_SERVICE_EXPOSED = "LATERAL_SERVICE_EXPOSED"
    DATA_ASSET_KNOWN = "DATA_ASSET_KNOWN"
    GOAL_RELEVANT_DATA_LOCATED = "GOAL_RELEVANT_DATA_LOCATED"
    GOAL_STATE_SATISFIED = "GOAL_STATE_SATISFIED"


class ActionNodeType(str, Enum):
    """Controlled action-template categories for planning."""

    ENUMERATE_HOST = "ENUMERATE_HOST"
    VALIDATE_SERVICE = "VALIDATE_SERVICE"
    VALIDATE_REACHABILITY = "VALIDATE_REACHABILITY"
    ESTABLISH_PIVOT_ROUTE = "ESTABLISH_PIVOT_ROUTE"
    ESTABLISH_MANAGED_SESSION = "ESTABLISH_MANAGED_SESSION"
    REUSE_CREDENTIAL_ON_HOST = "REUSE_CREDENTIAL_ON_HOST"
    EXPLOIT_LATERAL_SERVICE = "EXPLOIT_LATERAL_SERVICE"
    ENUMERATE_IDENTITY_CONTEXT = "ENUMERATE_IDENTITY_CONTEXT"
    VALIDATE_PRIVILEGE_STATE = "VALIDATE_PRIVILEGE_STATE"
    LOCATE_GOAL_RELEVANT_DATA = "LOCATE_GOAL_RELEVANT_DATA"
    VALIDATE_GOAL_CONDITION = "VALIDATE_GOAL_CONDITION"


class GoalNodeType(str, Enum):
    """Goal categories used by the planner."""

    HOST_PROFILE_SUFFICIENT = "HOST_PROFILE_SUFFICIENT"
    TARGET_CONTEXT_VALIDATED = "TARGET_CONTEXT_VALIDATED"
    GOAL_RELEVANT_DATA_PRESENT = "GOAL_RELEVANT_DATA_PRESENT"
    OBJECTIVE_SATISFIED = "OBJECTIVE_SATISFIED"


class ConstraintNodeType(str, Enum):
    """Constraint categories that may gate planning branches."""

    SCOPE_BOUNDARY = "SCOPE_BOUNDARY"
    HOST_LOCK = "HOST_LOCK"
    SESSION_LOCK = "SESSION_LOCK"
    CONCURRENCY_LIMIT = "CONCURRENCY_LIMIT"
    TIME_BUDGET = "TIME_BUDGET"
    TOKEN_BUDGET = "TOKEN_BUDGET"
    NOISE_BUDGET = "NOISE_BUDGET"
    RISK_BUDGET = "RISK_BUDGET"
    APPROVAL_GATE = "APPROVAL_GATE"


class TruthStatus(str, Enum):
    """Truthiness of a projected state."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    VALIDATED = "validated"
    STALE = "stale"
    REVOKED = "revoked"


class ActivationStatus(str, Enum):
    """Runtime-free activation classification for planning nodes."""

    UNKNOWN = "unknown"
    ACTIVE = "active"
    ACTIVATABLE = "activatable"
    BLOCKED = "blocked"
    SATISFIED = "satisfied"
    DORMANT = "dormant"


class AGEdgeType(str, Enum):
    """Relationship types inside the Attack Graph."""

    REQUIRES = "REQUIRES"
    PRODUCES = "PRODUCES"
    ENABLES = "ENABLES"
    BLOCKED_BY = "BLOCKED_BY"
    COMPETES_WITH = "COMPETES_WITH"
    DOMINATES = "DOMINATES"


class GraphRef(BaseModel):
    """Reference to a source object in KG, AG or a derived query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph: Literal["kg", "ag", "tg", "query"]
    ref_id: str = Field(min_length=1)
    ref_type: str | None = None
    label: str | None = None

    def key(self) -> str:
        """Return a stable key for indexing."""

        return f"{self.graph}:{self.ref_id}"


class GraphBinding(BaseModel):
    """Structured binding from a logical argument name to a value or graph ref."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: str | int | float | bool | None = None
    graph_ref: GraphRef | None = None


class ActivationCondition(BaseModel):
    """Planner-facing activation constraint attached to an action."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    required_refs: list[GraphRef] = Field(default_factory=list)
    expression: dict[str, Any] = Field(default_factory=dict)
    status: ActivationStatus = ActivationStatus.UNKNOWN
    reason: str | None = None


class ProjectionTrace(BaseModel):
    """Trace metadata describing how an AG entity was projected."""

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(min_length=1)
    source_graph: str = "kg"
    input_refs: list[GraphRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    projected_at: datetime = Field(default_factory=utc_now)


class BaseAGNode(BaseModel):
    """Common fields shared by all Attack Graph nodes."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: Literal["state", "action", "goal", "constraint"]
    properties: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[GraphRef] = Field(default_factory=list)
    projection_traces: list[ProjectionTrace] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)


class BaseAGEdge(BaseModel):
    """Common fields shared by all Attack Graph edges."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    edge_type: AGEdgeType
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[GraphRef] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)


class StateNode(BaseAGNode):
    """Planner-visible state node derived from KG facts."""

    kind: Literal["state"] = "state"
    node_type: StateNodeType
    subject_refs: list[GraphRef] = Field(default_factory=list)
    truth_status: TruthStatus = TruthStatus.CANDIDATE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    created_from: list[GraphRef] = Field(default_factory=list)
    first_seen: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)
    ttl: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_time_window(self) -> "StateNode":
        """Keep state timestamps monotonic."""

        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must be greater than or equal to first_seen")
        return self


class ActionNode(BaseAGNode):
    """Controlled action template or bound planning action."""

    kind: Literal["action"] = "action"
    action_type: ActionNodeType
    bound_args: dict[str, Any] = Field(default_factory=dict)
    required_inputs: list[str] = Field(default_factory=list)
    precondition_schema: dict[str, Any] = Field(default_factory=dict)
    postcondition_schema: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: set[str] = Field(default_factory=set)
    cost: float = Field(default=0.0, ge=0.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    noise: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_value: float = Field(default=0.0, ge=0.0, le=1.0)
    success_probability_prior: float = Field(default=0.5, ge=0.0, le=1.0)
    goal_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    parallelizable: bool = False
    cooldown_seconds: int = Field(default=0, ge=0)
    retry_policy: dict[str, Any] = Field(default_factory=dict)
    approval_required: bool = False
    resource_keys: set[str] = Field(default_factory=set)
    activation_status: ActivationStatus = ActivationStatus.UNKNOWN
    activation_conditions: list[ActivationCondition] = Field(default_factory=list)


class GoalNode(BaseAGNode):
    """Planner target or intermediate planning objective."""

    kind: Literal["goal"] = "goal"
    goal_type: GoalNodeType
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=50, ge=0)
    business_value: float = Field(default=0.5, ge=0.0, le=1.0)
    scope_refs: list[GraphRef] = Field(default_factory=list)


class ConstraintNode(BaseAGNode):
    """Budget, scope, approval or concurrency restriction."""

    kind: Literal["constraint"] = "constraint"
    constraint_type: ConstraintNodeType
    hard_or_soft: Literal["hard", "soft"] = "hard"
    budget_value: float | int | None = None
    current_usage: float | int | None = None
    applies_to: list[GraphRef] = Field(default_factory=list)
    activation_status: ActivationStatus = ActivationStatus.ACTIVE


AGNode = StateNode | ActionNode | GoalNode | ConstraintNode


NODE_KIND_MAP: dict[str, type[BaseAGNode]] = {
    "state": StateNode,
    "action": ActionNode,
    "goal": GoalNode,
    "constraint": ConstraintNode,
}


class AttackGraph:
    """In-memory Attack Graph with small indexes for planning."""

    def __init__(self) -> None:
        self._nodes: dict[str, AGNode] = {}
        self._edges: dict[str, BaseAGEdge] = {}
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
        用于把当前 AG 快照与来源 KG 版本、投影批次和下游 TG 构建关联起来。
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

    def add_edge(self, edge: BaseAGEdge) -> BaseAGEdge:
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

    def get_node(self, node_id: str) -> AGNode:
        """Return a node by ID."""

        return self._nodes[node_id]

    def get_edge(self, edge_id: str) -> BaseAGEdge:
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

    def remove_edge(self, edge_id: str) -> BaseAGEdge:
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

    def list_edges(self, edge_type: AGEdgeType | str | None = None) -> list[BaseAGEdge]:
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
        edge_type: AGEdgeType | str | None = None,
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

    def find_states(
        self,
        state_type: StateNodeType | str | None = None,
        active_only: bool = False,
    ) -> list[StateNode]:
        """Return projected state nodes."""

        states = [node for node in self.list_nodes(state_type) if isinstance(node, StateNode)]
        if active_only:
            states = [
                node
                for node in states
                if node.truth_status in {TruthStatus.ACTIVE, TruthStatus.VALIDATED}
            ]
        return states

    def find_actions(
        self,
        action_type: ActionNodeType | str | None = None,
        activatable_only: bool = False,
    ) -> list[ActionNode]:
        """Return action nodes, optionally only activatable ones."""

        actions = [node for node in self.list_nodes(action_type) if isinstance(node, ActionNode)]
        if activatable_only:
            actions = [
                node
                for node in actions
                if node.activation_status in {ActivationStatus.ACTIVATABLE, ActivationStatus.SATISFIED}
            ]
        return actions

    def get_goal_nodes(self) -> list[GoalNode]:
        """Return all goal nodes."""

        return [node for node in self.list_nodes() if isinstance(node, GoalNode)]

    def get_constraint_nodes(self) -> list[ConstraintNode]:
        """Return all constraint nodes."""

        return [node for node in self.list_nodes() if isinstance(node, ConstraintNode)]

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
        """Restore a graph from serialized data."""

        graph = cls()
        for node_data in payload.get("nodes", []):
            graph.add_node(parse_ag_node(node_data))
        for edge_data in payload.get("edges", []):
            graph.add_edge(BaseAGEdge.model_validate(edge_data))
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
        if isinstance(node, StateNode):
            return node.node_type.value
        if isinstance(node, ActionNode):
            return node.action_type.value
        if isinstance(node, GoalNode):
            return node.goal_type.value
        return node.constraint_type.value

    @staticmethod
    def _refs_for_index(node: AGNode) -> list[GraphRef]:
        refs: list[GraphRef] = []
        if isinstance(node, StateNode):
            refs.extend(node.subject_refs)
            refs.extend(node.created_from)
        elif isinstance(node, GoalNode):
            refs.extend(node.scope_refs)
        elif isinstance(node, ConstraintNode):
            refs.extend(node.applies_to)
        refs.extend(node.source_refs)
        unique: dict[str, GraphRef] = {ref.key(): ref for ref in refs}
        return list(unique.values())


def parse_ag_node(data: dict[str, Any]) -> AGNode:
    """Instantiate a typed AG node from serialized data."""

    return NODE_KIND_MAP[data["kind"]].model_validate(data)
