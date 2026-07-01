"""Pydantic models for the knowledge graph module."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.models.kg_enums import ChangeOperation, EdgeType, EntityStatus, NodeType
from src.core.models.kg_types import JsonDict, Properties


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


class GraphEntityRef(BaseModel):
    """Stable reference to a graph entity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1)
    entity_kind: Literal["node", "edge"] = "node"
    entity_type: str | None = None
    label: str | None = None

    def key(self) -> str:
        """Return a storage-safe composite key."""

        return f"{self.entity_kind}:{self.entity_id}"


class GraphChange(BaseModel):
    """Single mutation event recorded by the graph."""

    model_config = ConfigDict(extra="forbid")

    operation: ChangeOperation
    entity_ref: GraphEntityRef
    before: JsonDict | None = None
    after: JsonDict | None = None
    changed_at: datetime = Field(default_factory=utc_now)


class GraphDelta(BaseModel):
    """Collection of graph changes."""

    model_config = ConfigDict(extra="forbid")

    # version 表示当前 KG 快照的逻辑版本号；每次结构变更都会推进。
    version: int = Field(default=0, ge=0)
    change_count: int = Field(default=0, ge=0)
    last_patch_batch_id: str | None = None
    last_changed_at: datetime | None = None
    changes: list[GraphChange] = Field(default_factory=list)


class BaseGraphEntity(BaseModel):
    """Common fields shared by nodes and edges."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    properties: Properties = Field(default_factory=dict)
    status: EntityStatus = EntityStatus.OBSERVED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # Single evidence-pointer field for every node/edge (was split evidence_ids
    # vs the typed evidence_refs on Finding/GoalCheck/GoalProof — merged onto the
    # project-wide `evidence_refs` name, which ExecutionResult/AttackStepNode/
    # Finding/evaluation models all already use). The former evidence_ids was
    # dead on the KG write path (ResultApplier only ever emitted evidence_refs).
    evidence_refs: list[str] = Field(default_factory=list)
    first_seen: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_time_window(self) -> "BaseGraphEntity":
        """Ensure timestamps remain monotonic."""

        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must be greater than or equal to first_seen")
        return self

    def to_ref(self, kind: Literal["node", "edge"]) -> GraphEntityRef:
        """Return a stable graph reference for the entity."""

        return GraphEntityRef(
            entity_id=self.id,
            entity_kind=kind,
            entity_type=getattr(self, "type").value,
            label=self.label,
        )


class BaseNode(BaseGraphEntity):
    """Base class for graph nodes."""

    type: NodeType


class BaseEdge(BaseGraphEntity):
    """Base class for graph edges."""

    type: EdgeType
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)

#节点类型定义为 Pydantic 类
class Host(BaseNode):
    type: Literal[NodeType.HOST] = NodeType.HOST
    hostname: str | None = None
    address: str | None = None
    platform: str | None = None


class Service(BaseNode):
    type: Literal[NodeType.SERVICE] = NodeType.SERVICE
    service_name: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    protocol: str | None = None


class Session(BaseNode):
    type: Literal[NodeType.SESSION] = NodeType.SESSION
    session_kind: str | None = None
    session_state: str | None = None


class Evidence(BaseNode):
    type: Literal[NodeType.EVIDENCE] = NodeType.EVIDENCE
    evidence_kind: str | None = None
    content_hash: str | None = None
    content_ref: str | None = None
    summary: str | None = None


class Finding(BaseNode):
    type: Literal[NodeType.FINDING] = NodeType.FINDING
    finding_kind: str | None = None
    affected_asset_refs: list[str] = Field(default_factory=list)
    service_ref: str | None = None
    vulnerability_ref: str | None = None
    validation_status: str | None = None
    severity: str | None = None
    cvss: float | None = Field(default=None, ge=0.0, le=10.0)
    epss: float | None = Field(default=None, ge=0.0, le=1.0)
    kev: bool = False
    false_positive_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    remediation: str | None = None
    risk_score: float | None = Field(default=None, ge=0.0, le=100.0)
    summary: str | None = None


class NetworkZone(BaseNode):
    type: Literal[NodeType.NETWORK_ZONE] = NodeType.NETWORK_ZONE
    cidr: str | None = None
    zone_kind: str | None = None


class Goal(BaseNode):
    type: Literal[NodeType.GOAL] = NodeType.GOAL
    category: str | None = None
    description: str | None = None


class PostAccessObservation(BaseNode):
    type: Literal[NodeType.POST_ACCESS_OBSERVATION] = NodeType.POST_ACCESS_OBSERVATION
    target_ref: str | None = None
    session_ref: str | None = None
    observation_path: str | None = None
    zone_ref: str | None = None


class GoalCheck(BaseNode):
    type: Literal[NodeType.GOAL_CHECK] = NodeType.GOAL_CHECK
    goal_id: str | None = None
    passed: bool = False
    redacted_summary: str | None = None
    proof_token: str | None = None


class GoalProof(BaseNode):
    type: Literal[NodeType.GOAL_PROOF] = NodeType.GOAL_PROOF
    goal_id: str | None = None
    proof_token: str | None = None
    redacted_summary: str | None = None


class PivotRouteNode(BaseNode):
    """KG representation of an established pivot route."""

    type: Literal[NodeType.PIVOT_ROUTE] = NodeType.PIVOT_ROUTE
    route_id: str | None = None
    from_zone_ref: str | None = None
    to_zone_ref: str | None = None
    via_host: str | None = None
    session_ref: str | None = None
    route_type: str | None = None

#边
class HostsEdge(BaseEdge):
    type: Literal[EdgeType.HOSTS] = EdgeType.HOSTS


class BelongsToZoneEdge(BaseEdge):
    type: Literal[EdgeType.BELONGS_TO_ZONE] = EdgeType.BELONGS_TO_ZONE


class SupportedByEdge(BaseEdge):
    type: Literal[EdgeType.SUPPORTED_BY] = EdgeType.SUPPORTED_BY
    evidence_kind: str | None = None


class TargetsEdge(BaseEdge):
    type: Literal[EdgeType.TARGETS] = EdgeType.TARGETS


NODE_MODEL_BY_TYPE: dict[NodeType, type[BaseNode]] = {
    NodeType.HOST: Host,
    NodeType.SERVICE: Service,
    NodeType.SESSION: Session,
    NodeType.EVIDENCE: Evidence,
    NodeType.FINDING: Finding,
    NodeType.NETWORK_ZONE: NetworkZone,
    NodeType.GOAL: Goal,
    NodeType.POST_ACCESS_OBSERVATION: PostAccessObservation,
    NodeType.GOAL_CHECK: GoalCheck,
    NodeType.GOAL_PROOF: GoalProof,
    NodeType.PIVOT_ROUTE: PivotRouteNode,
}


EDGE_MODEL_BY_TYPE: dict[EdgeType, type[BaseEdge]] = {
    EdgeType.HOSTS: HostsEdge,
    EdgeType.BELONGS_TO_ZONE: BelongsToZoneEdge,
    EdgeType.SUPPORTED_BY: SupportedByEdge,
    EdgeType.TARGETS: TargetsEdge,
}


def parse_node(data: JsonDict) -> BaseNode:
    """Instantiate a typed node from serialized data."""

    node_type = NodeType(data["type"])
    return NODE_MODEL_BY_TYPE[node_type].model_validate(data)


def parse_edge(data: JsonDict) -> BaseEdge:
    """Instantiate a typed edge from serialized data."""

    edge_type = EdgeType(data["type"])
    return EDGE_MODEL_BY_TYPE[edge_type].model_validate(data)
