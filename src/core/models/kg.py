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
    source_refs: list[GraphEntityRef] = Field(default_factory=list)
    first_seen: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)
    ttl: int | None = Field(default=None, ge=1)
    tags: set[str] = Field(default_factory=set)

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


class Identity(BaseNode):
    type: Literal[NodeType.IDENTITY] = NodeType.IDENTITY
    username: str | None = None
    domain: str | None = None


class Credential(BaseNode):
    type: Literal[NodeType.CREDENTIAL] = NodeType.CREDENTIAL
    credential_kind: str | None = None
    principal: str | None = None


class Session(BaseNode):
    type: Literal[NodeType.SESSION] = NodeType.SESSION
    session_kind: str | None = None
    session_state: str | None = None


class PrivilegeState(BaseNode):
    type: Literal[NodeType.PRIVILEGE_STATE] = NodeType.PRIVILEGE_STATE
    privilege_level: str | None = None
    scope: str | None = None


class DataAsset(BaseNode):
    type: Literal[NodeType.DATA_ASSET] = NodeType.DATA_ASSET
    asset_kind: str | None = None
    sensitivity: str | None = None


class Observation(BaseNode):
    type: Literal[NodeType.OBSERVATION] = NodeType.OBSERVATION
    observation_kind: str | None = None
    observed_at: datetime = Field(default_factory=utc_now)
    summary: str | None = None


class Evidence(BaseNode):
    type: Literal[NodeType.EVIDENCE] = NodeType.EVIDENCE
    evidence_kind: str | None = None
    content_hash: str | None = None
    summary: str | None = None


class Finding(BaseNode):
    type: Literal[NodeType.FINDING] = NodeType.FINDING
    finding_kind: str | None = None
    severity: str | None = None
    summary: str | None = None


class NetworkZone(BaseNode):
    type: Literal[NodeType.NETWORK_ZONE] = NodeType.NETWORK_ZONE
    cidr: str | None = None
    zone_kind: str | None = None


class Goal(BaseNode):
    type: Literal[NodeType.GOAL] = NodeType.GOAL
    category: str | None = None
    description: str | None = None


class HostsEdge(BaseEdge):
    type: Literal[EdgeType.HOSTS] = EdgeType.HOSTS


class BelongsToZoneEdge(BaseEdge):
    type: Literal[EdgeType.BELONGS_TO_ZONE] = EdgeType.BELONGS_TO_ZONE


class IdentityPresentOnEdge(BaseEdge):
    type: Literal[EdgeType.IDENTITY_PRESENT_ON] = EdgeType.IDENTITY_PRESENT_ON


class IdentityAvailableOnEdge(BaseEdge):
    type: Literal[EdgeType.IDENTITY_AVAILABLE_ON] = EdgeType.IDENTITY_AVAILABLE_ON


class AuthenticatesAsEdge(BaseEdge):
    type: Literal[EdgeType.AUTHENTICATES_AS] = EdgeType.AUTHENTICATES_AS


class ReusesCredentialEdge(BaseEdge):
    type: Literal[EdgeType.REUSES_CREDENTIAL] = EdgeType.REUSES_CREDENTIAL


class SessionForEdge(BaseEdge):
    type: Literal[EdgeType.SESSION_FOR] = EdgeType.SESSION_FOR


class SessionOnEdge(BaseEdge):
    type: Literal[EdgeType.SESSION_ON] = EdgeType.SESSION_ON


class HasPrivilegeStateEdge(BaseEdge):
    type: Literal[EdgeType.HAS_PRIVILEGE_STATE] = EdgeType.HAS_PRIVILEGE_STATE


class PrivilegeSourceEdge(BaseEdge):
    type: Literal[EdgeType.PRIVILEGE_SOURCE] = EdgeType.PRIVILEGE_SOURCE


class AppliesToHostEdge(BaseEdge):
    type: Literal[EdgeType.APPLIES_TO_HOST] = EdgeType.APPLIES_TO_HOST


class CanReachEdge(BaseEdge):
    type: Literal[EdgeType.CAN_REACH] = EdgeType.CAN_REACH


class PivotsToEdge(BaseEdge):
    type: Literal[EdgeType.PIVOTS_TO] = EdgeType.PIVOTS_TO


class ObservedOnEdge(BaseEdge):
    type: Literal[EdgeType.OBSERVED_ON] = EdgeType.OBSERVED_ON


class SupportedByEdge(BaseEdge):
    type: Literal[EdgeType.SUPPORTED_BY] = EdgeType.SUPPORTED_BY


class DerivedFromEdge(BaseEdge):
    type: Literal[EdgeType.DERIVED_FROM] = EdgeType.DERIVED_FROM


class RelatedToEdge(BaseEdge):
    type: Literal[EdgeType.RELATED_TO] = EdgeType.RELATED_TO


class ContainsEdge(BaseEdge):
    type: Literal[EdgeType.CONTAINS] = EdgeType.CONTAINS


class TargetsEdge(BaseEdge):
    type: Literal[EdgeType.TARGETS] = EdgeType.TARGETS


class CoOccursWithEdge(BaseEdge):
    type: Literal[EdgeType.CO_OCCURS_WITH] = EdgeType.CO_OCCURS_WITH


NODE_MODEL_BY_TYPE: dict[NodeType, type[BaseNode]] = {
    NodeType.HOST: Host,
    NodeType.SERVICE: Service,
    NodeType.IDENTITY: Identity,
    NodeType.CREDENTIAL: Credential,
    NodeType.SESSION: Session,
    NodeType.PRIVILEGE_STATE: PrivilegeState,
    NodeType.DATA_ASSET: DataAsset,
    NodeType.OBSERVATION: Observation,
    NodeType.EVIDENCE: Evidence,
    NodeType.FINDING: Finding,
    NodeType.NETWORK_ZONE: NetworkZone,
    NodeType.GOAL: Goal,
}


EDGE_MODEL_BY_TYPE: dict[EdgeType, type[BaseEdge]] = {
    EdgeType.HOSTS: HostsEdge,
    EdgeType.BELONGS_TO_ZONE: BelongsToZoneEdge,
    EdgeType.IDENTITY_PRESENT_ON: IdentityPresentOnEdge,
    EdgeType.IDENTITY_AVAILABLE_ON: IdentityAvailableOnEdge,
    EdgeType.AUTHENTICATES_AS: AuthenticatesAsEdge,
    EdgeType.REUSES_CREDENTIAL: ReusesCredentialEdge,
    EdgeType.SESSION_FOR: SessionForEdge,
    EdgeType.SESSION_ON: SessionOnEdge,
    EdgeType.HAS_PRIVILEGE_STATE: HasPrivilegeStateEdge,
    EdgeType.PRIVILEGE_SOURCE: PrivilegeSourceEdge,
    EdgeType.APPLIES_TO_HOST: AppliesToHostEdge,
    EdgeType.CAN_REACH: CanReachEdge,
    EdgeType.PIVOTS_TO: PivotsToEdge,
    EdgeType.OBSERVED_ON: ObservedOnEdge,
    EdgeType.SUPPORTED_BY: SupportedByEdge,
    EdgeType.DERIVED_FROM: DerivedFromEdge,
    EdgeType.RELATED_TO: RelatedToEdge,
    EdgeType.CONTAINS: ContainsEdge,
    EdgeType.TARGETS: TargetsEdge,
    EdgeType.CO_OCCURS_WITH: CoOccursWithEdge,
}


def parse_node(data: JsonDict) -> BaseNode:
    """Instantiate a typed node from serialized data."""

    node_type = NodeType(data["type"])
    return NODE_MODEL_BY_TYPE[node_type].model_validate(data)


def parse_edge(data: JsonDict) -> BaseEdge:
    """Instantiate a typed edge from serialized data."""

    edge_type = EdgeType(data["type"])
    return EDGE_MODEL_BY_TYPE[edge_type].model_validate(data)
