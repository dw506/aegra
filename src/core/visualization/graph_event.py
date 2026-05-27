"""Read-only graph visualization event models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


GraphName = Literal["kg", "ag", "tg", "runtime"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GraphOperation(str, Enum):
    UPSERT_NODE = "upsert_node"
    UPSERT_EDGE = "upsert_edge"
    DELETE_NODE = "delete_node"
    DELETE_EDGE = "delete_edge"
    UPDATE_STATUS = "update_status"


class VisualNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    type: str | None = None
    graph: GraphName
    properties: dict[str, Any] = Field(default_factory=dict)
    status: str | None = None


class VisualEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str | None = None
    type: str | None = None
    graph: GraphName
    properties: dict[str, Any] = Field(default_factory=dict)


class VisualGraphChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: GraphOperation
    entity_id: str = Field(min_length=1)
    entity_type: str | None = None
    label: str | None = None
    source: str | None = None
    target: str | None = None
    edge_type: str | None = None
    status: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class VisualGraphDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["graph_delta"] = "graph_delta"
    operation_id: str = Field(min_length=1)
    graph: GraphName
    version: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=utc_now)
    changes: list[VisualGraphChange] = Field(default_factory=list)


class VisualGraphState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=0, ge=0)
    nodes: list[VisualNode] = Field(default_factory=list)
    edges: list[VisualEdge] = Field(default_factory=list)


class VisualGraphSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["graph_snapshot"] = "graph_snapshot"
    operation_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=utc_now)
    graphs: dict[GraphName, VisualGraphState]
