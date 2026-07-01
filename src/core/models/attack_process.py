"""Attack-process graph models for AG process history."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.graph_common import stable_node_id, utc_now


class AttackProcessNodeType(str, Enum):
    """Result-tier node types stored in AG."""

    ATTACK_STEP = "ATTACK_STEP"


class AttackProcessEdgeType(str, Enum):
    """Result-tier edge types connecting AG records."""

    NEXT = "NEXT"


class AttackProcessNode(BaseModel):
    """An AG node recording one attack-process event or decision."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    node_type: AttackProcessNodeType
    label: str = Field(min_length=1)
    kind: Literal["process"] = "process"
    operation_id: str = Field(min_length=1)
    cycle_index: int | None = Field(default=None, ge=0)
    agent_name: str | None = None
    status: str | None = None
    summary: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AttackStepNode(AttackProcessNode):
    """One execution round's *result*, recorded as a single AG node.

    Coarse as a node, precise via links: ``kg_node_refs`` points at the KG nodes
    this round produced/used. Full per-round detail lives in the operation trace
    log (operation-trace.txt) and KG, not in AG.
    """

    node_type: Literal[AttackProcessNodeType.ATTACK_STEP] = AttackProcessNodeType.ATTACK_STEP
    capability: str | None = None
    kg_node_refs: list[str] = Field(default_factory=list)


class AttackProcessEdge(BaseModel):
    """An AG edge connecting attack-process nodes."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(min_length=1)
    edge_type: AttackProcessEdgeType
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


__all__ = [
    "AttackProcessEdge",
    "AttackProcessEdgeType",
    "AttackStepNode",
    "AttackProcessNode",
    "AttackProcessNodeType",
    "stable_node_id",
]
