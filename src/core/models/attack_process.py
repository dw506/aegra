"""Attack-process graph models for AG process history."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.graph_common import GraphRef, stable_node_id, utc_now


class AttackProcessNodeType(str, Enum):
    """Node types for attack-process records stored in AG."""

    ATTACK_CYCLE = "ATTACK_CYCLE"
    PLANNER_DECISION = "PLANNER_DECISION"
    AGENT_EXECUTION = "AGENT_EXECUTION"
    TOOL_CALL = "TOOL_CALL"
    STAGE_RESULT = "STAGE_RESULT"
    HANDOFF_SUGGESTION = "HANDOFF_SUGGESTION"
    BLOCKED_REASON = "BLOCKED_REASON"
    GOAL_CHECK = "GOAL_CHECK"


class AttackProcessEdgeType(str, Enum):
    """Edge types connecting attack-process records in AG."""

    NEXT_CYCLE = "NEXT_CYCLE"
    PLANNED = "PLANNED"
    DISPATCHED_TO = "DISPATCHED_TO"
    CALLED_TOOL = "CALLED_TOOL"
    PRODUCED_RESULT = "PRODUCED_RESULT"
    SUPPORTED_BY_EVIDENCE = "SUPPORTED_BY_EVIDENCE"
    UPDATED_KG = "UPDATED_KG"
    SUGGESTED_HANDOFF = "SUGGESTED_HANDOFF"
    ACCEPTED_BY_PLANNER = "ACCEPTED_BY_PLANNER"
    REJECTED_BY_PLANNER = "REJECTED_BY_PLANNER"
    BLOCKED_BY_POLICY = "BLOCKED_BY_POLICY"
    SATISFIED_GOAL = "SATISFIED_GOAL"


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
    stage_type: str | None = None
    status: str | None = None
    summary: str | None = None
    refs: list[GraphRef] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AttackCycleNode(AttackProcessNode):
    """A single planner/stage cycle recorded as an attack-process node."""

    node_type: Literal[AttackProcessNodeType.ATTACK_CYCLE] = AttackProcessNodeType.ATTACK_CYCLE


class PlannerDecisionNode(AttackProcessNode):
    """A planner decision recorded as an attack-process node."""

    node_type: Literal[AttackProcessNodeType.PLANNER_DECISION] = AttackProcessNodeType.PLANNER_DECISION


class AgentExecutionNode(AttackProcessNode):
    """An agent execution recorded as an attack-process node."""

    node_type: Literal[AttackProcessNodeType.AGENT_EXECUTION] = AttackProcessNodeType.AGENT_EXECUTION


class ToolCallNode(AttackProcessNode):
    """A tool invocation recorded as an attack-process node."""

    node_type: Literal[AttackProcessNodeType.TOOL_CALL] = AttackProcessNodeType.TOOL_CALL


class StageResultNode(AttackProcessNode):
    """A stage result recorded as an attack-process node."""

    node_type: Literal[AttackProcessNodeType.STAGE_RESULT] = AttackProcessNodeType.STAGE_RESULT


class HandoffSuggestionNode(AttackProcessNode):
    """A stage-to-planner handoff suggestion recorded as an attack-process node."""

    node_type: Literal[AttackProcessNodeType.HANDOFF_SUGGESTION] = AttackProcessNodeType.HANDOFF_SUGGESTION


class BlockedReasonNode(AttackProcessNode):
    """A policy/runtime blocked reason recorded as an attack-process node."""

    node_type: Literal[AttackProcessNodeType.BLOCKED_REASON] = AttackProcessNodeType.BLOCKED_REASON


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
    "AttackCycleNode",
    "AttackProcessNode",
    "AttackProcessNodeType",
    "AgentExecutionNode",
    "BlockedReasonNode",
    "GraphRef",
    "HandoffSuggestionNode",
    "PlannerDecisionNode",
    "StageResultNode",
    "stable_node_id",
    "ToolCallNode",
]
