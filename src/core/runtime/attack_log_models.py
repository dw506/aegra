"""Structured attack-log extraction output models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.attack_process import AttackProcessEdge, AttackProcessNode


class AttackLogExtraction(BaseModel):
    """AG process mutations extracted from planner, stage and tool records."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    operation_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    ag_nodes: list[AttackProcessNode] = Field(default_factory=list)
    ag_edges: list[AttackProcessEdge] = Field(default_factory=list)
    summary: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["AttackLogExtraction"]
