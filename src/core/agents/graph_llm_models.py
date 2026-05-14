"""Structured graph-planning proposals produced by LLM advisors."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import new_record_id
from src.core.models.ag import GraphRef


class GraphLLMTaskProposal(BaseModel):
    """One bounded proposal for a future graph task.

    The proposal is intentionally declarative. It can name target graph refs,
    task category, expected evidence and planner hints, but it cannot carry a
    command or payload for direct execution.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    proposal_id: str = Field(default_factory=lambda: new_record_id("graph-llm-task"))
    task_type: str = Field(min_length=1)
    target_refs: list[GraphRef] = Field(default_factory=list)
    rationale: str | None = None
    expected_evidence: list[str] = Field(default_factory=list)
    tool_hint: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    estimated_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_noise: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=50, ge=0, le=100)
    requires_human_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphLLMRankAdjustment(BaseModel):
    """Bounded rank adjustment for an existing graph/planner item."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    target_ref: GraphRef
    score_delta: float = Field(default=0.0, ge=-0.5, le=0.5)
    rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphLLMPlanProposal(BaseModel):
    """Top-level graph-planning proposal returned by an LLM."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    proposal_id: str = Field(default_factory=lambda: new_record_id("graph-llm-plan"))
    task_proposals: list[GraphLLMTaskProposal] = Field(default_factory=list)
    rank_adjustments: list[GraphLLMRankAdjustment] = Field(default_factory=list)
    replan_hint: str | None = None
    risk_notes: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphLLMPlanValidationResult(BaseModel):
    """Result of validating a graph LLM planning proposal."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    accepted: bool = False
    requires_human_review: bool = False
    reason: str = Field(min_length=1)
    sanitized_payload: dict[str, Any] = Field(default_factory=dict)
    rejected_items: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def accepted_result(
        cls,
        *,
        sanitized_payload: dict[str, Any],
        requires_human_review: bool = False,
        reason: str = "accepted",
    ) -> "GraphLLMPlanValidationResult":
        return cls(
            accepted=True,
            requires_human_review=requires_human_review,
            reason=reason,
            sanitized_payload=sanitized_payload,
        )

    @classmethod
    def rejected_result(
        cls,
        *,
        reason: str,
        sanitized_payload: dict[str, Any] | None = None,
        rejected_items: list[dict[str, Any]] | None = None,
    ) -> "GraphLLMPlanValidationResult":
        return cls(
            accepted=False,
            requires_human_review=False,
            reason=reason,
            sanitized_payload=sanitized_payload or {},
            rejected_items=rejected_items or [],
        )


__all__ = [
    "GraphLLMPlanProposal",
    "GraphLLMPlanValidationResult",
    "GraphLLMRankAdjustment",
    "GraphLLMTaskProposal",
]
