"""Goal validator abstractions used by GoalWorker."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.events import AgentTaskRequest


class GoalEvaluation(BaseModel):
    """Standardized goal validation result consumed by GoalWorker."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    satisfied: bool = True
    missing_requirements: list[str] = Field(default_factory=list)
    validated_ref_ids: list[str] = Field(default_factory=list)
    supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    blocked: bool = False
    failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoalValidator(ABC):
    """Interface for producing one standardized goal evaluation."""

    @abstractmethod
    def evaluate(self, request: AgentTaskRequest) -> GoalEvaluation:
        """Return one structured goal evaluation for the request."""


class MetadataGoalValidator(GoalValidator):
    """Compatibility validator that normalizes metadata-backed goal results."""

    def evaluate(self, request: AgentTaskRequest) -> GoalEvaluation:
        raw = request.metadata.get("goal_validator_output") or request.context.metadata.get("goal_validator_output")
        if raw is None:
            raw = request.metadata.get("goal_evaluation") or request.context.metadata.get("goal_evaluation") or {}
        if "satisfied" not in raw:
            raw = {"satisfied": bool(request.metadata.get("goal_satisfied", True)), **dict(raw)}
        validated_ref_ids = raw.get("validated_ref_ids")
        if not isinstance(validated_ref_ids, list):
            validated_ref_ids = [ref.ref_id for ref in request.target_refs]
        supporting_evidence = raw.get("supporting_evidence")
        if not isinstance(supporting_evidence, list):
            supporting_evidence = []
        return GoalEvaluation(
            satisfied=bool(raw.get("satisfied", True)),
            missing_requirements=[str(item) for item in raw.get("missing_requirements", []) if str(item).strip()],
            validated_ref_ids=[str(item) for item in validated_ref_ids if str(item).strip()],
            supporting_evidence=[dict(item) for item in supporting_evidence if isinstance(item, dict)],
            confidence=float(raw.get("confidence", request.metadata.get("confidence", 0.9))),
            blocked=bool(raw.get("blocked", False)),
            failure_reason=(
                str(raw.get("failure_reason") or raw.get("reason"))
                if raw.get("failure_reason") is not None or raw.get("reason") is not None
                else None
            ),
            metadata={
                k: v
                for k, v in dict(raw).items()
                if k
                not in {
                    "satisfied",
                    "missing_requirements",
                    "validated_ref_ids",
                    "supporting_evidence",
                    "confidence",
                    "blocked",
                    "failure_reason",
                    "reason",
                }
            },
        )


__all__ = ["GoalEvaluation", "GoalValidator", "MetadataGoalValidator"]
