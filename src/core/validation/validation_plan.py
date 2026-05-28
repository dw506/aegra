"""Execution plan for one controlled validation profile."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationPlan(BaseModel):
    """Bounded validation plan built from a profile and concrete target."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1)
    target_ref: str = Field(min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    tool_sequence: list[dict[str, Any]] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    max_attempts: int = Field(default=2, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    risk_level: str = "low"


__all__ = ["ValidationPlan"]
