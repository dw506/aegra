"""Shared worker service result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkerDomainResult(BaseModel):
    """Stable domain result shape for worker validation services."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    success: bool
    status: str
    summary: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    fact_write_requests: list[dict[str, Any]] = Field(default_factory=list)
    projection_requests: list[dict[str, Any]] = Field(default_factory=list)
    runtime_requests: list[dict[str, Any]] = Field(default_factory=list)
    critic_signals: list[dict[str, Any]] = Field(default_factory=list)
    replan_hints: list[dict[str, Any]] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


__all__ = ["WorkerDomainResult"]
