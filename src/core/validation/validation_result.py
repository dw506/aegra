"""Normalized safe validation result model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ValidationStatus = Literal["validated", "suspected", "not_detected", "blocked", "failed"]


class ValidationResult(BaseModel):
    """Compact validation result suitable for parsed MCP tool output."""

    model_config = ConfigDict(extra="forbid")

    vulnerability_id: str = Field(min_length=1)
    status: ValidationStatus
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    safe_payload_summary: str = Field(default="no exploit payload executed", min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    tool: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None


__all__ = ["ValidationResult", "ValidationStatus"]
