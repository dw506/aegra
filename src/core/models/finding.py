"""Finding, evidence and risk-score models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


FindingValidationStatus = Literal["validated", "suspected"]
FindingSeverity = Literal["informational", "low", "medium", "high", "critical"]


class BaseFindingModel(BaseModel):
    """Shared validation settings for finding-domain models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RiskScore(BaseFindingModel):
    """Normalized risk score attached to one finding."""

    score: float = Field(ge=0.0, le=100.0)
    severity: FindingSeverity
    factors: dict[str, Any] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)


class EvidenceArtifactRecord(BaseFindingModel):
    """Operation-level normalized evidence artifact."""

    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    payload_ref: str = Field(min_length=1)
    task_ref: str = Field(min_length=1)
    tool_output_ref: str | None = None
    refs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class Finding(BaseFindingModel):
    """Formal vulnerability finding promoted from validation evidence."""

    finding_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    affected_asset_refs: list[str] = Field(default_factory=list)
    service_ref: str = Field(min_length=1)
    vulnerability_ref: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    validation_status: FindingValidationStatus
    severity: FindingSeverity
    cvss: float | None = Field(default=None, ge=0.0, le=10.0)
    epss: float | None = Field(default=None, ge=0.0, le=1.0)
    kev: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    false_positive_risk: float = Field(ge=0.0, le=1.0)
    remediation: str = Field(default="")
    risk_score: RiskScore
    created_at: datetime = Field(default_factory=utc_now)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("affected_asset_refs", "evidence_refs")
    @classmethod
    def _dedupe_strings(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result


__all__ = [
    "EvidenceArtifactRecord",
    "Finding",
    "FindingSeverity",
    "FindingValidationStatus",
    "RiskScore",
    "utc_now",
]
