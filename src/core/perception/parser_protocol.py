"""Parser protocol for worker result perception plugins."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import OutcomeRecord


class ParsedWorkerResult(BaseModel):
    """Normalized parser output consumed by `PerceptionAgent`."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    observations: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    fact_write_requests: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultParser(Protocol):
    """Plugin interface for worker raw-result parsers."""

    name: str

    def supports(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> bool:
        """Return True when this parser can handle the worker result."""

    def parse(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> ParsedWorkerResult:
        """Parse a raw worker result into normalized perception payloads."""


__all__ = ["ParsedWorkerResult", "ResultParser"]
