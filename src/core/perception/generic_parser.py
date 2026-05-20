"""Generic fallback parser for worker results."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_models import OutcomeRecord
from src.core.perception.parser_protocol import ParsedWorkerResult


class GenericParser:
    """Fallback parser that preserves the existing generic perception behavior."""

    name = "generic_parser"

    def supports(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> bool:
        return True

    def parse(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> ParsedWorkerResult:
        return ParsedWorkerResult(metadata={"parser": self.name})


__all__ = ["GenericParser"]
