"""Registry for worker result parser plugins."""

from __future__ import annotations

from typing import Any, Iterable

from src.core.agents.agent_models import OutcomeRecord
from src.core.perception.generic_parser import GenericParser
from src.core.perception.parser_protocol import ParsedWorkerResult, ResultParser


class ParserRegistry:
    """Ordered parser registry with a generic fallback parser."""

    def __init__(self, parsers: Iterable[ResultParser] | None = None) -> None:
        self.parsers: list[ResultParser] = list(parsers or [])
        self._generic = GenericParser()

    @classmethod
    def default(cls) -> "ParserRegistry":
        return cls()

    def register(self, parser: ResultParser) -> ResultParser:
        self.parsers.append(parser)
        return parser

    def parse(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> ParsedWorkerResult:
        for parser in self.parsers:
            if parser.supports(raw_result, outcome):
                parsed = parser.parse(raw_result, outcome)
                metadata = dict(parsed.metadata)
                metadata.setdefault("parser", parser.name)
                return parsed.model_copy(update={"metadata": metadata})
        return self._generic.parse(raw_result, outcome)


__all__ = ["ParserRegistry"]
