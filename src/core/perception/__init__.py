"""Parsers that convert worker results into perception records."""

from src.core.perception.generic_parser import GenericParser
from src.core.perception.parser_protocol import ParsedWorkerResult, ResultParser
from src.core.perception.parser_registry import ParserRegistry

__all__ = [
    "GenericParser",
    "ParsedWorkerResult",
    "ParserRegistry",
    "ResultParser",
]
