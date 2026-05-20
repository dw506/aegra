"""Parsers that convert worker results into perception records."""

from src.core.perception.generic_parser import GenericParser
from src.core.perception.parser_protocol import ParsedWorkerResult, ResultParser
from src.core.perception.parser_registry import ParserRegistry


def parse_incalmo_command_output(*args, **kwargs):
    """Compatibility wrapper for the Incalmo-specific parser helper."""

    from src.integrations.incalmo.perception import parse_incalmo_command_output as _parse

    return _parse(*args, **kwargs)

__all__ = [
    "GenericParser",
    "ParsedWorkerResult",
    "ParserRegistry",
    "ResultParser",
    "parse_incalmo_command_output",
]
