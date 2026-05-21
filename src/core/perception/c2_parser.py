"""Deprecated compatibility import.

Use src.integrations.incalmo.perception.IncalmoC2Parser instead.
"""

from src.integrations.incalmo.perception import (
    IncalmoC2CommandParser,
    IncalmoC2Parser,
    parse_incalmo_command_output,
)


__all__ = ["IncalmoC2Parser", "IncalmoC2CommandParser", "parse_incalmo_command_output"]
