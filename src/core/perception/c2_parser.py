"""Compatibility import for legacy C2 perception helpers.

New code should import Incalmo-specific parsing from
`src.integrations.incalmo.perception`.
"""

from src.integrations.incalmo.perception import IncalmoC2CommandParser, parse_incalmo_command_output


__all__ = ["IncalmoC2CommandParser", "parse_incalmo_command_output"]
