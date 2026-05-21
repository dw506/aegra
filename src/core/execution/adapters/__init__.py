"""Execution adapters."""

from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.adapters.incalmo_c2_adapter import IncalmoC2Adapter
from src.core.execution.adapters.local_shell_adapter import LocalShellAdapter

__all__ = ["ExecutionAdapter", "IncalmoC2Adapter", "LocalShellAdapter"]
