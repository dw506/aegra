"""Execution adapters."""

from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.adapters.local_shell_adapter import LocalShellAdapter

__all__ = ["ExecutionAdapter", "LocalShellAdapter"]
