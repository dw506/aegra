"""Execution adapter protocol."""

from __future__ import annotations

from typing import Protocol

from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult


class ExecutionAdapter(Protocol):
    """Adapter contract for ToolPlan execution backends."""

    name: str

    def supports(self, plan: ToolPlan) -> bool:
        """Return True when this adapter can execute the plan."""

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        """Execute the plan and return a canonical tool result."""


__all__ = ["ExecutionAdapter"]
