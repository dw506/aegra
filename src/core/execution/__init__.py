"""Execution planning and adapter dispatch."""

from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.adapters.incalmo_c2_adapter import IncalmoC2Adapter
from src.core.execution.adapters.local_shell_adapter import LocalShellAdapter
from src.core.execution.executor import ExecutionExecutor, LegacyToolAdapter, ToolExecutor
from src.core.execution.tool_plan import ToolPlan, build_tool_plan
from src.core.execution.tool_policy import ToolPolicy, ToolPolicyDecision
from src.core.execution.tool_result import ToolExecutionResult

__all__ = [
    "ExecutionExecutor",
    "ExecutionAdapter",
    "IncalmoC2Adapter",
    "LocalShellAdapter",
    "LegacyToolAdapter",
    "ToolExecutor",
    "ToolExecutionResult",
    "ToolPlan",
    "ToolPolicy",
    "ToolPolicyDecision",
    "build_tool_plan",
]
