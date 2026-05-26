"""Tunnel endpoint execution adapter."""

from __future__ import annotations

from src.core.execution.pivot_context import PivotExecutionContextResolver
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult
from src.core.models.runtime import RuntimeState


class TunnelAdapter:
    """Validate and expose a pre-opened tunnel endpoint to downstream tools."""

    name = "tcp_tunnel"

    def __init__(self, runtime_state: RuntimeState | None = None) -> None:
        self._runtime_state = runtime_state
        self._resolver = PivotExecutionContextResolver()

    def supports(self, plan: ToolPlan) -> bool:
        return plan.adapter in {self.name, "tunnel"}

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        context = self._resolver.resolve(plan, self._runtime_state)
        endpoint = context.tunnel_endpoint or plan.target
        if endpoint is None:
            return ToolExecutionResult(
                adapter=self.name,
                tool=plan.tool,
                success=False,
                exit_code="missing_tunnel",
                stderr="tunnel execution requires tunnel_endpoint or target",
                metadata={"tool_plan": plan.model_dump(mode="json"), "execution_context": context.model_dump(mode="json")},
            )
        return ToolExecutionResult(
            adapter=self.name,
            tool=plan.tool,
            success=True,
            exit_code=0,
            stdout=str(endpoint),
            metadata={
                "tool_plan": plan.model_dump(mode="json"),
                "execution_context": context.model_dump(mode="json"),
                "tunnel_endpoint": endpoint,
            },
        )


__all__ = ["TunnelAdapter"]
