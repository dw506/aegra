"""Adapter dispatch for ToolPlan execution."""

from __future__ import annotations

from typing import Protocol

from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_policy import ToolPolicy
from src.core.models.events import AgentResultStatus, AgentRole, AgentTaskResult
from src.core.models.runtime import RuntimeState
from src.core.models.tg import BaseTaskNode


class ExecutionAdapter(Protocol):
    """Adapter contract for external execution backends."""

    def execute(self, plan: ToolPlan, runtime_state: RuntimeState) -> AgentTaskResult:
        """Execute a plan and return a canonical worker result."""


class ToolExecutor:
    """Policy-aware ToolPlan dispatcher."""

    def __init__(
        self,
        *,
        adapters: dict[str, ExecutionAdapter] | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._adapters = dict(adapters or {})
        self._policy = policy or ToolPolicy()

    def register_adapter(self, name: str, adapter: ExecutionAdapter) -> None:
        self._adapters[name] = adapter

    def execute(
        self,
        plan: ToolPlan,
        runtime_state: RuntimeState,
        *,
        task: BaseTaskNode | None = None,
    ) -> AgentTaskResult:
        decision = self._policy.evaluate(plan, runtime_state, task=task)
        if not decision.allowed:
            return AgentTaskResult(
                request_id=f"tool-plan::{plan.task_id}",
                agent_role=AgentRole.RECON_WORKER,
                operation_id=runtime_state.operation_id,
                task_id=plan.task_id,
                tg_node_id=str(plan.metadata.get("tg_node_id") or plan.task_id),
                status=AgentResultStatus.BLOCKED,
                summary=decision.reason,
                metadata={"tool_policy": decision.model_dump(mode="json"), "tool_plan": plan.model_dump(mode="json")},
            )
        adapter = self._adapters.get(plan.adapter)
        if adapter is None:
            return AgentTaskResult(
                request_id=f"tool-plan::{plan.task_id}",
                agent_role=AgentRole.RECON_WORKER,
                operation_id=runtime_state.operation_id,
                task_id=plan.task_id,
                tg_node_id=str(plan.metadata.get("tg_node_id") or plan.task_id),
                status=AgentResultStatus.FAILED,
                summary=f"execution adapter '{plan.adapter}' is not registered",
                error_message=f"missing adapter: {plan.adapter}",
                metadata={"tool_plan": plan.model_dump(mode="json")},
            )
        return adapter.execute(plan, runtime_state)
