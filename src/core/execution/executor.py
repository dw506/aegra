"""Adapter dispatch for ToolPlan execution."""

from __future__ import annotations

from typing import Protocol

from src.core.execution.adapter_resolver import ToolAdapterResolver
from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_policy import ToolPolicy
from src.core.execution.tool_result import ToolExecutionResult
from src.core.models.events import AgentResultStatus, AgentRole, AgentTaskResult
from src.core.models.runtime import RuntimeState
from src.core.models.tg import BaseTaskNode


class LegacyToolAdapter(Protocol):
    """Compatibility adapter contract used by ToolExecutor."""

    def execute(self, plan: ToolPlan, runtime_state: RuntimeState) -> AgentTaskResult:
        """Execute a plan and return a legacy task result."""


class ExecutionExecutor:
    """Dispatch ToolPlans to adapter-neutral execution backends."""

    def __init__(
        self,
        adapters: list[ExecutionAdapter] | None = None,
        *,
        resolver: ToolAdapterResolver | None = None,
    ) -> None:
        self._adapters = list(adapters or [])
        self._resolver = resolver or ToolAdapterResolver()

    def register_adapter(self, adapter: ExecutionAdapter) -> ExecutionAdapter:
        self._adapters.append(adapter)
        return adapter

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        adapter_map = {adapter.name: adapter for adapter in self._adapters}
        resolution = self._resolver.resolve(plan, adapter_map)
        if not resolution.allowed:
            raise ValueError(f"No adapter supports plan adapter={plan.adapter}, tool={plan.tool}: {resolution.reason}")
        resolved_plan = self._resolver.plan_for_resolution(plan, resolution)
        for adapter in self._adapters:
            if adapter.supports(resolved_plan):
                result = adapter.execute(resolved_plan)
                result.metadata.setdefault("adapter_resolution", resolution.model_dump(mode="json"))
                return result
        raise ValueError(f"No adapter supports plan adapter={resolved_plan.adapter}, tool={resolved_plan.tool}")


class ToolExecutor:
    """Policy-aware ToolPlan dispatcher."""

    def __init__(
        self,
        *,
        adapters: dict[str, LegacyToolAdapter] | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._adapters = dict(adapters or {})
        self._policy = policy or ToolPolicy()

    def register_adapter(self, name: str, adapter: LegacyToolAdapter) -> None:
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
