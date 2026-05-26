"""Resolve ToolPlans to execution adapters with MCP-first defaults."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.execution.adapters.base import ExecutionAdapter
from src.core.execution.tool_plan import ToolPlan


class ToolBinding(BaseModel):
    """Configured mapping from semantic tool intent to adapter details."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    default_adapter: str | None = None
    fallback_adapters: list[str] = Field(default_factory=list)
    mcp: dict[str, str] = Field(default_factory=dict)
    allowed_task_types: list[str] = Field(default_factory=list)


class AdapterPolicyConfig(BaseModel):
    """MCP-first adapter selection policy."""

    model_config = ConfigDict(extra="forbid")

    default_preference: list[str] = Field(default_factory=lambda: ["mcp", "local_shell"])
    mcp_first: bool = True
    allow_local_fallback: bool = True
    force_mcp_for_task_types: list[str] = Field(default_factory=list)
    allow_local_fallback_for_task_types: list[str] = Field(default_factory=list)
    deny_local_fallback_for_task_types: list[str] = Field(default_factory=list)
    deny_adapters: list[str] = Field(default_factory=list)
    allow_mcp_servers: list[str] = Field(default_factory=list)
    deny_mcp_servers: list[str] = Field(default_factory=list)
    allow_mcp_tools: list[str] = Field(default_factory=list)
    deny_mcp_tools: list[str] = Field(default_factory=list)


class AdapterResolution(BaseModel):
    """Resolved adapter decision for one ToolPlan."""

    model_config = ConfigDict(extra="forbid")

    adapter: str | None
    allowed: bool
    reason: str
    fallback_allowed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolAdapterResolver:
    """Choose the concrete adapter for a ToolPlan without executing it."""

    def __init__(
        self,
        *,
        policy: AdapterPolicyConfig | dict[str, Any] | None = None,
        bindings: list[ToolBinding | dict[str, Any]] | None = None,
    ) -> None:
        self._policy = self._coerce_policy(policy)
        self._bindings = {
            binding.tool: binding
            for binding in [self._coerce_binding(item) for item in (bindings or [])]
        }

    @property
    def policy(self) -> AdapterPolicyConfig:
        return self._policy

    def resolve(
        self,
        plan: ToolPlan,
        adapters: dict[str, ExecutionAdapter],
    ) -> AdapterResolution:
        task_type = self._task_type(plan)
        binding = self._bindings.get(plan.tool)
        enriched = self._apply_binding(plan, binding)
        policy = self._policy

        denied = self._deny_reason(enriched, policy)
        if denied:
            return AdapterResolution(adapter=None, allowed=False, reason=denied)

        if plan.adapter:
            if self._adapter_available(plan.adapter, enriched, adapters):
                return AdapterResolution(adapter=plan.adapter, allowed=True, reason="explicit adapter selected")
            return AdapterResolution(adapter=None, allowed=False, reason=f"explicit adapter '{plan.adapter}' is not available")

        if task_type in set(policy.force_mcp_for_task_types):
            return self._select_mcp(enriched, adapters, reason="task type requires MCP")

        candidates: list[str] = []
        if binding and binding.default_adapter:
            candidates.append(binding.default_adapter)
        if policy.mcp_first:
            candidates.extend(policy.default_preference)
        else:
            candidates.extend([item for item in policy.default_preference if item != "mcp"])
            candidates.append("mcp")
        if binding:
            candidates.extend(binding.fallback_adapters)

        seen: set[str] = set()
        for adapter_name in candidates:
            if not adapter_name or adapter_name in seen:
                continue
            seen.add(adapter_name)
            if adapter_name != "mcp" and not self._fallback_allowed(adapter_name, task_type, policy):
                continue
            candidate = enriched.model_copy(update={"adapter": adapter_name}, deep=True)
            denied = self._deny_reason(candidate, policy)
            if denied:
                continue
            if self._adapter_available(adapter_name, candidate, adapters):
                return AdapterResolution(
                    adapter=adapter_name,
                    allowed=True,
                    reason="adapter selected by resolver",
                    fallback_allowed=adapter_name != "mcp",
                    metadata={"task_type": task_type, "binding": binding.model_dump(mode="json") if binding else None},
                )

        return AdapterResolution(adapter=None, allowed=False, reason=f"no allowed adapter is available for tool '{plan.tool}'")

    def plan_for_resolution(self, plan: ToolPlan, resolution: AdapterResolution) -> ToolPlan:
        if resolution.adapter is None:
            return plan
        binding = self._bindings.get(plan.tool)
        return self._apply_binding(plan, binding).model_copy(update={"adapter": resolution.adapter}, deep=True)

    def _select_mcp(
        self,
        plan: ToolPlan,
        adapters: dict[str, ExecutionAdapter],
        *,
        reason: str,
    ) -> AdapterResolution:
        candidate = plan.model_copy(update={"adapter": "mcp"}, deep=True)
        denied = self._deny_reason(candidate, self._policy)
        if denied:
            return AdapterResolution(adapter=None, allowed=False, reason=denied)
        if self._adapter_available("mcp", candidate, adapters):
            return AdapterResolution(adapter="mcp", allowed=True, reason=reason)
        return AdapterResolution(adapter=None, allowed=False, reason=f"{reason}, but MCP is not available")

    def _apply_binding(self, plan: ToolPlan, binding: ToolBinding | None) -> ToolPlan:
        if binding is None:
            return plan
        metadata = dict(plan.metadata)
        if binding.mcp.get("server_id"):
            metadata.setdefault("mcp_server_id", binding.mcp["server_id"])
        if binding.mcp.get("tool_name"):
            metadata.setdefault("mcp_tool_name", binding.mcp["tool_name"])
        adapter = plan.adapter or binding.default_adapter
        if binding.allowed_task_types:
            task_type = self._task_type(plan)
            if task_type and task_type not in set(binding.allowed_task_types):
                metadata["binding_task_type_denied"] = True
        return plan.model_copy(update={"adapter": adapter, "metadata": metadata}, deep=True)

    def _deny_reason(self, plan: ToolPlan, policy: AdapterPolicyConfig) -> str | None:
        if plan.adapter in set(policy.deny_adapters):
            return f"adapter '{plan.adapter}' is denied by policy"
        task_type = self._task_type(plan)
        if plan.adapter and plan.adapter != "mcp" and task_type in set(policy.deny_local_fallback_for_task_types):
            return f"task type '{task_type}' denies non-MCP adapters"
        if plan.metadata.get("binding_task_type_denied"):
            return f"tool '{plan.tool}' is not allowed for task type '{self._task_type(plan)}'"
        server_id = self._string(plan.metadata.get("mcp_server_id") or plan.args.get("mcp_server_id"))
        tool_name = self._string(plan.metadata.get("mcp_tool_name") or plan.args.get("mcp_tool_name") or plan.tool)
        if plan.adapter == "mcp":
            if server_id and server_id in set(policy.deny_mcp_servers):
                return f"MCP server '{server_id}' is denied by policy"
            if tool_name and tool_name in set(policy.deny_mcp_tools):
                return f"MCP tool '{tool_name}' is denied by policy"
            if policy.allow_mcp_servers and server_id not in set(policy.allow_mcp_servers):
                return f"MCP server '{server_id}' is not allowlisted"
            if policy.allow_mcp_tools and tool_name not in set(policy.allow_mcp_tools):
                return f"MCP tool '{tool_name}' is not allowlisted"
        return None

    def _fallback_allowed(self, adapter_name: str, task_type: str | None, policy: AdapterPolicyConfig) -> bool:
        if adapter_name == "mcp":
            return True
        if not policy.allow_local_fallback:
            return False
        if task_type in set(policy.deny_local_fallback_for_task_types):
            return False
        if policy.allow_local_fallback_for_task_types:
            return task_type in set(policy.allow_local_fallback_for_task_types)
        return True

    @staticmethod
    def _adapter_available(adapter_name: str, plan: ToolPlan, adapters: dict[str, ExecutionAdapter]) -> bool:
        adapter = adapters.get(adapter_name)
        if adapter and adapter.supports(plan):
            return True
        return any(candidate.supports(plan) for candidate in adapters.values())

    @staticmethod
    def _task_type(plan: ToolPlan) -> str | None:
        value = plan.metadata.get("task_type") or plan.args.get("task_type")
        return str(value) if value is not None else None

    @staticmethod
    def _coerce_policy(policy: AdapterPolicyConfig | dict[str, Any] | None) -> AdapterPolicyConfig:
        if isinstance(policy, AdapterPolicyConfig):
            return policy
        return AdapterPolicyConfig.model_validate(policy or {})

    @staticmethod
    def _coerce_binding(binding: ToolBinding | dict[str, Any]) -> ToolBinding:
        if isinstance(binding, ToolBinding):
            return binding
        return ToolBinding.model_validate(binding)

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["AdapterPolicyConfig", "AdapterResolution", "ToolAdapterResolver", "ToolBinding"]
