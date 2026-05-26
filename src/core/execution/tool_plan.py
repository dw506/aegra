"""Build executable tool plans from Task Graph nodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.agents.agent_protocol import GraphRef
from src.core.models.tg import BaseTaskNode


class ToolPlan(BaseModel):
    """Minimal adapter-facing plan for one TG task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    tool: str = ""
    adapter: str | None = None
    command: str | None = None
    target: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=60, gt=0)
    scope_refs: list[GraphRef] = Field(default_factory=list)
    payloads: dict[str, Any] = Field(default_factory=dict)
    target_agent_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_tool_name(self) -> "ToolPlan":
        if not self.tool:
            self.tool = str(self.command or self.metadata.get("task_type") or "unknown_tool")
        return self


def build_tool_plan(task: BaseTaskNode, *, default_adapter: str | None = None) -> ToolPlan:
    """Convert one TG task into a ToolPlan without executing it."""

    bindings = dict(task.input_bindings)
    selected_route = bindings.get("selected_route") if isinstance(bindings.get("selected_route"), dict) else {}
    route_metadata = selected_route.get("metadata") if isinstance(selected_route.get("metadata"), dict) else {}
    route_transport = route_metadata.get("transport") if isinstance(route_metadata.get("transport"), dict) else {}
    adapter_value = bindings.get("adapter") or bindings.get("execution_adapter") or route_transport.get("adapter") or default_adapter
    adapter = str(adapter_value) if adapter_value is not None else None
    command = str(
        bindings.get("command")
        or bindings.get("command_name")
        or bindings.get("tool_hint")
        or bindings.get("tool")
        or task.task_type.value.lower()
    )
    target = bindings.get("target") or bindings.get("target_address") or bindings.get("host")
    target_agent_ref = bindings.get("target_agent_ref") or bindings.get("agent_id") or bindings.get("session_id")
    timeout = bindings.get("timeout_seconds") or bindings.get("timeout_sec")
    route_id = bindings.get("route_id") or bindings.get("selected_route_id")
    session_id = bindings.get("session_id")
    return ToolPlan(
        task_id=task.id,
        tool=command,
        adapter=adapter,
        command=command,
        target=str(target) if target is not None else None,
        args=bindings,
        payloads=bindings,
        target_agent_ref=str(target_agent_ref) if target_agent_ref is not None else None,
        timeout_seconds=int(timeout) if timeout is not None else 60,
        metadata={
            "tg_node_id": task.id,
            "task_type": task.task_type.value,
            "source_action_id": task.source_action_id,
            "resource_keys": sorted(task.resource_keys),
            "approval_required": task.approval_required,
            "route_id": str(route_id) if route_id is not None else None,
            "selected_route_id": str(route_id) if route_id is not None else None,
            "selected_route": selected_route,
            "session_id": str(session_id) if session_id is not None else None,
            "proxy_url": bindings.get("proxy_url"),
            "tunnel_endpoint": bindings.get("tunnel_endpoint"),
            "network_namespace": bindings.get("network_namespace"),
            "tool_hint": bindings.get("tool_hint") or bindings.get("tool"),
            "mcp_server_id": bindings.get("mcp_server_id"),
            "mcp_tool_name": bindings.get("mcp_tool_name"),
        },
    )
