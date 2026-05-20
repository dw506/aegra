"""Build executable tool plans from Task Graph nodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.tg import BaseTaskNode


class ToolPlan(BaseModel):
    """Minimal adapter-facing plan for one TG task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    adapter: str = Field(default="local", min_length=1)
    command: str = Field(min_length=1)
    payloads: dict[str, Any] = Field(default_factory=dict)
    target_agent_ref: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_tool_plan(task: BaseTaskNode, *, default_adapter: str = "local") -> ToolPlan:
    """Convert one TG task into a ToolPlan without executing it."""

    bindings = dict(task.input_bindings)
    adapter = str(bindings.get("adapter") or bindings.get("execution_adapter") or default_adapter)
    command = str(
        bindings.get("command")
        or bindings.get("command_name")
        or bindings.get("tool_hint")
        or bindings.get("tool")
        or task.task_type.value.lower()
    )
    target_agent_ref = bindings.get("target_agent_ref") or bindings.get("agent_id") or bindings.get("session_id")
    timeout = bindings.get("timeout_seconds") or bindings.get("timeout_sec")
    return ToolPlan(
        task_id=task.id,
        adapter=adapter,
        command=command,
        payloads=bindings,
        target_agent_ref=str(target_agent_ref) if target_agent_ref is not None else None,
        timeout_seconds=float(timeout) if timeout is not None else None,
        metadata={
            "tg_node_id": task.id,
            "task_type": task.task_type.value,
            "source_action_id": task.source_action_id,
            "resource_keys": sorted(task.resource_keys),
            "approval_required": task.approval_required,
        },
    )
