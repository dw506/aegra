"""Deterministic action templates for Task Graph generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.ag import GraphRef
from src.core.models.tg import TaskNode, TaskStatus, TaskType


class ActionTemplate(BaseModel):
    """Execution metadata attached to a supported AG action type."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(min_length=1)
    task_type: TaskType
    label: str = Field(min_length=1)
    estimated_risk: float = Field(default=0.1, ge=0.0, le=1.0)
    estimated_noise: float = Field(default=0.1, ge=0.0, le=1.0)
    approval_required: bool = False
    resource_key_fields: tuple[str, ...] = Field(default_factory=tuple)
    default_bindings: dict[str, Any] = Field(default_factory=dict)


class ActionTemplateInput(BaseModel):
    """Input required to instantiate one action template."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    bindings: dict[str, Any] = Field(default_factory=dict)
    target_refs: list[GraphRef] = Field(default_factory=list)
    priority: int = Field(default=50, ge=0, le=100)
    assigned_agent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionTemplateOutput(BaseModel):
    """Result of rendering an action template."""

    model_config = ConfigDict(extra="forbid")

    task: TaskNode
    template: ActionTemplate


TEMPLATES: dict[str, ActionTemplate] = {
    "enumerate_host": ActionTemplate(
        action_type="enumerate_host",
        task_type=TaskType.ASSET_CONFIRMATION,
        label="Enumerate host",
        resource_key_fields=("host_id", "target_address"),
        default_bindings={"tool_hint": "nmap"},
    ),
    "validate_service": ActionTemplate(
        action_type="validate_service",
        task_type=TaskType.SERVICE_VALIDATION,
        label="Validate service",
        resource_key_fields=("service_id", "host_id", "target_address"),
        default_bindings={"tool_hint": "nmap"},
    ),
    "validate_reachability": ActionTemplate(
        action_type="validate_reachability",
        task_type=TaskType.REACHABILITY_VALIDATION,
        label="Validate reachability",
        resource_key_fields=("host_id", "target_address", "pivot_route_id"),
        default_bindings={"tool_hint": "ping"},
    ),
    "establish_pivot": ActionTemplate(
        action_type="establish_pivot",
        task_type=TaskType.PRIVILEGE_CONFIGURATION_VALIDATION,
        label="Establish pivot",
        estimated_risk=0.7,
        estimated_noise=0.4,
        approval_required=True,
        resource_key_fields=("host_id", "session_id", "pivot_route_id"),
        default_bindings={"tool_hint": "incalmo"},
    ),
    "reuse_credential": ActionTemplate(
        action_type="reuse_credential",
        task_type=TaskType.IDENTITY_CONTEXT_CONFIRMATION,
        label="Reuse credential",
        estimated_risk=0.6,
        estimated_noise=0.3,
        approval_required=True,
        resource_key_fields=("credential_id", "host_id", "service_id"),
        default_bindings={"tool_hint": "incalmo"},
    ),
    "validate_goal": ActionTemplate(
        action_type="validate_goal",
        task_type=TaskType.GOAL_CONDITION_VALIDATION,
        label="Validate goal",
        estimated_risk=0.2,
        estimated_noise=0.1,
        resource_key_fields=("goal_id", "host_id"),
        default_bindings={"tool_hint": "incalmo"},
    ),
}


def build_task_from_action(action: ActionTemplateInput) -> ActionTemplateOutput:
    """Render a supported action into a deterministic TG task node."""

    if action.action_type not in TEMPLATES:
        raise ValueError(f"unsupported action template '{action.action_type}'")
    template = TEMPLATES[action.action_type]
    bindings = {**template.default_bindings, **dict(action.bindings)}
    task_id = str(action.metadata.get("task_id") or f"task::{action.action_id}")
    resource_keys = {
        f"{field}:{bindings[field]}"
        for field in template.resource_key_fields
        if bindings.get(field) is not None and str(bindings[field]).strip()
    }
    task = TaskNode(
        id=task_id,
        label=str(action.metadata.get("label") or template.label),
        task_type=template.task_type,
        status=TaskStatus.READY,
        source_action_id=action.action_id,
        input_bindings=bindings,
        target_refs=list(action.target_refs),
        estimated_risk=template.estimated_risk,
        estimated_noise=template.estimated_noise,
        priority=action.priority,
        resource_keys=resource_keys,
        approval_required=template.approval_required,
        assigned_agent=action.assigned_agent,
        reason=str(action.metadata.get("reason") or f"generated from {action.action_type}"),
        tags={"action-template", action.action_type},
    )
    return ActionTemplateOutput(task=task, template=template)
