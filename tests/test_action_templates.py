from __future__ import annotations

from src.core.actions import ActionTemplateInput, build_task_from_action
from src.core.models.ag import GraphRef
from src.core.models.tg import TaskType


def test_action_template_maps_validate_service_deterministically() -> None:
    rendered = build_task_from_action(
        ActionTemplateInput(
            action_id="action-1",
            action_type="validate_service",
            bindings={"host_id": "host-1", "service_id": "svc-1"},
            target_refs=[GraphRef(graph="kg", ref_id="svc-1", ref_type="Service")],
        )
    )

    task = rendered.task
    assert task.id == "task::action-1"
    assert task.task_type == TaskType.SERVICE_VALIDATION
    assert task.input_bindings["tool_hint"] == "nmap"
    assert task.resource_keys == {"host_id:host-1", "service_id:svc-1"}
    assert task.approval_required is False


def test_action_template_marks_pivot_as_approval_required() -> None:
    rendered = build_task_from_action(
        ActionTemplateInput(
            action_id="pivot-1",
            action_type="establish_pivot",
            bindings={"host_id": "host-1", "session_id": "session-1"},
        )
    )

    assert rendered.task.task_type == TaskType.PRIVILEGE_CONFIGURATION_VALIDATION
    assert rendered.task.approval_required is True
    assert rendered.task.estimated_risk == 0.7
