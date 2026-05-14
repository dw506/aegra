"""Runtime approval request helpers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.runtime import RuntimeState, TaskRuntimeStatus, utc_now
from src.core.runtime.observability import append_audit_log


class ApprovalRequest(BaseModel):
    """One approval decision submitted by a user or control-plane client."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1)
    task_id: str | None = None
    decision: Literal["approve", "deny"] = "approve"
    reason: str = Field(default="manual approval", min_length=1)
    requested_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalManager:
    """Store and apply approval decisions in RuntimeState metadata/cache."""

    def list_approvals(self, state: RuntimeState) -> list[dict[str, Any]]:
        approvals = state.execution.metadata.setdefault("approvals", {})
        result = [dict(item) for item in approvals.values() if isinstance(item, dict)]
        for task in state.execution.tasks.values():
            decision = task.metadata.get("policy_decision")
            if task.status == TaskRuntimeStatus.WAITING_APPROVAL and isinstance(decision, dict):
                approval_id = str(decision.get("approval_id") or f"task:{task.task_id}:approved")
                if not any(item.get("approval_id") == approval_id for item in result):
                    result.append(
                        {
                            "approval_id": approval_id,
                            "task_id": task.task_id,
                            "status": "pending",
                            "reason": decision.get("reason", "approval required"),
                            "policy_decision": decision,
                        }
                    )
        return sorted(result, key=lambda item: str(item.get("approval_id")))

    def apply(self, state: RuntimeState, request: ApprovalRequest) -> dict[str, Any]:
        approved = request.decision == "approve"
        state.budgets.approval_cache[request.approval_id] = approved
        if request.task_id and request.task_id in state.execution.tasks:
            task = state.execution.tasks[request.task_id]
            if task.status == TaskRuntimeStatus.WAITING_APPROVAL and approved:
                task.status = TaskRuntimeStatus.PENDING
                task.metadata["approval_status"] = "approved"
            elif task.status == TaskRuntimeStatus.WAITING_APPROVAL:
                task.status = TaskRuntimeStatus.BLOCKED
                task.metadata["approval_status"] = "denied"
        record = {
            "approval_id": request.approval_id,
            "task_id": request.task_id,
            "status": "approved" if approved else "denied",
            "reason": request.reason,
            "requested_by": request.requested_by,
            "metadata": dict(request.metadata),
            "decided_at": utc_now().isoformat(),
        }
        state.execution.metadata.setdefault("approvals", {})[request.approval_id] = record
        append_audit_log(state, {"event_type": "approval_decision", **record})
        state.last_updated = utc_now()
        return record


__all__ = ["ApprovalManager", "ApprovalRequest"]
