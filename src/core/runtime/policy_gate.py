"""Lightweight scheduler-facing policy gate.

The gate is intentionally narrower than the full runtime policy engine. It is
used immediately before scheduling so duplicate or inadmissible TG tasks never
reach worker assignment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.runtime import RuntimeState
from src.core.models.tg import BaseTaskNode
from src.core.runtime.budgets import RuntimeBudgetManager
from src.core.runtime.observability import append_audit_log
from src.core.runtime.policy import PolicyDecision, RuntimePolicy, policy_from_runtime_state
from src.core.runtime.policy_engine import PolicyEngine


class PolicyGateAction(str, Enum):
    """Scheduler-facing policy gate outcome."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    NEED_APPROVAL = "NEED_APPROVAL"


class PolicyGateDecision(BaseModel):
    """Decision emitted by the lightweight policy gate."""

    model_config = ConfigDict(extra="forbid")

    action: PolicyGateAction
    reason: str = Field(min_length=1)
    gate: str = Field(min_length=1)
    task_id: str | None = None
    target: str | None = None
    approval_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def allow(self) -> bool:
        return self.action == PolicyGateAction.ALLOW

    @property
    def deny(self) -> bool:
        return self.action == PolicyGateAction.DENY

    @property
    def need_approval(self) -> bool:
        return self.action == PolicyGateAction.NEED_APPROVAL


class PolicyGate:
    """Perform five pre-scheduler checks: scope, risk, tool, approval, budget."""

    def __init__(self, policy: RuntimePolicy | None = None) -> None:
        self._policy = policy
        self._budgets = RuntimeBudgetManager()

    def evaluate(
        self,
        task: BaseTaskNode,
        *,
        runtime_state: RuntimeState | None = None,
        budget_summary: dict[str, Any] | None = None,
    ) -> PolicyGateDecision:
        """Evaluate one TG task before worker scheduling."""

        policy = self._resolve_policy(runtime_state)
        engine = PolicyEngine(policy)

        scope_decision = engine.evaluate_task_policy(task, runtime_state)
        if scope_decision.gate == "scope" and scope_decision.decision != "allow":
            return self._from_policy_decision(scope_decision, task_id=task.id)

        risk_decision = self._risk_check(task, policy=policy, runtime_state=runtime_state)
        if risk_decision.action != PolicyGateAction.ALLOW:
            return risk_decision

        tool_decision = self._tool_check(task, policy=policy)
        if tool_decision.action != PolicyGateAction.ALLOW:
            return tool_decision

        approval_decision = self._approval_check(task, runtime_state=runtime_state)
        if approval_decision.action != PolicyGateAction.ALLOW:
            return approval_decision

        budget_decision = self._budget_check(
            task,
            runtime_state=runtime_state,
            budget_summary=budget_summary or {},
        )
        if budget_decision.action != PolicyGateAction.ALLOW:
            return budget_decision

        return PolicyGateDecision(action=PolicyGateAction.ALLOW, gate="policy_gate", task_id=task.id, reason="policy gate allowed")

    def audit(self, runtime_state: RuntimeState, decision: PolicyGateDecision) -> None:
        """Append one audit entry for a gate decision."""

        append_audit_log(
            runtime_state,
            {
                "event_type": "policy_gate_decision",
                "decision": decision.model_dump(mode="json"),
            },
        )

    def _resolve_policy(self, runtime_state: RuntimeState | None) -> RuntimePolicy:
        if self._policy is not None:
            return self._policy
        if runtime_state is not None:
            return policy_from_runtime_state(runtime_state)
        return RuntimePolicy()

    def _risk_check(
        self,
        task: BaseTaskNode,
        *,
        policy: RuntimePolicy,
        runtime_state: RuntimeState | None,
    ) -> PolicyGateDecision:
        max_numeric = self._numeric_policy_value(
            runtime_state,
            keys=("max_task_risk", "risk_threshold", "max_risk"),
        )
        if max_numeric is not None and task.estimated_risk > max_numeric:
            return PolicyGateDecision(
                action=PolicyGateAction.DENY,
                gate="risk",
                task_id=task.id,
                reason="task risk exceeds numeric threshold",
                metadata={"estimated_risk": task.estimated_risk, "threshold": max_numeric},
            )

        risk_decision = PolicyEngine(policy).evaluate_task_policy(task, runtime_state)
        if risk_decision.gate == "risk" and risk_decision.decision != "allow":
            return self._from_policy_decision(risk_decision, task_id=task.id)
        return PolicyGateDecision(action=PolicyGateAction.ALLOW, gate="risk", task_id=task.id, reason="risk allowed")

    def _tool_check(self, task: BaseTaskNode, *, policy: RuntimePolicy) -> PolicyGateDecision:
        tool_name = self._tool_name(task)
        if not tool_name:
            return PolicyGateDecision(action=PolicyGateAction.ALLOW, gate="tool", task_id=task.id, reason="no tool hint provided")

        disabled = {item.lower() for item in policy.disabled_tools}
        if tool_name.lower() in disabled:
            return PolicyGateDecision(action=PolicyGateAction.DENY, gate="tool", task_id=task.id, reason="tool disabled by policy")

        allowlist = {item.lower() for item in policy.command_allowlist}
        if allowlist and tool_name.lower() not in allowlist:
            return PolicyGateDecision(
                action=PolicyGateAction.DENY,
                gate="tool",
                task_id=task.id,
                reason="tool outside allowlist",
                metadata={"tool": tool_name, "allowlist": sorted(allowlist)},
            )
        return PolicyGateDecision(action=PolicyGateAction.ALLOW, gate="tool", task_id=task.id, reason="tool allowed")

    def _approval_check(
        self,
        task: BaseTaskNode,
        *,
        runtime_state: RuntimeState | None,
    ) -> PolicyGateDecision:
        if not task.approval_required and not task.gate_ids:
            return PolicyGateDecision(action=PolicyGateAction.ALLOW, gate="approval", task_id=task.id, reason="approval not required")
        approval_id = f"task:{task.id}:approved"
        if runtime_state is not None and runtime_state.budgets.approval_cache.get(approval_id):
            return PolicyGateDecision(action=PolicyGateAction.ALLOW, gate="approval", task_id=task.id, reason="approval already granted")
        return PolicyGateDecision(
            action=PolicyGateAction.NEED_APPROVAL,
            gate="approval",
            task_id=task.id,
            approval_id=approval_id,
            reason="task requires approval",
        )

    def _budget_check(
        self,
        task: BaseTaskNode,
        *,
        runtime_state: RuntimeState | None,
        budget_summary: dict[str, Any],
    ) -> PolicyGateDecision:
        if runtime_state is not None:
            exhausted = self._budgets.would_exceed_budget(
                runtime_state,
                operations=1,
                noise=task.estimated_noise,
                risk=task.estimated_risk,
            )
        else:
            exhausted = self._summary_budget_exhausted(task=task, remaining=budget_summary)
        if exhausted:
            return PolicyGateDecision(action=PolicyGateAction.DENY, gate="budget", task_id=task.id, reason="budget exhausted")
        return PolicyGateDecision(action=PolicyGateAction.ALLOW, gate="budget", task_id=task.id, reason="budget available")

    @staticmethod
    def _from_policy_decision(decision: PolicyDecision, *, task_id: str) -> PolicyGateDecision:
        action = PolicyGateAction.NEED_APPROVAL if decision.decision == "requires_approval" else PolicyGateAction.DENY
        return PolicyGateDecision(
            action=action,
            gate=decision.gate,
            task_id=decision.task_id or task_id,
            target=decision.target,
            approval_id=decision.approval_id,
            reason=decision.reason,
            metadata={"policy_decision": decision.model_dump(mode="json")},
        )

    @staticmethod
    def _tool_name(task: BaseTaskNode) -> str | None:
        for key in ("tool_hint", "tool", "tool_name", "recipe_id", "command_name"):
            value = task.input_bindings.get(key)
            if value is not None and str(value).strip():
                return str(value).strip().rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        return None

    @staticmethod
    def _numeric_policy_value(runtime_state: RuntimeState | None, *, keys: tuple[str, ...]) -> float | None:
        if runtime_state is None:
            return None
        sources = [
            runtime_state.execution.metadata.get("policy_gate", {}),
            runtime_state.execution.metadata.get("runtime_policy", {}),
        ]
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                if key not in source:
                    continue
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _summary_budget_exhausted(*, task: BaseTaskNode, remaining: dict[str, Any]) -> bool:
        noise_remaining = remaining.get("noise_budget_remaining")
        risk_remaining = remaining.get("risk_budget_remaining")
        operation_remaining = remaining.get("operation_budget_remaining")
        if noise_remaining is not None and float(noise_remaining) < task.estimated_noise:
            return True
        if risk_remaining is not None and float(risk_remaining) < task.estimated_risk:
            return True
        if operation_remaining is not None and int(operation_remaining) < 1:
            return True
        return False


__all__ = [
    "PolicyGate",
    "PolicyGateAction",
    "PolicyGateDecision",
]
