"""Goal validation worker for goal-state confirmation tasks."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput, GraphRef, GraphScope
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec


class GoalValidationWorker(BaseWorkerAgent):
    """Concrete worker for goal reached and target-state confirmation.

    This worker is read-only with respect to KG, AG, and TG. It validates goal
    conditions only and emits:
    - one `OutcomeRecord`
    - one raw result pointer payload

    Future real integrations should replace the placeholder adapters while
    preserving the same output contract.
    """

    capabilities = frozenset(
        {
            WorkerCapability.GOAL_VALIDATION,
            WorkerCapability.CONTEXT_VALIDATION,
        }
    )
    supported_task_types = frozenset(
        {
            "goal_reached_verification",
            "target_state_confirmation",
        }
    )

    def __init__(self, name: str = "goal_validation_worker") -> None:
        super().__init__(name=name)

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        """Return True when the task is one of the supported goal operations."""

        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        """Execute one goal-validation task and return outcome plus raw result."""

        if task_spec.task_type == "goal_reached_verification":
            raw_result = self._execute_goal_reached_verification(task_spec, agent_input)
        elif task_spec.task_type == "target_state_confirmation":
            raw_result = self._execute_target_state_confirmation(task_spec, agent_input)
        else:
            raise ValueError(f"unsupported goal validation task type: {task_spec.task_type}")

        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=bool(raw_result.get("success", True)),
            summary=str(raw_result["summary"]),
            raw_result_ref=str(raw_result["payload_ref"]),
            confidence=float(raw_result.get("confidence", 0.8)),
            refs=task_spec.target_refs,
            payload={
                "task_type": task_spec.task_type,
                "executor": raw_result.get("executor", "placeholder"),
                "result_type": raw_result.get("result_type", task_spec.task_type),
                "validation_status": raw_result.get("validation_status", "unknown"),
                "goal_satisfied": bool(raw_result.get("goal_satisfied", False)),
            },
        )
        return AgentOutput(
            outcomes=[outcome.to_agent_output_fragment()],
            evidence=[raw_result],
            logs=[
                f"worker={self.name}",
                f"task_id={task_spec.task_id}",
                f"task_type={task_spec.task_type}",
                f"target_count={len(task_spec.target_refs)}",
                f"resource_keys={len(task_spec.resource_keys)}",
                f"raw_result_ref={raw_result['payload_ref']}",
                str(raw_result["summary"]),
            ],
        )

    def _execute_goal_reached_verification(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Execute goal reached verification via the current adapter."""

        return self._goal_reached_verification_adapter(task_spec, agent_input)

    def _execute_target_state_confirmation(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Execute target-state confirmation via the current adapter."""

        return self._target_state_confirmation_adapter(task_spec, agent_input)

    def _goal_reached_verification_adapter(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Placeholder adapter for future goal reached verification tooling."""

        goal_ref = (
            task_spec.input_bindings.get("goal")
            or task_spec.input_bindings.get("goal_id")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Goal")
            or "unknown-goal"
        )
        target_state = (
            task_spec.input_bindings.get("target_state")
            or task_spec.constraints.get("expected_state")
            or "reached"
        )
        goal_satisfied = bool(task_spec.constraints.get("goal_satisfied", True))
        verification = {
            "goal_ref": goal_ref,
            "target_state": target_state,
            "goal_satisfied": goal_satisfied,
            "source": "placeholder_goal_reached_verification",
        }
        base = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="goal_reached_verification_result",
            summary=f"goal reached verification completed for {goal_ref}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/goal-reached-verification",
            refs=task_spec.target_refs,
            extra={
                "executor": "placeholder_goal_reached_verification_adapter",
                "operation_id": agent_input.context.operation_id,
                "goal_reached_verification": verification,
                "resource_keys": list(task_spec.resource_keys),
                "kg_refs": self._filter_refs(task_spec.target_refs, GraphScope.KG),
                "ag_refs": self._filter_refs(task_spec.target_refs, GraphScope.AG),
                "tg_refs": self._filter_refs(task_spec.target_refs, GraphScope.TG),
            },
        )
        return base | {
            "success": True,
            "confidence": 0.84 if goal_satisfied else 0.76,
            "executor": "placeholder_goal_reached_verification_adapter",
            "validation_status": "validated",
            "goal_satisfied": goal_satisfied,
            "goal_reached_verification": verification,
        }

    def _target_state_confirmation_adapter(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Placeholder adapter for future target-state confirmation tooling."""

        target_ref = (
            task_spec.input_bindings.get("target")
            or task_spec.input_bindings.get("target_id")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Target")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="State")
            or "unknown-target"
        )
        expected_state = (
            task_spec.input_bindings.get("expected_state")
            or task_spec.constraints.get("expected_state")
            or "confirmed"
        )
        observed_state = (
            task_spec.input_bindings.get("observed_state")
            or task_spec.constraints.get("observed_state")
            or expected_state
        )
        goal_satisfied = observed_state == expected_state
        confirmation = {
            "target_ref": target_ref,
            "expected_state": expected_state,
            "observed_state": observed_state,
            "goal_satisfied": goal_satisfied,
            "source": "placeholder_target_state_confirmation",
        }
        base = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="target_state_confirmation_result",
            summary=f"target state confirmation completed for {target_ref}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/target-state-confirmation",
            refs=task_spec.target_refs,
            extra={
                "executor": "placeholder_target_state_confirmation_adapter",
                "operation_id": agent_input.context.operation_id,
                "target_state_confirmation": confirmation,
                "resource_keys": list(task_spec.resource_keys),
                "kg_refs": self._filter_refs(task_spec.target_refs, GraphScope.KG),
                "ag_refs": self._filter_refs(task_spec.target_refs, GraphScope.AG),
                "tg_refs": self._filter_refs(task_spec.target_refs, GraphScope.TG),
            },
        )
        return base | {
            "success": True,
            "confidence": 0.82 if goal_satisfied else 0.7,
            "executor": "placeholder_target_state_confirmation_adapter",
            "validation_status": "validated",
            "goal_satisfied": goal_satisfied,
            "target_state_confirmation": confirmation,
        }

    @staticmethod
    def _filter_refs(refs: list[GraphRef], scope: GraphScope) -> list[dict[str, Any]]:
        """Return refs for a specific graph scope as serialized payloads."""

        return [ref.model_dump(mode="json") for ref in refs if ref.graph == scope]

    @staticmethod
    def _primary_ref_id(refs: list[GraphRef], *, preferred_type: str) -> str | None:
        """Return the first ref id matching the preferred ref type."""

        for ref in refs:
            if (ref.ref_type or "").lower() == preferred_type.lower():
                return ref.ref_id
        return refs[0].ref_id if refs else None


__all__ = ["GoalValidationWorker"]
