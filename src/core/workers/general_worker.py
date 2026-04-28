"""Minimal general-purpose worker agent.

This worker provides a placeholder execution path for the most common early
task categories while keeping the interfaces stable for future integration with
real tool executors.
"""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput, GraphRef
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec


class GeneralWorkerAgent(BaseWorkerAgent):
    """MVP worker agent for basic recon, access validation and goal validation."""

    capabilities = frozenset(
        {
            WorkerCapability.RECON,
            WorkerCapability.ACCESS_VALIDATION,
            WorkerCapability.GOAL_VALIDATION,
        }
    )
    supported_task_types = frozenset({"recon", "access_validation", "goal_validation"})

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        """Return True when the task type is supported by the MVP worker."""

        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        """Execute one supported task through a placeholder executor."""

        if task_spec.task_type == "recon":
            raw_result = self._execute_recon(task_spec, agent_input)
        elif task_spec.task_type == "access_validation":
            raw_result = self._execute_access_validation(task_spec, agent_input)
        elif task_spec.task_type == "goal_validation":
            raw_result = self._execute_goal_validation(task_spec, agent_input)
        else:
            raise ValueError(f"unsupported task type: {task_spec.task_type}")

        raw_result_ref = raw_result["payload_ref"]
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=bool(raw_result.get("success", True)),
            summary=str(raw_result["summary"]),
            raw_result_ref=raw_result_ref,
            confidence=float(raw_result.get("confidence", 0.6)),
            refs=task_spec.target_refs,
            payload={
                "task_type": task_spec.task_type,
                "executor": raw_result.get("executor", "fake"),
                "result_type": raw_result.get("result_type"),
            },
        )
        return AgentOutput(
            outcomes=[outcome.to_agent_output_fragment()],
            logs=[
                f"worker={self.name}",
                f"task_id={task_spec.task_id}",
                f"task_type={task_spec.task_type}",
                f"raw_result_ref={raw_result_ref}",
                str(raw_result["summary"]),
            ],
            evidence=[raw_result],
        )

    def _execute_recon(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Placeholder executor for recon tasks."""

        host_hint = task_spec.input_bindings.get("host") or task_spec.input_bindings.get("host_id") or "unknown"
        return self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="recon_result",
            summary=f"mock recon completed for {host_hint}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/recon",
            refs=task_spec.target_refs,
            extra={
                "success": True,
                "confidence": 0.7,
                "executor": "fake_recon_executor",
                "discovered": [
                    {"host": host_hint, "status": "reachable"},
                ],
                "operation_id": agent_input.context.operation_id,
            },
        ) | {"success": True, "confidence": 0.7, "executor": "fake_recon_executor"}

    def _execute_access_validation(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Placeholder executor for access validation tasks."""

        identity = task_spec.input_bindings.get("identity") or task_spec.constraints.get("identity") or "unknown"
        target = self._primary_target_label(task_spec.target_refs)
        return self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="access_validation_result",
            summary=f"mock access validation completed for {identity} -> {target}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/access",
            refs=task_spec.target_refs,
            extra={
                "success": True,
                "confidence": 0.8,
                "executor": "fake_access_executor",
                "validated_identity": identity,
                "validated_target": target,
                "operation_id": agent_input.context.operation_id,
            },
        ) | {"success": True, "confidence": 0.8, "executor": "fake_access_executor"}

    def _execute_goal_validation(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Placeholder executor for goal validation tasks."""

        goal_id = task_spec.input_bindings.get("goal_id") or task_spec.constraints.get("goal_id") or "unknown-goal"
        simulated_success = bool(task_spec.constraints.get("expected_success", True))
        return self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="goal_validation_result",
            summary=f"mock goal validation {'passed' if simulated_success else 'failed'} for {goal_id}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/goal",
            refs=task_spec.target_refs,
            extra={
                "success": simulated_success,
                "confidence": 0.75,
                "executor": "fake_goal_executor",
                "goal_id": goal_id,
                "operation_id": agent_input.context.operation_id,
            },
        ) | {
            "success": simulated_success,
            "confidence": 0.75,
            "executor": "fake_goal_executor",
        }

    @staticmethod
    def _primary_target_label(target_refs: list[GraphRef]) -> str:
        """Return a readable primary target label for placeholder execution."""

        if not target_refs:
            return "unknown-target"
        primary = target_refs[0]
        return primary.metadata.get("label") or primary.ref_type or primary.ref_id


__all__ = ["GeneralWorkerAgent"]
