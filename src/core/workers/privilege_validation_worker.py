"""Privilege validation worker for privilege-state confirmation tasks."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput, GraphRef, GraphScope
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec


class PrivilegeValidationWorker(BaseWorkerAgent):
    """Concrete worker for privilege validation and escalation verification.

    This worker performs validation-only execution. It does not mutate KG, AG,
    or TG directly, and it does not perform planning. Its responsibility is to:
    - validate privilege state or escalation results
    - emit one `OutcomeRecord`
    - emit one raw result pointer payload

    Future real integrations should replace the placeholder adapters while
    preserving the same output contract.
    """

    capabilities = frozenset(
        {
            WorkerCapability.PRIVILEGE_VALIDATION,
            WorkerCapability.CONTEXT_VALIDATION,
        }
    )
    supported_task_types = frozenset(
        {
            "privilege_validation",
            "privilege_escalation_verification",
        }
    )

    def __init__(self, name: str = "privilege_validation_worker") -> None:
        super().__init__(name=name)

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        """Return True when the task is one of the supported privilege operations."""

        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        """Execute one privilege-validation task and return outcome plus raw result."""

        if task_spec.task_type == "privilege_validation":
            raw_result = self._execute_privilege_validation(task_spec, agent_input)
        elif task_spec.task_type == "privilege_escalation_verification":
            raw_result = self._execute_privilege_escalation_verification(task_spec, agent_input)
        else:
            raise ValueError(f"unsupported privilege validation task type: {task_spec.task_type}")

        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=bool(raw_result.get("success", True)),
            summary=str(raw_result["summary"]),
            raw_result_ref=str(raw_result["payload_ref"]),
            confidence=float(raw_result.get("confidence", 0.75)),
            refs=task_spec.target_refs,
            payload={
                "task_type": task_spec.task_type,
                "executor": raw_result.get("executor", "placeholder"),
                "result_type": raw_result.get("result_type", task_spec.task_type),
                "validation_status": raw_result.get("validation_status", "unknown"),
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

    def _execute_privilege_validation(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Execute privilege validation via the current adapter."""

        return self._privilege_validation_adapter(task_spec, agent_input)

    def _execute_privilege_escalation_verification(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Execute privilege escalation verification via the current adapter."""

        return self._privilege_escalation_verification_adapter(task_spec, agent_input)

    def _privilege_validation_adapter(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Placeholder adapter for future privilege validation tooling."""

        principal = (
            task_spec.input_bindings.get("principal")
            or task_spec.input_bindings.get("identity")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Identity")
            or "unknown-principal"
        )
        privilege_level = (
            task_spec.input_bindings.get("privilege_level")
            or task_spec.constraints.get("required_privilege")
            or "unknown"
        )
        session_hint = (
            task_spec.constraints.get("session_id")
            or task_spec.input_bindings.get("session_id")
            or "no-session"
        )
        validation = {
            "principal": principal,
            "privilege_level": privilege_level,
            "session_id": session_hint,
            "validated": True,
            "source": "placeholder_privilege_validation",
        }
        base = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="privilege_validation_result",
            summary=f"privilege validation completed for {principal}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/privilege-validation",
            refs=task_spec.target_refs,
            extra={
                "executor": "placeholder_privilege_validation_adapter",
                "operation_id": agent_input.context.operation_id,
                "privilege_validation": validation,
                "resource_keys": list(task_spec.resource_keys),
                "kg_refs": self._filter_refs(task_spec.target_refs, GraphScope.KG),
                "ag_refs": self._filter_refs(task_spec.target_refs, GraphScope.AG),
                "runtime_refs": self._filter_refs(task_spec.target_refs, GraphScope.RUNTIME),
            },
        )
        return base | {
            "success": True,
            "confidence": 0.79,
            "executor": "placeholder_privilege_validation_adapter",
            "validation_status": "validated",
            "privilege_validation": validation,
        }

    def _privilege_escalation_verification_adapter(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Placeholder adapter for future privilege escalation verification tooling."""

        principal = (
            task_spec.input_bindings.get("principal")
            or task_spec.input_bindings.get("identity")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Identity")
            or "unknown-principal"
        )
        target_state = (
            task_spec.input_bindings.get("target_privilege_state")
            or task_spec.constraints.get("expected_state")
            or "elevated"
        )
        escalation_path = (
            task_spec.input_bindings.get("escalation_path")
            or task_spec.constraints.get("path")
            or "unknown-path"
        )
        verification = {
            "principal": principal,
            "target_privilege_state": target_state,
            "escalation_path": escalation_path,
            "verified": True,
            "source": "placeholder_privilege_escalation_verification",
        }
        base = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="privilege_escalation_verification_result",
            summary=f"privilege escalation verification completed for {principal}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/privilege-escalation-verification",
            refs=task_spec.target_refs,
            extra={
                "executor": "placeholder_privilege_escalation_verification_adapter",
                "operation_id": agent_input.context.operation_id,
                "privilege_escalation_verification": verification,
                "resource_keys": list(task_spec.resource_keys),
                "kg_refs": self._filter_refs(task_spec.target_refs, GraphScope.KG),
                "ag_refs": self._filter_refs(task_spec.target_refs, GraphScope.AG),
                "runtime_refs": self._filter_refs(task_spec.target_refs, GraphScope.RUNTIME),
            },
        )
        return base | {
            "success": True,
            "confidence": 0.76,
            "executor": "placeholder_privilege_escalation_verification_adapter",
            "validation_status": "validated",
            "privilege_escalation_verification": verification,
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


__all__ = ["PrivilegeValidationWorker"]
