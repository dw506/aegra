"""Access validation worker for path, session, and access-context checks."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_protocol import AgentInput, AgentOutput, GraphRef, GraphScope
from src.core.workers.base import BaseWorkerAgent, WorkerCapability, WorkerTaskSpec


class AccessValidationWorker(BaseWorkerAgent):
    """Concrete worker for access and session/path validation tasks.

    This worker executes validation-style work only. It does not mutate KG, AG,
    or TG directly. Its responsibility is to validate access conditions and emit:
    - one `OutcomeRecord`
    - one raw result pointer payload

    Future real integrations should replace the placeholder adapter methods
    while preserving the same output contract.
    """

    capabilities = frozenset(
        {
            WorkerCapability.ACCESS_VALIDATION,
            WorkerCapability.CONTEXT_VALIDATION,
        }
    )
    supported_task_types = frozenset(
        {
            "access_validation",
            "session_path_validation",
        }
    )

    def __init__(self, name: str = "access_validation_worker") -> None:
        super().__init__(name=name)

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        """Return True when the task is one of the supported access operations."""

        return task_spec.task_type in self.supported_task_types

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        """Execute one access-validation task and return outcome plus raw result."""

        if task_spec.task_type == "access_validation":
            raw_result = self._execute_access_validation(task_spec, agent_input)
        elif task_spec.task_type == "session_path_validation":
            raw_result = self._execute_session_path_validation(task_spec, agent_input)
        else:
            raise ValueError(f"unsupported access validation task type: {task_spec.task_type}")

        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type=task_spec.task_type,
            success=bool(raw_result.get("success", True)),
            summary=str(raw_result["summary"]),
            raw_result_ref=str(raw_result["payload_ref"]),
            confidence=float(raw_result.get("confidence", 0.72)),
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

    def _execute_access_validation(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Execute access validation via the current validation adapter."""

        return self._access_validation_adapter(task_spec, agent_input)

    def _execute_session_path_validation(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Execute session/path validation via the current validation adapter."""

        return self._session_path_validation_adapter(task_spec, agent_input)

    def _access_validation_adapter(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> dict[str, Any]:
        """Placeholder adapter for future access validation tooling."""

        access_target = (
            task_spec.input_bindings.get("access_target")
            or task_spec.input_bindings.get("principal")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Identity")
            or "unknown-access-target"
        )
        path_hint = (
            task_spec.input_bindings.get("path")
            or task_spec.constraints.get("path")
            or task_spec.constraints.get("target_path")
        )
        session_hint = (
            task_spec.constraints.get("session_id")
            or task_spec.input_bindings.get("session_id")
            or "no-session"
        )
        validation = {
            "access_target": access_target,
            "path": path_hint,
            "session_id": session_hint,
            "access_validated": True,
            "source": "placeholder_access_validation",
        }
        base = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="access_validation_result",
            summary=f"access validation completed for {access_target}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/access-validation",
            refs=task_spec.target_refs,
            extra={
                "executor": "placeholder_access_validation_adapter",
                "operation_id": agent_input.context.operation_id,
                "access_validation": validation,
                "resource_keys": list(task_spec.resource_keys),
                "kg_refs": self._filter_refs(task_spec.target_refs, GraphScope.KG),
                "ag_refs": self._filter_refs(task_spec.target_refs, GraphScope.AG),
                "runtime_refs": self._filter_refs(task_spec.target_refs, GraphScope.RUNTIME),
            },
        )
        return base | {
            "success": True,
            "confidence": 0.78,
            "executor": "placeholder_access_validation_adapter",
            "validation_status": "validated",
            "access_validation": validation,
        }

    def _session_path_validation_adapter(
        self,
        task_spec: WorkerTaskSpec,
        agent_input: AgentInput,
    ) -> dict[str, Any]:
        """Placeholder adapter for future session/path validation tooling."""

        session_hint = (
            task_spec.constraints.get("session_id")
            or task_spec.input_bindings.get("session_id")
            or self._primary_ref_id(task_spec.target_refs, preferred_type="Session")
            or "unknown-session"
        )
        path_hint = (
            task_spec.input_bindings.get("path")
            or task_spec.input_bindings.get("route")
            or task_spec.constraints.get("path")
            or "unknown-path"
        )
        context_hint = (
            task_spec.input_bindings.get("context")
            or task_spec.constraints.get("access_context")
            or "default-context"
        )
        validation = {
            "session_id": session_hint,
            "path": path_hint,
            "context": context_hint,
            "path_validated": True,
            "source": "placeholder_session_path_validation",
        }
        base = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="session_path_validation_result",
            summary=f"session/path validation completed for {session_hint}",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}/session-path-validation",
            refs=task_spec.target_refs,
            extra={
                "executor": "placeholder_session_path_validation_adapter",
                "operation_id": agent_input.context.operation_id,
                "session_path_validation": validation,
                "resource_keys": list(task_spec.resource_keys),
                "kg_refs": self._filter_refs(task_spec.target_refs, GraphScope.KG),
                "ag_refs": self._filter_refs(task_spec.target_refs, GraphScope.AG),
                "runtime_refs": self._filter_refs(task_spec.target_refs, GraphScope.RUNTIME),
            },
        )
        return base | {
            "success": True,
            "confidence": 0.74,
            "executor": "placeholder_session_path_validation_adapter",
            "validation_status": "validated",
            "session_path_validation": validation,
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


__all__ = ["AccessValidationWorker"]
