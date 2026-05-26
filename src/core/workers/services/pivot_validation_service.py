"""Domain service for pivot route validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import AgentInput
from src.core.execution.executor import ExecutionExecutor
from src.core.execution.tool_plan import ToolPlan
from src.core.models.events import AgentResultStatus, RuntimeControlRequest, RuntimeControlType
from src.core.models.runtime import RuntimeState
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager
from src.core.workers.base import WorkerTaskSpec
from src.core.workers.services.result_builders import WorkerDomainResult


class PivotValidationRequest(BaseModel):
    """Domain input consumed by `PivotValidationService`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    operation_id: str
    task_id: str
    task_type: str
    task_label: str
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    target_refs: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    context_metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_task_spec(cls, *, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> "PivotValidationRequest":
        raw = dict(agent_input.raw_payload)
        metadata = dict(raw.get("metadata", {}))
        selected_route = task_spec.input_bindings.get("selected_route")
        route_id = task_spec.constraints.get("route_id") or task_spec.input_bindings.get("route_id")
        if isinstance(selected_route, dict):
            metadata["selected_route"] = dict(selected_route)
        if route_id:
            metadata.setdefault("selected_route", {})["route_id"] = str(route_id)
        return cls(
            operation_id=agent_input.context.operation_id,
            task_id=task_spec.task_id,
            task_type=task_spec.task_type,
            task_label=str(raw.get("task_label") or task_spec.task_type),
            input_bindings=dict(task_spec.input_bindings),
            target_refs=list(task_spec.target_refs),
            metadata=metadata,
            constraints=dict(task_spec.constraints),
            context_metadata=dict(agent_input.context.extra),
        )


class PivotValidationService:
    """Validate a selected pivot route with an optional adapter-backed probe."""

    def __init__(self, *, executor: ExecutionExecutor | None = None) -> None:
        self._executor = executor
        self._pivot_routes = RuntimePivotRouteManager()

    def validate(self, request: PivotValidationRequest) -> WorkerDomainResult:
        runtime_state = self._runtime_snapshot(request)
        selected_route = self._selected_route(request, runtime_state)
        route_id = self._string(selected_route.get("route_id"))
        destination_host = self._string(
            selected_route.get("destination_host")
            or request.input_bindings.get("target_host_id")
            or request.input_bindings.get("destination_host")
            or request.input_bindings.get("host_id")
        )
        if route_id is None and destination_host is None:
            return WorkerDomainResult(
                success=False,
                status=AgentResultStatus.BLOCKED.value,
                summary="pivot validation requires a route_id or destination host",
                raw_payload={"blocked_on": "pivot_route", "validated": False},
            )

        tool_execution = self._run_probe(request, selected_route)
        reachable = bool(tool_execution.get("success", True))
        status = AgentResultStatus.SUCCEEDED.value if reachable else AgentResultStatus.BLOCKED.value
        runtime_requests = [
            RuntimeControlRequest(
                request_type=RuntimeControlType.VERIFY_PIVOT_ROUTE,
                source_task_id=request.task_id,
                metadata={
                    "route_id": route_id,
                    "reachable": reachable,
                    "selected_route": selected_route,
                    "tool_execution": tool_execution or None,
                },
                reason=None if reachable else "pivot probe failed",
            ).model_dump(mode="json")
        ]
        reachability = {
            "reachable": reachable,
            "via": "pivot",
            "route_id": route_id,
            "target_id": destination_host,
            "source_id": selected_route.get("source_host"),
            "via_host": selected_route.get("via_host"),
            "protocol": selected_route.get("protocol"),
            "allowed_ports": selected_route.get("allowed_ports"),
        }
        raw_payload = {
            "status": status,
            "validated": reachable,
            "selected_route": selected_route,
            "reachability": reachability,
            "runtime_requests": runtime_requests,
        }
        if tool_execution:
            raw_payload["tool_execution"] = tool_execution
        return WorkerDomainResult(
            success=reachable,
            status=status,
            summary=("pivot route validated" if reachable else "pivot route probe failed"),
            confidence=0.85 if reachable else 0.4,
            evidence=[
                {
                    "kind": "pivot_validation",
                    "summary": f"Pivot validation evidence for {request.task_id}",
                    "payload_ref": f"runtime://outcomes/{request.task_id}/pivot",
                    "metadata": {"selected_route_id": route_id, "tool_execution": tool_execution},
                }
            ],
            runtime_requests=runtime_requests,
            raw_payload=raw_payload,
        )

    def _run_probe(self, request: PivotValidationRequest, selected_route: dict[str, Any]) -> dict[str, Any]:
        if self._executor is None:
            return {}
        command = request.metadata.get("probe_command") or request.input_bindings.get("probe_command")
        adapter = request.metadata.get("execution_adapter") or request.input_bindings.get("execution_adapter")
        if command is None or adapter is None:
            return {}
        args = dict(request.input_bindings)
        args.setdefault("selected_route", selected_route)
        if selected_route.get("route_id"):
            args.setdefault("route_id", selected_route.get("route_id"))
        if isinstance(command, list):
            args["argv"] = [str(part) for part in command]
            command_text = " ".join(str(part) for part in command)
        else:
            command_text = str(command)
        result = self._executor.execute(
            ToolPlan(
                task_id=request.task_id,
                tool="pivot_probe",
                adapter=str(adapter),
                command=command_text,
                args=args,
                payloads=args,
                timeout_seconds=int(request.metadata.get("timeout_seconds") or request.input_bindings.get("timeout_seconds") or 30),
                metadata={
                    "route_id": selected_route.get("route_id"),
                    "selected_route": selected_route,
                    "session_id": selected_route.get("session_id"),
                },
            )
        )
        return result.model_dump(mode="json")

    def _selected_route(self, request: PivotValidationRequest, runtime_state: RuntimeState | None) -> dict[str, Any]:
        selected = dict(request.metadata.get("selected_route")) if isinstance(request.metadata.get("selected_route"), dict) else {}
        if isinstance(request.input_bindings.get("selected_route"), dict):
            selected = dict(request.input_bindings["selected_route"]) | selected
        route_id = self._string(selected.get("route_id") or request.input_bindings.get("route_id") or request.constraints.get("route_id"))
        if route_id is not None:
            selected.setdefault("route_id", route_id)
        if runtime_state is not None and route_id is not None and route_id in runtime_state.pivot_routes:
            route = self._pivot_routes.get_route(runtime_state, route_id)
            return route.model_dump(mode="json") | selected
        return selected

    @staticmethod
    def _runtime_snapshot(request: PivotValidationRequest) -> RuntimeState | None:
        raw = request.metadata.get("runtime_snapshot") or request.context_metadata.get("runtime_snapshot")
        if raw is None:
            return None
        if isinstance(raw, RuntimeState):
            return raw.model_copy(deep=True)
        if isinstance(raw, dict):
            return RuntimeState.model_validate(raw)
        raise ValueError("runtime_snapshot must be a RuntimeState or serialized runtime snapshot")

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["PivotValidationRequest", "PivotValidationService"]
