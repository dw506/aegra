"""Execution adapter for an external Incalmo C2 API."""

from __future__ import annotations

from typing import Any

from src.core.execution.tool_plan import ToolPlan
from src.core.execution.tool_result import ToolExecutionResult
from src.integrations.incalmo.client import IncalmoClient


class IncalmoC2Adapter:
    """Run ToolPlans through Incalmo and return adapter-neutral results."""

    name = "incalmo_c2"

    def __init__(self, client: IncalmoClient) -> None:
        self.client = client

    def supports(self, plan: ToolPlan) -> bool:
        return plan.adapter == self.name

    def execute(self, plan: ToolPlan) -> ToolExecutionResult:
        agent_id = self._resolve_agent_id(plan)
        command = self._resolve_command(plan)
        command_response = self.client.send_command(
            agent_id=agent_id,
            command=command,
            payloads=plan.payloads or plan.args or [],
        )
        raw_command_response = self._coerce_mapping(command_response)
        command_id = self._extract_command_id(raw_command_response)
        result = self.client.wait_for_command_result(
            command_id=command_id,
            max_attempts=max(1, int(plan.timeout_seconds)),
        )
        raw_result = self._coerce_mapping(result)
        exit_code = raw_result.get("exit_code")
        status = str(raw_result.get("status") or "").lower()
        success = str(exit_code).lower() in {"0", "success"} or status in {"completed", "succeeded", "success"}
        return ToolExecutionResult(
            adapter=self.name,
            tool=plan.tool,
            success=success,
            exit_code=exit_code,
            stdout=str(raw_result.get("output") or raw_result.get("stdout") or ""),
            stderr=str(raw_result.get("stderr") or ""),
            command_id=command_id,
            payload_ref=str(raw_result.get("payload_ref") or f"incalmo://commands/{command_id}"),
            metadata={
                "agent_id": agent_id,
                "raw_command_response": raw_command_response,
                "raw_result": raw_result,
                "tool_plan": plan.model_dump(mode="json"),
            },
        )

    def _resolve_agent_id(self, plan: ToolPlan) -> str:
        agent_id = plan.target_agent_ref or plan.metadata.get("agent_id") or plan.args.get("agent_id")
        if agent_id is None:
            raise ValueError("Incalmo C2 execution requires target_agent_ref, metadata.agent_id, or args.agent_id")
        value = str(agent_id).strip()
        if not value:
            raise ValueError("Incalmo C2 execution requires a non-empty agent id")
        return value

    def _resolve_command(self, plan: ToolPlan) -> str:
        command = plan.command or plan.args.get("command")
        if command is None:
            raise ValueError("Incalmo C2 execution requires command or args.command")
        value = str(command).strip()
        if not value:
            raise ValueError("Incalmo C2 execution requires a non-empty command")
        return value

    def _extract_command_id(self, payload: dict[str, Any]) -> str:
        command_id = payload.get("id") or payload.get("command_id") or payload.get("uuid")
        if command_id is None:
            raise ValueError("Incalmo C2 command response did not include an id")
        value = str(command_id).strip()
        if not value:
            raise ValueError("Incalmo C2 command response included an empty id")
        return value

    def _coerce_mapping(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "model_dump"):
            return dict(value.model_dump(mode="json"))
        raise TypeError(f"Incalmo C2 client returned unsupported payload type: {type(value).__name__}")


__all__ = ["IncalmoC2Adapter"]
