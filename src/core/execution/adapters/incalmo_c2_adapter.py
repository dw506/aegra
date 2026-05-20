"""Execution adapter for an external Incalmo C2 API."""

from __future__ import annotations

from src.core.execution.tool_plan import ToolPlan
from src.core.models.events import AgentResultStatus, AgentRole, AgentTaskResult
from src.core.models.runtime import RuntimeState
from src.integrations.incalmo.client import IncalmoClient
from src.integrations.incalmo.mapper import IncalmoMapper
from src.integrations.incalmo.perception import parse_incalmo_command_output


class IncalmoC2Adapter:
    """Run ToolPlans through Incalmo without mutating Aegra stores directly."""

    def __init__(self, client: IncalmoClient, mapper: IncalmoMapper | None = None) -> None:
        self._client = client
        self._mapper = mapper or IncalmoMapper()

    def execute(self, plan: ToolPlan, runtime_state: RuntimeState) -> AgentTaskResult:
        agent_ref = plan.target_agent_ref or str(plan.payloads.get("agent_id") or "")
        if not agent_ref:
            return AgentTaskResult(
                request_id=f"incalmo::{plan.task_id}",
                agent_role=AgentRole.RECON_WORKER,
                operation_id=runtime_state.operation_id,
                task_id=plan.task_id,
                tg_node_id=str(plan.metadata.get("tg_node_id") or plan.task_id),
                status=AgentResultStatus.BLOCKED,
                summary="Incalmo C2 execution requires target_agent_ref or agent_id",
                metadata={"tool_plan": plan.model_dump(mode="json")},
            )
        try:
            command = self._client.send_command(agent_ref, plan.command, plan.payloads)
            result = self._client.wait_for_command(
                agent_ref,
                command.command_id,
                timeout_sec=plan.timeout_seconds,
            )
        except Exception as exc:
            return AgentTaskResult(
                request_id=f"incalmo::{plan.task_id}",
                agent_role=AgentRole.RECON_WORKER,
                operation_id=runtime_state.operation_id,
                task_id=plan.task_id,
                tg_node_id=str(plan.metadata.get("tg_node_id") or plan.task_id),
                status=AgentResultStatus.FAILED,
                summary="Incalmo C2 command failed before result collection",
                error_message=str(exc),
                metadata={"tool_plan": plan.model_dump(mode="json"), "integration": "incalmo"},
            )
        task_result = self._mapper.command_result_to_task_result(
            result,
            operation_id=runtime_state.operation_id,
            task_id=plan.task_id,
            tg_node_id=str(plan.metadata.get("tg_node_id") or plan.task_id),
        )
        observations, facts = parse_incalmo_command_output(result, source_task_id=plan.task_id)
        task_result.observations.extend(observations)
        task_result.fact_write_requests.extend(facts)
        return task_result
