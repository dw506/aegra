"""Mapping between Incalmo protocol objects and Aegra runtime/result models."""

from __future__ import annotations

from src.core.models.events import AgentResultStatus, AgentRole, AgentTaskResult, EvidenceArtifact
from src.core.models.runtime import PivotRouteRuntime, PivotRouteStatus, SessionRuntime, SessionStatus
from src.integrations.incalmo.models import Agent, CommandResult, CommandStatus


class IncalmoMapper:
    """Convert Incalmo objects into Aegra-owned runtime/result records."""

    def agent_to_session(self, agent: Agent) -> SessionRuntime:
        return SessionRuntime(
            session_id=f"incalmo::{agent.agent_id}",
            status=SessionStatus.ACTIVE if str(agent.status or "").lower() in {"active", "online", "ready"} else SessionStatus.OPENING,
            bound_target=agent.address or agent.hostname or agent.agent_id,
            metadata={"integration": "incalmo", "agent": agent.model_dump(mode="json")},
        )

    def agent_to_pivot(self, agent: Agent, *, destination_host: str) -> PivotRouteRuntime:
        return PivotRouteRuntime(
            route_id=f"incalmo-pivot::{agent.agent_id}::{destination_host}",
            destination_host=destination_host,
            via_host=agent.address or agent.hostname,
            session_id=f"incalmo::{agent.agent_id}",
            status=PivotRouteStatus.CANDIDATE,
            metadata={"integration": "incalmo", "agent_id": agent.agent_id},
        )

    def command_result_to_task_result(
        self,
        result: CommandResult,
        *,
        operation_id: str,
        task_id: str,
        tg_node_id: str,
    ) -> AgentTaskResult:
        status = AgentResultStatus.SUCCEEDED if result.status == CommandStatus.SUCCEEDED else AgentResultStatus.FAILED
        if result.status == CommandStatus.TIMEOUT:
            status = AgentResultStatus.FAILED
        evidence = EvidenceArtifact(
            kind="incalmo_command_output",
            summary=f"Incalmo command {result.command_id} returned {result.status.value}",
            payload_ref=f"incalmo://commands/{result.command_id}",
            tool_output_ref=f"incalmo://agents/{result.agent_id}/commands/{result.command_id}",
            metadata=result.model_dump(mode="json"),
        )
        return AgentTaskResult(
            request_id=f"incalmo::{result.command_id}",
            agent_role=AgentRole.RECON_WORKER,
            operation_id=operation_id,
            task_id=task_id,
            tg_node_id=tg_node_id,
            status=status,
            summary=evidence.summary,
            error_message=result.stderr or None if status != AgentResultStatus.SUCCEEDED else None,
            evidence=[evidence],
            outcome_payload={"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code},
            metadata={"integration": "incalmo", "command_result": result.model_dump(mode="json")},
        )
