"""Perception helpers for external C2 command output."""

from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.models.events import FactWriteKind, FactWriteRequest, ObservationRecord
from src.integrations.incalmo.models import CommandResult


def parse_incalmo_command_output(result: CommandResult, *, source_task_id: str) -> tuple[list[ObservationRecord], list[FactWriteRequest]]:
    """Parse command output into observations and fact write requests."""

    text = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    observations = [
        ObservationRecord(
            category="c2.command_output",
            summary=(text[:160] if text else f"Incalmo command {result.command_id} produced no output"),
            confidence=0.6,
            payload={"command_id": result.command_id, "agent_id": result.agent_id, "status": result.status.value},
        )
    ]
    subject = GraphRef(graph="kg", ref_id=f"incalmo-agent::{result.agent_id}", ref_type="C2Agent")
    facts = [
        FactWriteRequest(
            kind=FactWriteKind.ENTITY_UPSERT,
            source_task_id=source_task_id,
            subject_ref=subject,
            attributes={
                "command_id": result.command_id,
                "status": result.status.value,
                "exit_code": result.exit_code,
                "stdout_preview": result.stdout[:500],
                "stderr_preview": result.stderr[:500],
            },
            confidence=0.6,
            summary=f"Incalmo agent {result.agent_id} command {result.command_id} completed",
        )
    ]
    return observations, facts
