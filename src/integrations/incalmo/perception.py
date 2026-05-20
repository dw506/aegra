"""Perception parser plugin for Incalmo C2 command output."""

from __future__ import annotations

from typing import Any

from src.core.agents.agent_models import OutcomeRecord
from src.core.models.ag import GraphRef
from src.core.models.events import FactWriteKind, FactWriteRequest, ObservationRecord
from src.core.perception.parser_protocol import ParsedWorkerResult
from src.integrations.incalmo.models import CommandResult


class IncalmoC2CommandParser:
    """Parser plugin for Incalmo C2 command raw results."""

    name = "incalmo_c2_command_parser"

    def supports(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> bool:
        return raw_result.get("adapter") == "incalmo_c2" or raw_result.get("integration") == "incalmo"

    def parse(self, raw_result: dict[str, Any], outcome: OutcomeRecord) -> ParsedWorkerResult:
        return ParsedWorkerResult(
            observations=[
                {
                    "summary": raw_result.get("summary") or outcome.summary,
                    "payload": {
                        "adapter": "incalmo_c2",
                        "task_id": outcome.task_id,
                        "stdout": raw_result.get("stdout"),
                        "stderr": raw_result.get("stderr"),
                        "exit_code": raw_result.get("exit_code"),
                    },
                }
            ],
            evidence=[
                {
                    "summary": raw_result.get("evidence_summary") or raw_result.get("summary") or outcome.summary,
                    "payload_ref": raw_result.get("payload_ref") or outcome.raw_result_ref,
                    "payload": {"adapter": "incalmo_c2", "raw_result": dict(raw_result)},
                }
            ],
            metadata={"parser": self.name},
        )


def parse_incalmo_command_output(
    result: CommandResult,
    *,
    source_task_id: str,
) -> tuple[list[ObservationRecord], list[FactWriteRequest]]:
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


__all__ = ["IncalmoC2CommandParser", "parse_incalmo_command_output"]
