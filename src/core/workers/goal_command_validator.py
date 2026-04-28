"""Command-backed goal validator adapter."""

from __future__ import annotations

import json
from typing import Any

from src.core.models.events import AgentTaskRequest
from src.core.workers.goal_validator import GoalEvaluation, GoalValidator, MetadataGoalValidator
from src.core.workers.tool_runner import ToolExecutionSpec, ToolRunner


class CommandGoalValidator(GoalValidator):
    """Use an external validator command when provided, else fall back to metadata."""

    def __init__(
        self,
        *,
        tool_runner: ToolRunner | None = None,
        fallback: GoalValidator | None = None,
    ) -> None:
        self._tool_runner = tool_runner or ToolRunner()
        self._fallback = fallback or MetadataGoalValidator()

    def evaluate(self, request: AgentTaskRequest) -> GoalEvaluation:
        command = request.metadata.get("goal_validator_command")
        if not isinstance(command, list) or not command:
            return self._fallback.evaluate(request)
        result = self._tool_runner.run(
            ToolExecutionSpec(
                command=[str(item) for item in command],
                timeout_sec=int(request.metadata.get("goal_validator_timeout_sec", 15)),
                retries=int(request.metadata.get("goal_validator_retries", 0)),
                cwd=request.metadata.get("goal_validator_cwd"),
                env={str(key): str(value) for key, value in dict(request.metadata.get("goal_validator_env", {})).items()},
            )
        )
        if result.category in {"command_not_found", "process_error", "timeout"}:
            return GoalEvaluation(
                satisfied=False,
                confidence=0.0,
                blocked=True,
                failure_reason=result.error_message or f"goal validator {result.category}",
                metadata={"tool": result.to_payload(), "source": "goal_validator_command"},
            )
        payload = self._parse_payload(result.stdout)
        if payload is None:
            if not result.success:
                return GoalEvaluation(
                    satisfied=False,
                    confidence=0.0,
                    failure_reason=result.stderr.strip() or result.error_message or "goal validator failed",
                    metadata={"tool": result.to_payload(), "source": "goal_validator_command"},
                )
            return GoalEvaluation(
                satisfied=True,
                validated_ref_ids=[ref.ref_id for ref in request.target_refs],
                confidence=float(request.metadata.get("confidence", 0.9)),
                metadata={"tool": result.to_payload(), "source": "goal_validator_command"},
            )
        return GoalEvaluation(
            satisfied=bool(payload.get("satisfied", result.success)),
            missing_requirements=[str(item) for item in payload.get("missing_requirements", []) if str(item).strip()],
            validated_ref_ids=[
                str(item) for item in payload.get("validated_ref_ids", [ref.ref_id for ref in request.target_refs]) if str(item).strip()
            ],
            supporting_evidence=[dict(item) for item in payload.get("supporting_evidence", []) if isinstance(item, dict)],
            confidence=float(payload.get("confidence", request.metadata.get("confidence", 0.9))),
            blocked=bool(payload.get("blocked", False)),
            failure_reason=(
                str(payload.get("failure_reason") or payload.get("reason"))
                if payload.get("failure_reason") is not None or payload.get("reason") is not None
                else None
            ),
            metadata={
                **{
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "satisfied",
                        "missing_requirements",
                        "validated_ref_ids",
                        "supporting_evidence",
                        "confidence",
                        "blocked",
                        "failure_reason",
                        "reason",
                    }
                },
                "tool": result.to_payload(),
                "source": "goal_validator_command",
            },
        )

    @staticmethod
    def _parse_payload(stdout: str) -> dict[str, Any] | None:
        text = stdout.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else {"value": payload}


__all__ = ["CommandGoalValidator"]
