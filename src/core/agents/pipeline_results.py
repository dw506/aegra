"""Prompt-free pipeline cycle/step result models.

These structured results are retained for the LLM decision-history API
(`AppOrchestrator.record_llm_decision_cycle` / `LLMDecisionObserver`). They are
not tied to the legacy `AgentPipeline`; the stage-agent main path builds and
consumes lean domain objects instead.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_protocol import (
    AgentExecutionResult,
    AgentInput,
    AgentKind,
    AgentOutput,
)


class PipelineStepResult(BaseModel):
    """One executed pipeline step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step_name: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_kind: AgentKind
    success: bool
    agent_input: AgentInput
    agent_output: AgentOutput
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)

    @classmethod
    def from_execution(
        cls,
        *,
        step_name: str,
        agent_input: AgentInput,
        execution: AgentExecutionResult,
    ) -> "PipelineStepResult":
        return cls(
            step_name=step_name,
            agent_name=execution.agent_name,
            agent_kind=execution.agent_kind,
            success=execution.success,
            agent_input=agent_input,
            agent_output=execution.output,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            duration_ms=execution.duration_ms,
        )


class PipelineCycleResult(BaseModel):
    """One pipeline cycle result."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cycle_name: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    success: bool
    steps: list[PipelineStepResult] = Field(default_factory=list)
    final_output: AgentOutput = Field(default_factory=AgentOutput)
    logs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


__all__ = ["PipelineCycleResult", "PipelineStepResult"]
