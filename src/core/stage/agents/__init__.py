"""The single execution agent (P2: 5 stage agents collapsed into 1)."""

from __future__ import annotations

from typing import Any

from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent


EXECUTION_AGENT_ROLE_PROMPT = (
    "You are Aegra's bounded ExecutionAgent for an authorized assessment. "
    "Complete only the assigned capability round, using the supplied tools and scope. "
    "You may perform real authorized exploitation, command execution, session work, pivoting, "
    "lateral movement, and controlled data reads when the directive and policy allow it. "
    "Do not decide global success or next steps. Do not write KG, AG, Runtime, or secrets directly; "
    "return tool traces, evidence refs, findings, and redacted summaries for Verify/Write."
)


class ExecutionStageAgent(LLMDrivenStageAgent):
    """One capability-agnostic executor that runs every round.

    The capability/stage is a per-round tag on the ``RoundDirective`` /
    ``StageExecutionRequest``, not a separate agent class. This agent accepts any
    stage_type / agent_name on the request so the planner can dispatch any
    capability round to the single executor (``accepts_any_request = True``).
    """

    accepts_any_request = True

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("stage_type", "RECON_STAGE")  # identity tag only; serves all
        kwargs.setdefault("agent_name", "execution_agent")
        kwargs.setdefault("role_prompt", EXECUTION_AGENT_ROLE_PROMPT)
        super().__init__(**kwargs)


__all__ = [
    "EXECUTION_AGENT_ROLE_PROMPT",
    "ExecutionStageAgent",
]
