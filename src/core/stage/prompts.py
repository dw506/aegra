"""Shared prompt templates for stage-level agents."""

from __future__ import annotations

BASE_STAGE_AGENT_PROMPT = """You are Aegra's {stage_name}.
You complete a stage-level authorized security validation task, not a single tool call.
You may call MCP tools for this stage in multiple bounded steps.
Use Graph Context and Runtime Context to choose the next step.
Do not write KG, AG or Runtime directly.
Return only StageResult-compatible JSON when finishing.
Record what you did, what you observed, capabilities gained, failed hypotheses, and next stage candidates.
"""


STAGE_PROMPTS: dict[str, str] = {
    "RECON_STAGE": BASE_STAGE_AGENT_PROMPT.format(stage_name="ReconAgent"),
    "VULN_ANALYSIS_STAGE": BASE_STAGE_AGENT_PROMPT.format(stage_name="VulnAnalysisAgent"),
    "EXPLOIT_STAGE": BASE_STAGE_AGENT_PROMPT.format(stage_name="ExploitValidationAgent"),
    "ACCESS_PIVOT_STAGE": BASE_STAGE_AGENT_PROMPT.format(stage_name="AccessPivotAgent"),
    "GOAL_STAGE": BASE_STAGE_AGENT_PROMPT.format(stage_name="GoalAgent"),
}


__all__ = ["BASE_STAGE_AGENT_PROMPT", "STAGE_PROMPTS"]
