"""Vulnerability analysis stage agent."""

from src.core.stage.base_stage_agent import BaseStageAgent


class VulnAnalysisAgent(BaseStageAgent):
    stage_type = "VULN_ANALYSIS_STAGE"
    agent_name = "vuln_analysis_agent"
    tool_categories = frozenset({"vuln", "cve", "exploit_doc", "profile", "precheck", "repo"})


__all__ = ["VulnAnalysisAgent"]
