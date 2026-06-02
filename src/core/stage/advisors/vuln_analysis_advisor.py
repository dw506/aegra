"""Dedicated VulnAnalysisAgent advisor."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.stage.advisors._llm_stage_agent_advisor import DedicatedLLMStageAdvisor
from src.core.stage.context.vuln_analysis_context import build_vuln_analysis_context


class VulnAnalysisAdvisor(DedicatedLLMStageAdvisor):
    def __init__(self, *, client: PackyLLMClient) -> None:
        super().__init__(
            agent_name="vuln_analysis_agent",
            stage_type="VULN_ANALYSIS_STAGE",
            client=client,
            context_builder=build_vuln_analysis_context,
        )


__all__ = ["VulnAnalysisAdvisor"]
