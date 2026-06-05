"""Vulnerability analysis stage agent."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.stage.advisors.vuln_analysis_advisor import VulnAnalysisAdvisor
from src.core.stage.base_stage_agent import BaseStageAgent


class VulnAnalysisAgent(BaseStageAgent):
    stage_type = "VULN_ANALYSIS_STAGE"
    agent_name = "vuln_analysis_agent"

    def __init__(
        self,
        *,
        llm_client: PackyLLMClient | None = None,
        advisor=None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> None:
        super().__init__(
            advisor=advisor or (VulnAnalysisAdvisor(client=llm_client) if llm_client is not None else None),
            mcp_client=mcp_client,
            default_timeout_seconds=default_timeout_seconds,
        )


__all__ = ["VulnAnalysisAgent"]
