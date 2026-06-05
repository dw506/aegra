"""Access and pivot stage agent."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.stage.advisors.access_pivot_advisor import AccessPivotAdvisor
from src.core.stage.base_stage_agent import BaseStageAgent


class AccessPivotAgent(BaseStageAgent):
    stage_type = "ACCESS_PIVOT_STAGE"
    agent_name = "access_pivot_agent"

    def __init__(
        self,
        *,
        llm_client: PackyLLMClient | None = None,
        advisor=None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> None:
        super().__init__(
            advisor=advisor or (AccessPivotAdvisor(client=llm_client) if llm_client is not None else None),
            mcp_client=mcp_client,
            default_timeout_seconds=default_timeout_seconds,
        )


__all__ = ["AccessPivotAgent"]
