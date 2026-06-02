"""Goal verification stage agent."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.stage.advisors.goal_advisor import GoalAdvisor
from src.core.stage.base_stage_agent import BaseStageAgent


class GoalAgent(BaseStageAgent):
    stage_type = "GOAL_STAGE"
    agent_name = "goal_agent"
    allowed_tool_names = frozenset({"goal_check", "chain_goal_check", "internal_service_discover", "http_probe", "artifact_store"})
    denied_tool_names = frozenset({"safe_vuln_validate", "credential_check", "session_open_lab", "pivot_route_probe", "run_command"})
    tool_categories = frozenset({"goal", "check", "cleanup", "artifact"})

    def __init__(
        self,
        *,
        llm_client: PackyLLMClient | None = None,
        advisor=None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> None:
        super().__init__(
            advisor=advisor or (GoalAdvisor(client=llm_client) if llm_client is not None else None),
            mcp_client=mcp_client,
            default_timeout_seconds=default_timeout_seconds,
        )


__all__ = ["GoalAgent"]
