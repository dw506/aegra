"""Goal verification stage agent."""

from typing import Any

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent


class GoalAgent(LLMDrivenStageAgent):
    stage_type = "GOAL_STAGE"
    agent_name = "goal_agent"
    role_prompt = (
        "你是 Aegra 的自主目标确认 Agent，用于本地授权靶场。你可以使用 ToolCatalog 中的所有 MCP 工具，"
        "包括 goal_check、chain_goal_check、artifact_store、http_probe、run_command 等。"
        "你的任务是根据 evidence 判断目标是否完成，并生成 evidence-backed goal status。"
    )

    def __init__(
        self,
        *,
        llm_client: PackyLLMClient | None = None,
        mcp_client: MCPClient | None = None,
        operation_logger: TxtTraceLogger | None = None,
        default_timeout_seconds: int = 120,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            mcp_client=mcp_client,
            operation_logger=operation_logger,
            default_timeout_seconds=default_timeout_seconds,
            **kwargs,
        )


__all__ = ["GoalAgent"]
