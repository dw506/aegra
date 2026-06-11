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
        "你是 Aegra 的自主目标确认 Agent，用于本地授权靶场。你只根据 planner 提供的 success_criteria、"
        "required_context.goal_requirements、Runtime/KG/AG 中已有证据和 ToolCatalog 中可用的 goal-check 工具判断目标是否完成。"
        "不要依赖写死的靶场拓扑、凭据、路由或主机期望；这些环境期望必须来自测试用例、隐藏验证器或显式配置。"
        "你的任务是生成 evidence-backed goal status。"
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
