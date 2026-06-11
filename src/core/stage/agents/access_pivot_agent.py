"""Access and pivot stage agent."""

from typing import Any

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent


class AccessPivotAgent(LLMDrivenStageAgent):
    stage_type = "ACCESS_PIVOT_STAGE"
    agent_name = "access_pivot_agent"
    role_prompt = (
        "你是 Aegra 的自主访问与 pivot 观察 Agent，用于本地授权多主机靶场。你可以使用 ToolCatalog 中的所有 MCP 工具，"
        "包括 session_probe、session_open_lab、identity_context_probe、privilege_context_probe、pivot_route_probe、"
        "internal_service_discover、pivoted_nmap_scan、controlled_data_read_proof、run_command 等。你的任务是基于工具证据识别 session、"
        "credential、route、reachable subnet、internal service，并记录受控证明。不要依赖硬编码拓扑。"
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


__all__ = ["AccessPivotAgent"]
