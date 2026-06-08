"""Recon stage agent."""

from typing import Any

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent


class ReconAgent(LLMDrivenStageAgent):
    stage_type = "RECON_STAGE"
    agent_name = "recon_agent"
    role_prompt = (
        "你是 Aegra 的自主侦察 Agent，用于本地授权靶场。你可以使用 ToolCatalog 中的所有 MCP 工具，"
        "包括 nmap_scan、http_probe、web_fingerprint、web_discover、dns_lookup、tcp_connect_probe、run_command 等。"
        "你的任务是从用户输入目标和已有 evidence 出发，自主发现可达主机、服务、端口、版本、Web 指纹和负面证据。"
        "不要虚构事实。所有发现必须来自工具输出或已有 KG/Runtime evidence。"
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


__all__ = ["ReconAgent"]
