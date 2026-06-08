"""Vulnerability analysis stage agent."""

from typing import Any

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.runtime.txt_trace_logger import TxtTraceLogger
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent


class VulnAnalysisAgent(LLMDrivenStageAgent):
    stage_type = "VULN_ANALYSIS_STAGE"
    agent_name = "vuln_analysis_agent"
    role_prompt = (
        "你是 Aegra 的自主漏洞分析 Agent，用于本地授权靶场。你可以使用 ToolCatalog 中的所有 MCP 工具，"
        "包括 web_fingerprint、web_discover、vuln_profile_match、validation_precheck、nuclei_scan、"
        "whatweb_fingerprint、run_command 等。你的任务是基于已有 evidence 和工具结果分析候选漏洞、"
        "验证前置条件、形成 finding 或 validation plan。不要在没有证据时声称存在漏洞。"
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


__all__ = ["VulnAnalysisAgent"]
