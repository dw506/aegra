"""Recon stage agent."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.stage.advisors.recon_advisor import ReconAdvisor
from src.core.stage.base_stage_agent import BaseStageAgent


class ReconAgent(BaseStageAgent):
    stage_type = "RECON_STAGE"
    agent_name = "recon_agent"
    allowed_tool_names = frozenset({"nmap_scan", "http_probe", "web_fingerprint", "web_discover", "dns_lookup", "tls_probe", "tcp_connect_probe"})
    denied_tool_names = frozenset({"run_command", "safe_vuln_validate", "credential_check", "session_open_lab", "pivot_route_probe"})
    tool_categories = frozenset({"nmap", "scan", "probe", "discover", "dns", "tls", "fingerprint"})

    def __init__(
        self,
        *,
        llm_client: PackyLLMClient | None = None,
        advisor=None,
        mcp_client: MCPClient | None = None,
        default_timeout_seconds: int = 60,
    ) -> None:
        super().__init__(
            advisor=advisor or (ReconAdvisor(client=llm_client) if llm_client is not None else None),
            mcp_client=mcp_client,
            default_timeout_seconds=default_timeout_seconds,
        )


__all__ = ["ReconAgent"]
