"""Access and pivot stage agent."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.execution.mcp_client import MCPClient
from src.core.stage.advisors.access_pivot_advisor import AccessPivotAdvisor
from src.core.stage.base_stage_agent import BaseStageAgent


class AccessPivotAgent(BaseStageAgent):
    stage_type = "ACCESS_PIVOT_STAGE"
    agent_name = "access_pivot_agent"
    allowed_tool_names = frozenset(
        {
            "credential_check",
            "session_probe",
            "session_open_lab",
            "identity_context_probe",
            "privilege_context_probe",
            "pivot_route_probe",
            "internal_service_discover",
            "tcp_connect_probe",
            "http_probe",
        }
    )
    denied_tool_names = frozenset({"safe_vuln_validate", "nuclei_scan", "run_command"})
    tool_categories = frozenset({"credential", "session", "identity", "privilege", "pivot", "internal"})

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
