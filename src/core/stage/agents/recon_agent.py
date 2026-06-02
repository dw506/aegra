"""Recon stage agent."""

from src.core.stage.base_stage_agent import BaseStageAgent


class ReconAgent(BaseStageAgent):
    stage_type = "RECON_STAGE"
    agent_name = "recon_agent"
    tool_categories = frozenset({"nmap", "scan", "probe", "discover", "dns", "tls", "fingerprint"})


__all__ = ["ReconAgent"]
