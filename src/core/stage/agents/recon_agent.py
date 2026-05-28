"""Recon stage agent."""

from src.core.stage.base_stage_agent import BaseStageAgent
from src.core.stage.models import StageType


class ReconAgent(BaseStageAgent):
    stage_type = StageType.RECON_STAGE
    agent_name = "recon_agent"
    tool_categories = frozenset({"nmap", "scan", "probe", "discover", "dns", "tls", "fingerprint"})


__all__ = ["ReconAgent"]
