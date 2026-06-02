"""Dedicated ReconAgent advisor."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.stage.advisors._llm_stage_agent_advisor import DedicatedLLMStageAdvisor
from src.core.stage.context.recon_context import build_recon_context


class ReconAdvisor(DedicatedLLMStageAdvisor):
    def __init__(self, *, client: PackyLLMClient) -> None:
        super().__init__(agent_name="recon_agent", stage_type="RECON_STAGE", client=client, context_builder=build_recon_context)


__all__ = ["ReconAdvisor"]
