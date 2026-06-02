"""Dedicated AccessPivotAgent advisor."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.stage.advisors._llm_stage_agent_advisor import DedicatedLLMStageAdvisor
from src.core.stage.context.access_pivot_context import build_access_pivot_context


class AccessPivotAdvisor(DedicatedLLMStageAdvisor):
    def __init__(self, *, client: PackyLLMClient) -> None:
        super().__init__(
            agent_name="access_pivot_agent",
            stage_type="ACCESS_PIVOT_STAGE",
            client=client,
            context_builder=build_access_pivot_context,
        )


__all__ = ["AccessPivotAdvisor"]
