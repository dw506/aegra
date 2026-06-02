"""Dedicated GoalAgent advisor."""

from src.core.agents.packy_llm import PackyLLMClient
from src.core.stage.advisors._llm_stage_agent_advisor import DedicatedLLMStageAdvisor
from src.core.stage.context.goal_context import build_goal_context


class GoalAdvisor(DedicatedLLMStageAdvisor):
    def __init__(self, *, client: PackyLLMClient) -> None:
        super().__init__(agent_name="goal_agent", stage_type="GOAL_STAGE", client=client, context_builder=build_goal_context)


__all__ = ["GoalAdvisor"]
