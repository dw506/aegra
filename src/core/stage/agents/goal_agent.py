"""Goal verification stage agent."""

from src.core.stage.base_stage_agent import BaseStageAgent


class GoalAgent(BaseStageAgent):
    stage_type = "GOAL_STAGE"
    agent_name = "goal_agent"
    tool_categories = frozenset({"goal", "check", "cleanup", "artifact"})


__all__ = ["GoalAgent"]
