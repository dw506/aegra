"""Goal verification stage agent."""

from src.core.stage.base_stage_agent import BaseStageAgent
from src.core.stage.models import StageType


class GoalAgent(BaseStageAgent):
    stage_type = StageType.GOAL_STAGE
    agent_name = "goal_agent"
    tool_categories = frozenset({"goal", "check", "cleanup", "artifact"})


__all__ = ["GoalAgent"]
