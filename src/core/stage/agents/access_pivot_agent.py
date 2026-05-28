"""Access and pivot stage agent."""

from src.core.stage.base_stage_agent import BaseStageAgent
from src.core.stage.models import StageType


class AccessPivotAgent(BaseStageAgent):
    stage_type = StageType.ACCESS_PIVOT_STAGE
    agent_name = "access_pivot_agent"
    tool_categories = frozenset({"credential", "session", "identity", "privilege", "pivot", "internal"})


__all__ = ["AccessPivotAgent"]
