"""Stage-level agent architecture."""

from src.core.stage.adapters import StageResultAdapter
from src.core.stage.base_stage_agent import BaseStageAgent, StageAgentAdvisor, StageAgentDecision, StageToolCall
from src.core.stage.models import StageResult, StageTask, StageType, ToolTrace
from src.core.stage.registry import StageAgentRegistry

__all__ = [
    "BaseStageAgent",
    "StageAgentAdvisor",
    "StageAgentDecision",
    "StageAgentRegistry",
    "StageResult",
    "StageResultAdapter",
    "StageTask",
    "StageToolCall",
    "StageType",
    "ToolTrace",
]
