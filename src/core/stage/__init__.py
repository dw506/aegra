"""Stage-level agent architecture."""

from src.core.stage.adapters import StageResultAdapter
from src.core.stage.base_stage_agent import BaseStageAgent, StageAgentAdvisor, StageAgentDecision, StageToolCall
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent
from src.core.stage.llm_stage_advisor import LLMStageAdvisor, LLMStageAdvisorConfig
from src.core.stage.models import (
    GraphStateSnapshot,
    GraphUpdateIntent,
    StageName,
    StageExecutionRequest,
    StageHandoffSuggestion,
    StageResult,
    ToolTrace,
    normalize_stage_name,
)
from src.core.stage.registry import StageAgentRegistry
from src.core.stage.dispatcher import StageDispatcher

__all__ = [
    "BaseStageAgent",
    "GraphStateSnapshot",
    "GraphUpdateIntent",
    "LLMStageAdvisor",
    "LLMStageAdvisorConfig",
    "LLMDrivenStageAgent",
    "StageAgentAdvisor",
    "StageAgentDecision",
    "StageAgentRegistry",
    "StageDispatcher",
    "StageExecutionRequest",
    "StageHandoffSuggestion",
    "StageName",
    "StageResult",
    "StageResultAdapter",
    "StageToolCall",
    "ToolTrace",
    "normalize_stage_name",
]
