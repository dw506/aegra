"""Optional agent-pipeline assembly helpers.

中文注释：
这里专门放“装配层”逻辑，不把第三方网关配置、advisor 选择之类的细节
塞回 `AgentPipeline` 本体。默认 pipeline 行为保持不变，只有显式调用这些
builder 时才会启用可选能力。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_pipeline import AgentPipeline
from src.core.agents.agent_protocol import BaseAgent
from src.core.agents.critic import CriticAgent, CriticLLMAdvisor
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig
from src.core.agents.packy_critic_advisor import PackyCriticAdvisor
from src.core.agents.packy_planner_advisor import PackyPlannerAdvisor
from src.core.agents.packy_supervisor_advisor import PackySupervisorAdvisor
from src.core.agents.planner import PlannerAgent, PlannerLLMAdvisor
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.supervisor import SupervisorAgent, SupervisorLLMAdvisor
from src.core.agents.task_builder import TaskBuilderAgent


class AgentPipelineAssemblyOptions(BaseModel):
    """Configuration for optional pipeline assembly."""

    enable_packy_planner_advisor: bool = False
    enable_packy_critic_advisor: bool = False
    enable_packy_supervisor_advisor: bool = False
    include_task_builder: bool = True
    include_scheduler: bool = True
    include_critic: bool = True
    include_supervisor: bool = False
    extra_agents: list[BaseAgent] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)


def build_optional_agent_pipeline(
    *,
    options: AgentPipelineAssemblyOptions | None = None,
    planner_llm_advisor: PlannerLLMAdvisor | None = None,
    critic_llm_advisor: CriticLLMAdvisor | None = None,
    supervisor_llm_advisor: SupervisorLLMAdvisor | None = None,
    llm_client_config: PackyLLMConfig | None = None,
    event_sink: Callable[[list[dict[str, Any]]], None] | None = None,
    state_delta_sink: Callable[[list[dict[str, Any]]], None] | None = None,
) -> AgentPipeline:
    """Build a standard agent pipeline with optional Packy planner advice.

    中文注释：
    - 默认情况下，这个 builder 产出的 planner 与原逻辑一致，不挂 LLM advisor。
    - 只有显式传入 `enable_packy_planner_advisor=True`，或者直接传入
      `planner_llm_advisor`，才会启用可选的规划建议层。
    """

    resolved_options = options or AgentPipelineAssemblyOptions()
    resolved_advisor = planner_llm_advisor
    if resolved_advisor is None and resolved_options.enable_packy_planner_advisor:
        resolved_advisor = (
            PackyPlannerAdvisor(client=PackyLLMClient(llm_client_config))
            if llm_client_config is not None
            else PackyPlannerAdvisor.from_env()
        )
    resolved_critic_advisor = critic_llm_advisor
    if resolved_critic_advisor is None and resolved_options.enable_packy_critic_advisor:
        resolved_critic_advisor = (
            PackyCriticAdvisor(client=PackyLLMClient(llm_client_config))
            if llm_client_config is not None
            else PackyCriticAdvisor.from_env()
        )
    resolved_supervisor_advisor = supervisor_llm_advisor
    if resolved_supervisor_advisor is None and resolved_options.enable_packy_supervisor_advisor:
        resolved_supervisor_advisor = (
            PackySupervisorAdvisor(client=PackyLLMClient(llm_client_config))
            if llm_client_config is not None
            else PackySupervisorAdvisor.from_env()
        )

    agents: list[BaseAgent] = [PlannerAgent(llm_advisor=resolved_advisor)]
    if resolved_options.include_task_builder:
        agents.append(TaskBuilderAgent())
    if resolved_options.include_scheduler:
        agents.append(SchedulerAgent())
    if resolved_options.include_critic:
        agents.append(CriticAgent(llm_advisor=resolved_critic_advisor))
    if (
        resolved_options.include_supervisor
        or resolved_options.enable_packy_supervisor_advisor
        or resolved_supervisor_advisor is not None
    ):
        agents.append(SupervisorAgent(llm_advisor=resolved_supervisor_advisor))
    agents.extend(resolved_options.extra_agents)

    return AgentPipeline(
        agents=agents,
        event_sink=event_sink,
        state_delta_sink=state_delta_sink,
    )


__all__ = [
    "AgentPipelineAssemblyOptions",
    "build_optional_agent_pipeline",
]
