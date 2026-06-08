from __future__ import annotations

import json
from typing import Any

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.stage.base_stage_agent import BaseStageAgent
from src.core.stage.llm_driven_stage_agent import LLMDrivenStageAgent
from src.core.stage.registry import StageAgentRegistry


class FakePackyClient:
    def __init__(self) -> None:
        self.config = type("Config", (), {"model": "gpt-test"})()
        self.calls: list[dict[str, object]] = []

    def complete_chat(self, **kwargs: Any) -> PackyLLMResponse:
        self.calls.append(kwargs)
        return PackyLLMResponse(
            model="gpt-test",
            text=json.dumps({"action": "finish", "status": "succeeded", "summary": "done"}),
        )


def test_default_registry_gives_each_stage_agent_its_own_llm_driven_agent() -> None:
    registry = StageAgentRegistry.default(llm_client=FakePackyClient())

    agents = [
        registry.resolve("RECON_STAGE"),
        registry.resolve("VULN_ANALYSIS_STAGE"),
        registry.resolve("EXPLOIT_STAGE"),
        registry.resolve("ACCESS_PIVOT_STAGE"),
        registry.resolve("GOAL_STAGE"),
    ]

    assert len({id(agent) for agent in agents}) == 5
    assert all(isinstance(agent, LLMDrivenStageAgent) for agent in agents)
    assert all(not isinstance(agent, BaseStageAgent) for agent in agents)
    assert all(not hasattr(agent, "_advisor") for agent in agents)
    assert [agent.agent_name for agent in agents] == [
        "recon_agent",
        "vuln_analysis_agent",
        "exploit_validation_agent",
        "access_pivot_agent",
        "goal_agent",
    ]
