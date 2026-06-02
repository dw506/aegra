from __future__ import annotations

import json

from src.core.agents.packy_llm import PackyLLMResponse
from src.core.stage.registry import StageAgentRegistry


class FakePackyClient:
    def __init__(self) -> None:
        self.config = type("Config", (), {"model": "gpt-test"})()
        self.calls: list[dict[str, object]] = []

    def complete_chat(self, **kwargs):
        self.calls.append(kwargs)
        return PackyLLMResponse(
            model="gpt-test",
            text=json.dumps({"action": "finish", "rationale": "done", "finish": {"status": "succeeded", "summary": "done"}}),
        )


def test_default_registry_gives_each_stage_agent_its_own_advisor() -> None:
    registry = StageAgentRegistry.default(llm_client=FakePackyClient())

    agents = [
        registry.resolve("RECON_STAGE"),
        registry.resolve("VULN_ANALYSIS_STAGE"),
        registry.resolve("EXPLOIT_STAGE"),
        registry.resolve("ACCESS_PIVOT_STAGE"),
        registry.resolve("GOAL_STAGE"),
    ]

    advisors = [agent._advisor for agent in agents]  # noqa: SLF001
    assert len({id(advisor) for advisor in advisors}) == 5
    assert [getattr(advisor, "agent_name", None) for advisor in advisors] == [
        "recon_agent",
        "vuln_analysis_agent",
        "exploit_validation_agent",
        "access_pivot_agent",
        "goal_agent",
    ]
