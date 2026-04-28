"""Packy-backed supervisor advisor."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.llm_decision import contains_forbidden_llm_decision_key
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig, PackyLLMError
from src.core.agents.supervisor import SupervisorContext, SupervisorDecision


class PackySupervisorAdvisorConfig(BaseModel):
    """Configuration for the Packy-backed supervisor advisor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    system_prompt: str = (
        "你是 Aegra 的高层策略建议助手。"
        "你只能从允许的 supervisor strategy 中选择一个结构化建议，不能生成工具命令、任务、参数或图写入。"
        "请只返回 JSON，不要输出额外说明。"
    )


class PackySupervisorAdvisor:
    """Supervisor advisor backed by `PackyLLMClient`."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: PackySupervisorAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or PackySupervisorAdvisorConfig()

    @classmethod
    def from_env(
        cls,
        *,
        config: PackySupervisorAdvisorConfig | None = None,
        client_config: PackyLLMConfig | None = None,
    ) -> "PackySupervisorAdvisor":
        """Create an advisor from environment-driven client config."""

        return cls(
            client=PackyLLMClient(client_config or PackyLLMConfig.from_env()),
            config=config,
        )

    def advise(self, *, context: SupervisorContext) -> SupervisorDecision | None:
        user_prompt = self._build_user_prompt(context)
        try:
            response = self._client.complete_chat(
                user_prompt=user_prompt,
                system_prompt=self._config.system_prompt,
                model=self._config.model,
                temperature=0.0,
            )
        except PackyLLMError:
            return None
        return self._parse_decision_text(response.text)

    @staticmethod
    def _build_user_prompt(context: SupervisorContext) -> str:
        payload = {
            "runtime_summary": context.runtime_summary,
            "last_control_cycle": context.last_control_cycle,
            "planner_summary": context.planner_summary,
            "critic_summary": context.critic_summary,
            "budget_summary": context.budget_summary,
            "allowed_strategies": [
                "continue_planning",
                "continue_execution",
                "request_replan",
                "pause_for_review",
                "stop_when_quiescent",
            ],
            "response_schema": {
                "strategy": "必须是 allowed_strategies 之一",
                "rationale": "简短策略说明",
                "confidence": "[0, 1] 之间的浮点数",
                "requires_human_review": "布尔值",
                "metadata": {"reason": "可选建议原因"},
            },
        }
        return (
            "请基于下面的 operation 摘要返回 supervisor strategy JSON。"
            "不要生成任务、命令、工具参数、图 patch 或 cancel/replace 动作。\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    @classmethod
    def _parse_decision_text(cls, text: str) -> SupervisorDecision | None:
        payload = cls._extract_json_payload(text)
        if not isinstance(payload, dict):
            return None
        if contains_forbidden_llm_decision_key(payload) is not None:
            return None
        try:
            return SupervisorDecision.model_validate(payload)
        except Exception:
            return None

    @staticmethod
    def _extract_json_payload(text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None


__all__ = ["PackySupervisorAdvisor", "PackySupervisorAdvisorConfig"]
