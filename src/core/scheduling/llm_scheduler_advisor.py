"""LLM advisor for SchedulerAgent task selection."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.agents.packy_llm import PackyLLMClient, PackyLLMError
from src.core.scheduling.llm_scheduler_models import ScheduleDecision


SYSTEM_PROMPT = """You are Aegra's SchedulerAgent.

You own only task scheduling decisions for existing Task Graph tasks.
You do not perform global planning.
You choose which existing TG task should run now.
You must consider Runtime locks, worker state, sessions, credentials, budget, policy, tool catalog, and recent failures.
You may choose dispatch, defer, retry, wait, blocked, or stop.
You must not invent tasks.
You must not execute tools.
You must not write KG, AG, TG, or Runtime directly.
Code only assembles context and validates schema.
Return strict JSON only."""


class LLMSchedulerAdvisorConfig(BaseModel):
    """Prompt limits and model options for LLM scheduling."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_context_chars: int = Field(default=24000, ge=4000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMSchedulerAdvisor:
    """Ask an LLM to select the next task from deterministic candidates."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: LLMSchedulerAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or LLMSchedulerAdvisorConfig(model=getattr(client.config, "model", None))

    def choose_next_task(
        self,
        *,
        operation_id: str,
        graph_context: dict[str, Any],
        candidate_tasks: list[dict[str, Any]],
        runtime_summary: dict[str, Any],
        policy_context: dict[str, Any],
        tool_catalog: dict[str, Any],
        recent_outcomes: list[dict[str, Any]],
    ) -> ScheduleDecision:
        prompt = self._build_prompt(
            operation_id=operation_id,
            graph_context=graph_context,
            candidate_tasks=candidate_tasks,
            runtime_summary=runtime_summary,
            policy_context=policy_context,
            tool_catalog=tool_catalog,
            recent_outcomes=recent_outcomes,
        )
        try:
            response = self._client.complete_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                model=self._config.model,
                temperature=self._config.temperature,
            )
        except PackyLLMError as exc:
            return ScheduleDecision(
                decision="blocked",
                task_id=None,
                worker_id=None,
                rationale="scheduler_llm_unavailable",
                metadata={"accepted": False, "reason": "scheduler_llm_unavailable", "error": str(exc)},
            )

        payload = _extract_json_object(response.text)
        if payload is None:
            return ScheduleDecision(
                decision="blocked",
                rationale="invalid_scheduler_json",
                metadata={"accepted": False, "reason": "invalid_scheduler_json", "raw_text": response.text[:1000]},
            )
        payload.setdefault("metadata", {})
        payload = _normalize_schedule_payload(payload)
        payload["metadata"] = {
            **dict(payload["metadata"]),
            "accepted": True,
            "scheduler": "llm_scheduler",
            "model": response.model,
            "usage": response.usage,
        }
        try:
            return ScheduleDecision.model_validate(payload)
        except ValidationError as exc:
            return ScheduleDecision(
                decision="blocked",
                rationale="invalid_scheduler_schema",
                metadata={
                    "accepted": False,
                    "reason": "invalid_scheduler_schema",
                    "error": str(exc),
                    "raw_text": response.text[:2000],
                    "payload": payload,
                },
            )

    def _build_prompt(
        self,
        *,
        operation_id: str,
        graph_context: dict[str, Any],
        candidate_tasks: list[dict[str, Any]],
        runtime_summary: dict[str, Any],
        policy_context: dict[str, Any],
        tool_catalog: dict[str, Any],
        recent_outcomes: list[dict[str, Any]],
    ) -> str:
        payload = {
            "operation_id": operation_id,
            "graph_context": graph_context,
            "candidate_tasks": candidate_tasks,
            "runtime_summary": runtime_summary,
            "policy_context": policy_context,
            "tool_catalog": tool_catalog,
            "recent_outcomes": recent_outcomes,
            "response_schema": {
                "decision": "dispatch | defer | retry | wait | blocked | stop",
                "task_id": "candidate task_id or null",
                "worker_id": "selected worker id or null",
                "rationale": "short reason",
                "confidence": 0.0,
                "scheduled_task": {
                    "task_id": "same as task_id",
                    "stage_type": "candidate stage_type",
                    "objective": "candidate objective",
                    "target_refs": [],
                    "known_facts": [],
                    "constraints": [],
                    "allowed_tools": [],
                    "success_criteria": [],
                    "policy_context": {},
                    "runtime_context": {},
                },
                "runtime_update_intents": [],
                "metadata": {"accepted": True},
            },
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if len(text) > self._config.max_context_chars:
            payload["graph_context"] = {
                "operation_id": graph_context.get("operation_id"),
                "target_refs": graph_context.get("target_refs", []),
                "context_truncated": True,
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        return (
            "Return one strict JSON object matching response_schema. "
            "If no candidate should run, use wait, blocked, defer, retry, or stop. "
            "Do not include markdown or prose outside JSON.\n\n"
            f"{text}"
        )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        block = _find_first_json_object(stripped)
        if block is None:
            return None
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _normalize_schedule_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if normalized.get("decision") != "dispatch":
        normalized["scheduled_task"] = None
        return normalized
    scheduled_task = normalized.get("scheduled_task")
    if isinstance(scheduled_task, dict):
        task = dict(scheduled_task)
        task["known_facts"] = [
            item if isinstance(item, dict) else {"summary": str(item)}
            for item in task.get("known_facts", [])
            if item is not None
        ]
        task["target_refs"] = [dict(item) for item in task.get("target_refs", []) if isinstance(item, dict)]
        task["constraints"] = [str(item) for item in task.get("constraints", []) if item is not None]
        task["allowed_tools"] = [str(item) for item in task.get("allowed_tools", []) if item is not None]
        task["success_criteria"] = [
            str(item) for item in task.get("success_criteria", []) if item is not None
        ]
        task["policy_context"] = dict(task.get("policy_context") or {})
        task["runtime_context"] = dict(task.get("runtime_context") or {})
        normalized["scheduled_task"] = task
    return normalized


def _find_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


__all__ = ["LLMSchedulerAdvisor", "LLMSchedulerAdvisorConfig"]
