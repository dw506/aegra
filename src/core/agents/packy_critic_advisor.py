"""Packy-backed critic advisor.

中文注释：
这一层的职责是把底层 LLM 文本结果，转换成 `CriticAgent` 能消费的
`CriticLLMReview`。它只补充失败归纳和解释，不直接生成取消/替换动作。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.critic import (
    CriticContext,
    CriticFinding,
    CriticLLMReplanProposal,
    CriticLLMReview,
)
from src.core.agents.llm_decision import (
    LLMDecision,
    LLMDecisionSource,
    LLMDecisionStatus,
    LLMDecisionValidator,
)
from src.core.agents.packy_llm import PackyLLMClient, PackyLLMConfig, PackyLLMError
from src.core.models.runtime import RuntimeState


class PackyCriticAdvisorConfig(BaseModel):
    """Configuration for the Packy-backed critic advisor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str | None = None
    max_findings: int = Field(default=8, ge=1, le=20)
    system_prompt: str = (
        "你是 Aegra 的执行复盘助手。"
        "你只能归纳已有 finding 的失败原因、补充摘要和解释增强。"
        "你不能生成新的 finding_id，也不能直接建议执行命令、取消任务或修改图状态。"
        "请只返回 JSON，不要输出额外说明。"
    )


class PackyCriticAdvisor:
    """Critic advisor backed by `PackyLLMClient`."""

    def __init__(
        self,
        *,
        client: PackyLLMClient,
        config: PackyCriticAdvisorConfig | None = None,
    ) -> None:
        self._client = client
        self._config = config or PackyCriticAdvisorConfig()

    @classmethod
    def from_env(
        cls,
        *,
        config: PackyCriticAdvisorConfig | None = None,
        client_config: PackyLLMConfig | None = None,
    ) -> "PackyCriticAdvisor":
        """Create an advisor from environment-driven client config."""

        return cls(
            client=PackyLLMClient(client_config or PackyLLMConfig.from_env()),
            config=config,
        )

    def summarize_findings(
        self,
        *,
        findings: Sequence[CriticFinding],
        context: CriticContext,
        runtime_state: RuntimeState | None,
    ) -> list[CriticLLMReview]:
        if not findings:
            return []

        limited_findings = self._limit_findings(findings)
        allowed_finding_ids = {finding.finding_id for finding in limited_findings}
        user_prompt = self._build_user_prompt(
            findings=limited_findings,
            context=context,
            runtime_state=runtime_state,
        )

        try:
            response = self._client.complete_chat(
                user_prompt=user_prompt,
                system_prompt=self._config.system_prompt,
                model=self._config.model,
                temperature=0.0,
            )
        except PackyLLMError:
            return []

        return self._parse_review_text(response.text, allowed_finding_ids=allowed_finding_ids)

    def _limit_findings(self, findings: Sequence[CriticFinding]) -> list[CriticFinding]:
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            findings,
            key=lambda item: (severity_rank.get(item.severity.lower(), 3), item.finding_id),
        )[: self._config.max_findings]

    def _build_user_prompt(
        self,
        *,
        findings: Sequence[CriticFinding],
        context: CriticContext,
        runtime_state: RuntimeState | None,
    ) -> str:
        prompt_payload = {
            "runtime_summary": context.runtime_summary,
            "critic_context": {
                "failure_threshold": context.failure_threshold,
                "low_value_threshold": context.low_value_threshold,
                "invalidated_ref_keys": sorted(context.invalidated_ref_keys),
            },
            "runtime_snapshot": self._runtime_snapshot(runtime_state),
            "findings": [self._finding_payload(finding) for finding in findings],
            "response_schema": {
                "reviews": [
                    {
                        "finding_id": "finding ID，必须来自输入 findings",
                        "summary_override": "可选，覆盖原 summary 的简短中文摘要",
                        "rationale_suffix": "可选，补在原 rationale 后面的简短中文归纳",
                        "replan_hint": "可选，只能描述重规划方向，不能包含具体执行动作",
                        "failure_summary": "可选，失败原因摘要",
                        "affected_task_ids": ["可选，必须来自 finding 的 TG task refs"],
                        "confidence": "可选，[0,1] 之间的置信度",
                        "requires_human_review": "可选，是否需要人工复核",
                        "metadata": {"category": "例如 dependency_failure / noisy_path / repeated_failure"},
                    }
                ]
            },
        }
        return (
            "请基于下面的 Critic findings 返回 JSON。"
            "只能补充归纳和摘要，不允许发明新的 finding_id。\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _runtime_snapshot(runtime_state: RuntimeState | None) -> dict[str, Any]:
        if runtime_state is None:
            return {}
        failed_task_ids = [
            task_id
            for task_id, task in runtime_state.execution.tasks.items()
            if task.status.value in {"failed", "timed_out"}
        ]
        return {
            "operation_status": runtime_state.operation_status.value,
            "task_count": len(runtime_state.execution.tasks),
            "failed_task_ids": failed_task_ids,
            "pending_event_count": len(runtime_state.pending_events),
        }

    @staticmethod
    def _finding_payload(finding: CriticFinding) -> dict[str, Any]:
        return {
            "finding_id": finding.finding_id,
            "finding_type": finding.finding_type,
            "severity": finding.severity,
            "summary": finding.summary,
            "rationale": finding.rationale,
            "subject_refs": [ref.model_dump(mode="json") for ref in finding.subject_refs],
            "metadata": finding.metadata,
        }

    def _parse_review_text(
        self,
        text: str,
        *,
        allowed_finding_ids: set[str],
    ) -> list[CriticLLMReview]:
        payload = self._extract_json_payload(text)
        if payload is None:
            return []

        if isinstance(payload, dict):
            raw_items = payload.get("reviews")
        elif isinstance(payload, list):
            raw_items = payload
        else:
            return []

        if not isinstance(raw_items, list):
            return []

        reviews: list[CriticLLMReview] = []
        validator = LLMDecisionValidator()
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            raw_validation = validator.validate_no_forbidden_payload(raw_item)
            if not raw_validation.accepted:
                continue
            finding_id = raw_item.get("finding_id")
            if not isinstance(finding_id, str) or finding_id not in allowed_finding_ids:
                continue
            metadata = raw_item.get("metadata")
            summary_override = raw_item.get("summary_override")
            failure_summary = raw_item.get("failure_summary")
            rationale_suffix = raw_item.get("rationale_suffix")
            replan_hint = raw_item.get("replan_hint")
            affected_task_ids = self._coerce_string_list(raw_item.get("affected_task_ids"))
            confidence = self._coerce_confidence(raw_item.get("confidence"))
            requires_human_review = bool(raw_item.get("requires_human_review", False))
            sanitized_payload = {
                "finding_id": finding_id,
                "summary_override": summary_override if isinstance(summary_override, str) else None,
                "rationale_suffix": rationale_suffix if isinstance(rationale_suffix, str) else None,
                "replan_hint": replan_hint if isinstance(replan_hint, str) else None,
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            }
            decision = LLMDecision(
                source=LLMDecisionSource.CRITIC,
                status=LLMDecisionStatus.ACCEPTED,
                decision_type="critic_finding_review",
                target_id=finding_id,
                target_kind="critic_finding",
                summary_override=sanitized_payload["summary_override"],
                rationale_suffix=sanitized_payload["rationale_suffix"],
                replan_hint=sanitized_payload["replan_hint"],
                metadata=sanitized_payload["metadata"],
            )
            validation = validator.validate_critic_decision(
                decision,
                allowed_finding_ids=allowed_finding_ids,
            )
            if not validation.accepted:
                continue
            proposal = None
            if isinstance(replan_hint, str) and replan_hint:
                proposal = CriticLLMReplanProposal(
                    finding_id=finding_id,
                    failure_summary=failure_summary if isinstance(failure_summary, str) else sanitized_payload["summary_override"],
                    replan_hint=replan_hint,
                    affected_task_ids=affected_task_ids,
                    confidence=confidence,
                    requires_human_review=requires_human_review,
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                    decision=LLMDecision(
                        source=LLMDecisionSource.CRITIC,
                        status=LLMDecisionStatus.ACCEPTED,
                        decision_type="critic_replan_proposal",
                        target_id=finding_id,
                        target_kind="critic_finding",
                        summary_override=failure_summary if isinstance(failure_summary, str) else sanitized_payload["summary_override"],
                        replan_hint=replan_hint,
                        metadata={
                            **(dict(metadata) if isinstance(metadata, dict) else {}),
                            "affected_task_ids": affected_task_ids,
                            "confidence": confidence,
                            "requires_human_review": requires_human_review,
                        },
                    ),
                )
            reviews.append(
                CriticLLMReview(
                    finding_id=finding_id,
                    summary_override=sanitized_payload["summary_override"],
                    rationale_suffix=sanitized_payload["rationale_suffix"],
                    replan_hint=sanitized_payload["replan_hint"],
                    metadata=sanitized_payload["metadata"],
                    decision=decision,
                    validation=validation,
                    replan_proposal=proposal,
                )
            )
        return reviews

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [item for item in items if isinstance(item, str) and item.strip()]

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, confidence))

    @classmethod
    def _extract_json_payload(cls, text: str) -> dict[str, Any] | list[Any] | None:
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("```"):
            stripped = cls._strip_code_fences(stripped)
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            block = cls._find_first_json_block(stripped)
            if block is None:
                return None
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, (dict, list)) else None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _find_first_json_block(text: str) -> str | None:
        start = -1
        opening = ""
        for index, char in enumerate(text):
            if char in "{[":
                start = index
                opening = char
                break
        if start < 0:
            return None

        closing = "}" if opening == "{" else "]"
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
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None


__all__ = [
    "PackyCriticAdvisor",
    "PackyCriticAdvisorConfig",
]
