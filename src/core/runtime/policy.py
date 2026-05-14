"""Runtime policy schema and compatibility helpers."""

from __future__ import annotations

from datetime import datetime
from ipaddress import ip_network
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.core.models.runtime import RuntimeState, utc_now
from src.core.models.scope import DenylistRule, Engagement, RiskPolicy, ScanWindow, Workspace


PolicyDecisionStatus = Literal["allow", "deny", "requires_approval"]


class PolicyDecision(BaseModel):
    """Normalized decision emitted by platform policy gates."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    decision: PolicyDecisionStatus
    reason: str = Field(min_length=1)
    gate: str = Field(min_length=1)
    target: str | None = None
    task_id: str | None = None
    approval_id: str | None = None
    matched_rule_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimePolicy(BaseModel):
    """轻量运行时策略模型。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    safety_stop: bool = False
    blocked_hosts: list[str] = Field(default_factory=list)
    authorized_hosts: list[str] = Field(default_factory=list)
    cidr_whitelist: list[str] = Field(default_factory=list)
    deny_egress: bool = False
    sensitive_task_types: list[str] = Field(default_factory=list)
    sensitive_tags: list[str] = Field(default_factory=list)
    session_policies: dict[str, Literal["exclusive", "shared_readonly", "shared"]] = Field(default_factory=dict)
    max_concurrent_per_host: dict[str, int] = Field(default_factory=dict)
    rate_limit_per_subnet_per_min: dict[str, int] = Field(default_factory=dict)
    workspace: Workspace | None = None
    engagement: Engagement | None = None
    denylist: list[DenylistRule] = Field(default_factory=list)
    scan_windows: list[ScanWindow] = Field(default_factory=list)
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    allow_safe_probe: bool = True
    allow_fingerprint: bool = True
    disabled_tools: list[str] = Field(default_factory=list)
    command_allowlist: list[str] = Field(default_factory=list)
    retry_backoff_base_sec: int = Field(default=0, ge=0)
    default_task_timeout_sec: int = Field(default=900, ge=1)
    policy_version: str = "v1"
    loaded_from: str = "settings"
    loaded_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "blocked_hosts",
        "authorized_hosts",
        "cidr_whitelist",
        "sensitive_task_types",
        "sensitive_tags",
        "disabled_tools",
        "command_allowlist",
        mode="before",
    )
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str) and not value.strip():
            return []
        if not isinstance(value, list):
            raise TypeError("policy list fields must be arrays")
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

    @field_validator("cidr_whitelist")
    @classmethod
    def _validate_cidr_whitelist(cls, value: list[str]) -> list[str]:
        for item in value:
            ip_network(item, strict=False)
        return value

    @field_validator("session_policies", mode="before")
    @classmethod
    def _normalize_session_policies(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, str) and not value.strip():
            return {}
        if not isinstance(value, dict):
            raise TypeError("session_policies must be an object")
        normalized: dict[str, str] = {}
        for key, item in value.items():
            session_id = str(key).strip()
            if not session_id:
                continue
            normalized[session_id] = str(item).strip()
        return normalized

    @field_validator("max_concurrent_per_host", "rate_limit_per_subnet_per_min", mode="before")
    @classmethod
    def _normalize_int_mapping(cls, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        if isinstance(value, str) and not value.strip():
            return {}
        if not isinstance(value, dict):
            raise TypeError("policy integer mapping fields must be objects")
        normalized: dict[str, int] = {}
        for key, item in value.items():
            mapping_key = str(key).strip()
            if not mapping_key:
                continue
            normalized[mapping_key] = int(item)
        return normalized

    @field_validator("denylist", "scan_windows", mode="before")
    @classmethod
    def _normalize_model_list(cls, value: Any) -> list[Any]:
        if value is None or value == "":
            return []
        if not isinstance(value, list):
            raise TypeError("policy model list fields must be arrays")
        return value

    @field_validator("workspace", "engagement", mode="before")
    @classmethod
    def _empty_mapping_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict) and not value:
            return None
        return value

    @field_validator("max_concurrent_per_host", "rate_limit_per_subnet_per_min")
    @classmethod
    def _validate_positive_int_mapping(cls, value: dict[str, int]) -> dict[str, int]:
        for key, item in value.items():
            if int(item) < 1:
                raise ValueError(f"policy mapping '{key}' must be >= 1")
        return value

    def to_runtime_metadata(self) -> dict[str, Any]:
        """导出到 RuntimeState metadata 的稳定 JSON 结构。"""

        return self.model_dump(mode="json")

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        payload = super().model_dump(*args, **kwargs)
        optional_defaults = {
            "workspace": None,
            "engagement": None,
            "denylist": [],
            "scan_windows": [],
            "risk_policy": RiskPolicy().model_dump(mode=kwargs.get("mode", "python")),
            "allow_safe_probe": True,
            "allow_fingerprint": True,
        }
        for key, default in optional_defaults.items():
            if key not in self.model_fields_set and payload.get(key) == default:
                payload.pop(key, None)
        return payload


def load_runtime_policy_payload(
    *,
    inline_policy: dict[str, Any] | None = None,
    policy_path: Path | None = None,
) -> RuntimePolicy:
    """Load, merge and validate one runtime policy."""

    payload: dict[str, Any] = dict(inline_policy or {})
    loaded_from = "settings"
    if policy_path is not None:
        try:
            import json

            file_payload = json.loads(policy_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"runtime policy file not found: {policy_path}") from exc
        except Exception as exc:
            raise ValueError(f"failed to read runtime policy file '{policy_path}': {exc}") from exc
        if not isinstance(file_payload, dict):
            raise ValueError(f"runtime policy file '{policy_path}' must contain a JSON object")
        payload.update(file_payload)
        loaded_from = str(policy_path)
    payload.setdefault("policy_version", "v1")
    payload["loaded_from"] = loaded_from
    payload["loaded_at"] = utc_now()
    try:
        return RuntimePolicy.model_validate(payload)
    except ValidationError as exc:
        source = str(policy_path) if policy_path is not None else "settings"
        raise ValueError(f"invalid runtime policy from '{source}': {exc}") from exc


def policy_from_runtime_state(runtime_state: RuntimeState) -> RuntimePolicy:
    """Read one normalized runtime policy from RuntimeState metadata."""

    value = runtime_state.execution.metadata.get("runtime_policy", {})
    if isinstance(value, RuntimePolicy):
        return value
    try:
        return RuntimePolicy.model_validate(value if isinstance(value, dict) else {})
    except ValidationError as exc:
        raise ValueError(f"invalid runtime policy in runtime metadata: {exc}") from exc


__all__ = [
    "PolicyDecision",
    "PolicyDecisionStatus",
    "RuntimePolicy",
    "load_runtime_policy_payload",
    "policy_from_runtime_state",
]
