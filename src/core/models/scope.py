"""Workspace, engagement and target scope models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AssetKind = Literal["host", "domain", "cidr", "url", "service"]
ScopeAction = Literal["allow", "deny"]

#“授权目标资产”的数据结构
class Asset(BaseModel):
    """One scoped target asset."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str | None = None
    kind: AssetKind = "host"
    value: str = Field(min_length=1)
    address: str | None = None
    hostname: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    protocol: str | None = None
    url: str | None = None
    platform: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def normalized_value(self) -> str:
        return (self.value or self.address or self.url or self.hostname or "").strip()


class ScopeRule(BaseModel):
    """Allow or deny rule for a scoped target pattern."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    action: ScopeAction = "allow"
    kind: AssetKind | Literal["any"] = "any"
    value: str = Field(min_length=1)
    reason: str | None = None


class DenylistRule(BaseModel):
    """Explicit denylist entry."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    kind: AssetKind | Literal["any"] = "any"
    value: str = Field(min_length=1)
    reason: str = "denylist"


class ScanWindow(BaseModel):
    """Optional time window in which scheduling is allowed."""

    model_config = ConfigDict(extra="forbid")

    window_id: str = Field(min_length=1)
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    timezone: str = "UTC"
    days_of_week: list[int] = Field(default_factory=list)


class RateLimitPolicy(BaseModel):
    """Rate and concurrency limits for scoped execution."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_per_host: int = Field(default=1, ge=1)
    max_tasks_per_minute: int | None = Field(default=None, ge=1)
    max_cidr_prefix_v4: int = Field(default=24, ge=0, le=32)
    max_cidr_prefix_v6: int = Field(default=64, ge=0, le=128)


class RiskPolicy(BaseModel):
    """Default operation risk policy.

    Real-penetration posture (design §4.3): in the authorized scope the framework
    blocks no attack action by hardcoded default. Every ``block_*`` flag is False
    and no action requires approval; the blocking mechanism stays available for a
    profile that explicitly opts back in (e.g. ``block_active_exploit=True``).
    """

    model_config = ConfigDict(extra="forbid")

    max_risk_level: Literal["low", "medium", "high"] = "high"
    block_active_exploit: bool = False
    block_destructive: bool = False
    block_command_execution: bool = False
    block_file_write: bool = False
    block_reverse_callback: bool = False
    require_approval_for_active_exploit: bool = False
    approval_required_tags: list[str] = Field(default_factory=list)


class Engagement(BaseModel):
    """An engagement groups assets and scope rules for one operation."""

    model_config = ConfigDict(extra="forbid")

    engagement_id: str = Field(min_length=1)
    name: str | None = None
    assets: list[Asset] = Field(default_factory=list)
    scope_rules: list[ScopeRule] = Field(default_factory=list)
    denylist: list[DenylistRule] = Field(default_factory=list)
    scan_windows: list[ScanWindow] = Field(default_factory=list)
    rate_limit: RateLimitPolicy = Field(default_factory=RateLimitPolicy)
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Workspace(BaseModel):
    """Top-level workspace security context."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1)
    name: str | None = None
    engagements: list[Engagement] = Field(default_factory=list)
    default_engagement_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def active_engagement(self) -> Engagement | None:
        if self.default_engagement_id:
            for engagement in self.engagements:
                if engagement.engagement_id == self.default_engagement_id:
                    return engagement
        return self.engagements[0] if self.engagements else None


__all__ = [
    "Asset",
    "AssetKind",
    "DenylistRule",
    "Engagement",
    "RateLimitPolicy",
    "RiskPolicy",
    "ScanWindow",
    "ScopeRule",
    "Workspace",
]
