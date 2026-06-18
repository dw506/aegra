"""Data models for profile-driven evaluation.

All models are environment-agnostic. Environment-specific values (CIDRs,
hostnames, flags, tokens) live only in YAML config files and environment
variables, never in this module.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ZoneBinding(BaseModel):
    """Generic zone description loaded from a profile."""

    model_config = ConfigDict(extra="allow")

    name: str
    cidrs: list[str] = Field(default_factory=list)
    directly_reachable: bool = True
    requires: list[str] = Field(default_factory=list)


class OperationProfile(BaseModel):
    """Generic operation profile loaded from profile.yml.

    Contains no hardcoded environment values. All environment-specific
    configuration comes from the YAML file referenced at load time.
    """

    model_config = ConfigDict(extra="allow")

    profile_id: str
    mode: str = "generic"
    environment_id: str | None = None
    entry_scope: dict[str, Any] = Field(default_factory=dict)
    zone_bindings: dict[str, ZoneBinding] = Field(default_factory=dict)
    authorization: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    success_contract_ref: str | None = None
    goal_oracle_ref: str | None = None
    mission: dict[str, Any] = Field(default_factory=dict)
    automation_defaults: dict[str, Any] = Field(default_factory=dict)

    def resolve_zone(self, zone_ref: str) -> ZoneBinding | None:
        """Return the ZoneBinding for a logical zone reference."""
        return self.zone_bindings.get(zone_ref)

    def is_zone_directly_reachable(self, zone_ref: str) -> bool:
        zone = self.resolve_zone(zone_ref)
        return zone.directly_reachable if zone else False

    def zone_requires(self, zone_ref: str) -> list[str]:
        zone = self.resolve_zone(zone_ref)
        return zone.requires if zone else []


class ConditionBinding(BaseModel):
    """One condition binding from a success contract."""

    model_config = ConfigDict(extra="allow")

    predicate: str
    args: dict[str, Any] = Field(default_factory=dict)


class SuccessContract(BaseModel):
    """Generic success contract loaded from success_contract.yml.

    Conditions are evaluated by PredicateEngine against KG/AG/Runtime.
    No environment-specific values here.
    """

    model_config = ConfigDict(extra="allow")

    contract_id: str
    mode: str = "generic"
    require_all: list[str] = Field(default_factory=list)
    require_chain: list[list[str]] = Field(default_factory=list)
    levels: dict[str, list[str]] = Field(default_factory=dict)
    target_level: str | None = None
    condition_bindings: dict[str, ConditionBinding] = Field(default_factory=dict)


class ConditionResult(BaseModel):
    """Result of evaluating one success condition."""

    model_config = ConfigDict(extra="forbid")

    condition: str
    satisfied: bool
    predicate: str
    evidence_refs: list[str] = Field(default_factory=list)
    matched_node_ids: list[str] = Field(default_factory=list)
    matched_edge_ids: list[str] = Field(default_factory=list)
    redacted_summary: str = ""
    error: str | None = None


class SuccessConditionProgress(BaseModel):
    """Evaluated progress against a success contract.

    Written to RuntimeState.execution.metadata["success_condition_progress"].
    Never contains private secrets, raw flags, or tokens.
    """

    model_config = ConfigDict(extra="forbid")

    profile_id: str = ""
    contract_id: str = ""
    mode: str = ""
    all_required_satisfied: bool = False
    chain_integrity: bool = False
    goal_proof_valid: bool = False
    eligible_for_stop: bool = False
    achieved_level: str | None = None
    target_level: str | None = None
    level_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    satisfied: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    condition_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    redacted_summary: str = ""
    last_updated_cycle: int = 0


class GoalOracleInput(BaseModel):
    """Input to the GoalOracle for proof validation."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str
    candidate_target: str = ""
    zone_ref: str = ""
    access_path_refs: list[str] = Field(default_factory=list)
    session_refs: list[str] = Field(default_factory=list)
    pivot_route_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    submitted_marker_hash: str = ""
    submitted_token_hash: str = ""


class GoalOracleOutput(BaseModel):
    """Output from the GoalOracle.

    proof_token is an opaque HMAC-signed token. Raw marker/token/flag values
    are never present in this model.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    proof_token: str = ""
    goal_id: str
    redacted_summary: str = ""
    satisfied_conditions: list[str] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    secret_redacted: bool = True


__all__ = [
    "ConditionBinding",
    "ConditionResult",
    "GoalOracleInput",
    "GoalOracleOutput",
    "OperationProfile",
    "SuccessContract",
    "SuccessConditionProgress",
    "ZoneBinding",
]
