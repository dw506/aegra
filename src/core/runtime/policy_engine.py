"""Platform scope and safety policy engine."""

from __future__ import annotations

from datetime import datetime
from ipaddress import ip_address, ip_network
from typing import Any
from urllib.parse import urlparse

from src.core.models.runtime import RuntimeState, utc_now
from src.core.models.scope import Asset, DenylistRule, Engagement, ScopeRule, Workspace
from src.core.runtime.observability import append_audit_log
from src.core.runtime.policy import PolicyDecision, RuntimePolicy, policy_from_runtime_state


class PolicyEngine:
    """Evaluate target scope, task, validator and tool policies."""

    SAFE_TAGS = {"safe_probe", "fingerprint", "service_fingerprint", "http-fingerprint"}
    ACTIVE_TAGS = {"active_exploit", "exploit", "destructive", "command_execution", "file_write", "reverse_callback"}

    def __init__(self, policy: RuntimePolicy | None = None) -> None:
        self._policy = policy

    def evaluate_target_scope(
        self,
        target: str | dict[str, Any] | Asset,
        runtime_state: RuntimeState | None = None,
    ) -> PolicyDecision:
        policy = self._resolve_policy(runtime_state)
        target_asset = self._asset_from_target(target)
        target_value = target_asset.normalized_value
        engagement = self._engagement(policy)

        for rule in self._denylist(policy, engagement):
            if self._rule_matches(rule.kind, rule.value, target_asset):
                return PolicyDecision(
                    decision="deny",
                    gate="scope",
                    target=target_value,
                    reason=f"denylist matched: {rule.reason}",
                    matched_rule_id=rule.rule_id,
                )

        allow_rules = [rule for rule in (engagement.scope_rules if engagement else []) if rule.action == "allow"]
        allow_assets = list(engagement.assets if engagement else [])
        legacy_allow_hosts = set(policy.authorized_hosts)
        legacy_cidrs = list(policy.cidr_whitelist)
        strict_scope = bool(allow_rules or allow_assets or legacy_allow_hosts or legacy_cidrs or policy.workspace or policy.engagement)
        if not strict_scope:
            return PolicyDecision(decision="allow", gate="scope", target=target_value, reason="no explicit scope configured")

        if any(self._rule_matches(rule.kind, rule.value, target_asset) for rule in allow_rules):
            return PolicyDecision(decision="allow", gate="scope", target=target_value, reason="allow scope rule matched")
        if any(self._asset_matches(asset, target_asset) for asset in allow_assets):
            return PolicyDecision(decision="allow", gate="scope", target=target_value, reason="allow asset matched")
        if target_value.lower() in {host.lower() for host in legacy_allow_hosts}:
            return PolicyDecision(decision="allow", gate="scope", target=target_value, reason="authorized host matched")
        if self._matches_any_cidr(target_value, legacy_cidrs):
            return PolicyDecision(decision="allow", gate="scope", target=target_value, reason="cidr whitelist matched")
        if self._is_opaque_runtime_id(target_value):
            return PolicyDecision(decision="allow", gate="scope", target=target_value, reason="opaque runtime target id allowed")

        return PolicyDecision(decision="deny", gate="scope", target=target_value, reason="target outside allowlist")

    def evaluate_task_policy(
        self,
        task: Any,
        runtime_state: RuntimeState | None = None,
    ) -> PolicyDecision:
        policy = self._resolve_policy(runtime_state)
        task_id = self._task_id(task)
        for target in self._task_targets(task):
            scope_decision = self.evaluate_target_scope(target, runtime_state)
            if scope_decision.decision != "allow":
                return scope_decision.model_copy(update={"task_id": task_id})

        window_decision = self._scan_window_decision(policy)
        if window_decision.decision != "allow":
            return window_decision.model_copy(update={"task_id": task_id})

        input_bindings = self._task_input_bindings(task)
        tags = {tag.lower() for tag in self._task_tags(task)}
        bindings = {str(key).lower(): str(value).lower() for key, value in input_bindings.items()}
        risk_decision = self._risk_decision(
            policy=policy,
            tags=tags,
            metadata=bindings,
            risk_level=self._task_risk_level(task),
            task_id=task_id,
        )
        if risk_decision.decision != "allow":
            return risk_decision

        if self._task_approval_required(task) or self._task_gate_ids(task):
            approval_id = f"task:{task_id}:approved"
            approved = bool(runtime_state and runtime_state.budgets.approval_cache.get(approval_id))
            if not approved:
                return PolicyDecision(
                    decision="requires_approval",
                    gate="approval",
                    task_id=task_id,
                    approval_id=approval_id,
                    reason="task requires approval",
                )
        return PolicyDecision(decision="allow", gate="task", task_id=task_id, reason="task policy allowed")

    def evaluate_validator_policy(self, validator_id: str, metadata: dict[str, Any] | None = None) -> PolicyDecision:
        metadata = metadata or {}
        tags = {str(item).lower() for item in metadata.get("tags", [])} if isinstance(metadata.get("tags"), list) else set()
        tags.add(str(validator_id).lower())
        if metadata.get("active_exploit"):
            tags.add("active_exploit")
        if str(validator_id).lower() in {"http-fingerprint", "http_fingerprint"}:
            return PolicyDecision(decision="allow", gate="validator", reason="fingerprint validator allowed")
        return self._risk_decision(policy=self._resolve_policy(None), tags=tags, metadata=metadata, risk_level=str(metadata.get("risk_level") or "low"))

    def evaluate_tool_policy(self, tool: dict[str, Any]) -> PolicyDecision:
        policy = self._resolve_policy(None)
        command = tool.get("command")
        command_name = ""
        if isinstance(command, list) and command:
            command_name = str(command[0]).rsplit("\\", 1)[-1].rsplit("/", 1)[-1].lower()
        explicit_name = str(tool.get("name") or tool.get("kind") or command_name).lower()
        disabled_tools = {item.lower() for item in policy.disabled_tools}
        if explicit_name in disabled_tools or command_name in disabled_tools:
            return PolicyDecision(decision="deny", gate="tool", reason=f"{explicit_name or command_name} disabled by policy")
        command_allowlist = {item.lower() for item in policy.command_allowlist}
        if command_allowlist and command_name and command_name not in command_allowlist:
            return PolicyDecision(decision="deny", gate="tool", reason=f"{command_name} outside command allowlist")
        lowered = " ".join(str(item).lower() for item in [tool.get("kind"), tool.get("operation"), *tool.get("tags", [])] if item)
        if not policy.allow_safe_probe and "safe_probe" in lowered:
            return PolicyDecision(decision="deny", gate="tool", reason="safe probe tools disabled by policy")
        if not policy.allow_fingerprint and "fingerprint" in lowered:
            return PolicyDecision(decision="deny", gate="tool", reason="fingerprint tools disabled by policy")
        if "safe_probe" in lowered or "fingerprint" in lowered:
            return PolicyDecision(decision="allow", gate="tool", reason="safe probe tool allowed")
        # Real-penetration default: active exploit / command execution / file write /
        # reverse callback / destructive actions are NOT blocked by hardcoded fiat.
        # We translate the action markers into risk tags and delegate to the
        # policy-driven risk gate, which defaults to allow (RiskPolicy.block_* are
        # all False by default). A profile that explicitly re-enables a block_* flag
        # still gets enforcement through the same path.
        action_markers = {
            "command execution": "command_execution",
            "command_execution": "command_execution",
            "file write": "file_write",
            "file_write": "file_write",
            "reverse callback": "reverse_callback",
            "reverse_callback": "reverse_callback",
            "active_exploit": "active_exploit",
            "destructive": "destructive",
        }
        tags: set[str] = set()
        for marker, tag in action_markers.items():
            if marker in lowered or bool(tool.get(tag)):
                tags.add(tag)
        if not tags:
            return PolicyDecision(decision="allow", gate="tool", reason="tool policy allowed")
        return self._risk_decision(
            policy=policy,
            tags=tags,
            metadata={},
            risk_level=str(tool.get("risk_level") or "low"),
        )

    def audit(self, runtime_state: RuntimeState, decision: PolicyDecision) -> None:
        append_audit_log(
            runtime_state,
            {
                "event_type": "policy_decision",
                "decision": decision.model_dump(mode="json"),
            },
        )

    def _risk_decision(
        self,
        *,
        policy: RuntimePolicy,
        tags: set[str],
        metadata: dict[str, Any],
        risk_level: str,
        task_id: str | None = None,
    ) -> PolicyDecision:
        risk_policy = policy.risk_policy
        if tags & self.SAFE_TAGS:
            return PolicyDecision(decision="allow", gate="risk", task_id=task_id, reason="safe probe/fingerprint allowed")
        flags = {
            "active_exploit": risk_policy.block_active_exploit,
            "destructive": risk_policy.block_destructive,
            "command_execution": risk_policy.block_command_execution,
            "file_write": risk_policy.block_file_write,
            "reverse_callback": risk_policy.block_reverse_callback,
        }
        for flag, blocked in flags.items():
            if not blocked:
                continue
            if flag in tags or str(metadata.get(flag, "")).lower() == "true":
                if flag == "active_exploit" and risk_policy.require_approval_for_active_exploit:
                    return PolicyDecision(
                        decision="requires_approval",
                        gate="risk",
                        task_id=task_id,
                        approval_id=f"task:{task_id}:approved" if task_id else None,
                        reason="active_exploit requires approval",
                    )
                return PolicyDecision(decision="deny", gate="risk", task_id=task_id, reason=f"{flag} blocked by default")
        order = {"low": 0, "medium": 1, "high": 2}
        if order.get(risk_level, 0) > order.get(risk_policy.max_risk_level, 0):
            return PolicyDecision(decision="requires_approval", gate="risk", task_id=task_id, reason="risk level exceeds policy")
        return PolicyDecision(decision="allow", gate="risk", task_id=task_id, reason="risk policy allowed")

    def _resolve_policy(self, runtime_state: RuntimeState | None) -> RuntimePolicy:
        if self._policy is not None:
            return self._policy
        if runtime_state is not None:
            return policy_from_runtime_state(runtime_state)
        return RuntimePolicy()

    @staticmethod
    def _engagement(policy: RuntimePolicy) -> Engagement | None:
        if policy.engagement is not None:
            return policy.engagement
        if policy.workspace is not None:
            return policy.workspace.active_engagement()
        return None

    @staticmethod
    def _denylist(policy: RuntimePolicy, engagement: Engagement | None) -> list[DenylistRule]:
        return [*policy.denylist, *((engagement.denylist if engagement else []))]

    @staticmethod
    def _scan_window_decision(policy: RuntimePolicy) -> PolicyDecision:
        windows = list(policy.scan_windows)
        engagement = PolicyEngine._engagement(policy)
        if engagement:
            windows.extend(engagement.scan_windows)
        if not windows:
            return PolicyDecision(decision="allow", gate="scan_window", reason="no scan window configured")
        now = utc_now()
        for window in windows:
            if window.starts_at and now < window.starts_at:
                continue
            if window.ends_at and now > window.ends_at:
                continue
            if window.days_of_week and now.weekday() not in window.days_of_week:
                continue
            return PolicyDecision(decision="allow", gate="scan_window", reason=f"scan window {window.window_id} open")
        return PolicyDecision(decision="deny", gate="scan_window", reason="outside allowed scan window")

    @staticmethod
    def _asset_from_target(target: str | dict[str, Any] | Asset) -> Asset:
        if isinstance(target, Asset):
            return target
        if isinstance(target, dict):
            payload = dict(target)
            if "value" not in payload:
                payload["value"] = payload.get("address") or payload.get("url") or payload.get("hostname") or payload.get("service_id")
            payload.setdefault("kind", PolicyEngine._infer_kind(str(payload.get("value", ""))))
            return Asset.model_validate(payload)
        return Asset(kind=PolicyEngine._infer_kind(target), value=target)

    @staticmethod
    def _infer_kind(value: str) -> str:
        lowered = value.lower()
        if lowered.startswith(("http://", "https://")):
            return "url"
        if "/" in lowered:
            return "cidr"
        if ":" in lowered and "/" not in lowered:
            return "service"
        try:
            ip_address(value)
            return "host"
        except ValueError:
            return "domain" if any(ch.isalpha() for ch in value) else "host"

    @staticmethod
    def _rule_matches(kind: str, value: str, target: Asset) -> bool:
        if kind != "any" and kind != target.kind:
            if not (kind == "host" and target.kind in {"service", "url"}):
                return False
        return PolicyEngine._value_matches(value, target)

    @staticmethod
    def _asset_matches(allowed: Asset, target: Asset) -> bool:
        if allowed.kind != target.kind and not (allowed.kind == "host" and target.kind in {"service", "url"}):
            return False
        return PolicyEngine._value_matches(allowed.normalized_value, target)

    @staticmethod
    def _value_matches(pattern: str, target: Asset) -> bool:
        target_values = {
            target.normalized_value.lower(),
            str(target.address or "").lower(),
            str(target.hostname or "").lower(),
            str(target.url or "").lower(),
        }
        parsed_host = urlparse(target.normalized_value).hostname
        if parsed_host:
            target_values.add(parsed_host.lower())
        if ":" in target.normalized_value and target.kind == "service":
            target_values.add(target.normalized_value.split(":", 1)[0].lower())
        pattern_lower = pattern.lower()
        if pattern_lower in target_values:
            return True
        return PolicyEngine._matches_any_cidr(next(iter(target_values)), [pattern])

    @staticmethod
    def _matches_any_cidr(value: str, cidrs: list[str]) -> bool:
        host = urlparse(value).hostname or value.split(":", 1)[0]
        try:
            address = ip_address(host)
        except ValueError:
            return False
        for cidr in cidrs:
            try:
                if address in ip_network(cidr, strict=False):
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _task_targets(task: Any) -> list[str | dict[str, Any]]:
        targets: list[str | dict[str, Any]] = []
        input_bindings = PolicyEngine._task_input_bindings(task)
        for key in ("target_url", "url", "service_id", "target_address", "address", "host_id", "target_host_id"):
            value = input_bindings.get(key)
            if isinstance(value, str) and value.strip():
                targets.append(value.strip())
        for ref in PolicyEngine._task_target_refs(task):
            if ref.ref_type in {"Host", "Service"}:
                targets.append(ref.ref_id)
        return targets or [PolicyEngine._task_id(task)]

    @staticmethod
    def _task_risk_level(task: Any) -> str:
        input_bindings = PolicyEngine._task_input_bindings(task)
        if "risk_level" in input_bindings:
            return str(input_bindings["risk_level"]).lower()
        estimated_risk = PolicyEngine._task_float(task, "estimated_risk")
        if estimated_risk >= 0.7:
            return "high"
        if estimated_risk >= 0.3:
            return "medium"
        return "low"

    @staticmethod
    def _task_id(task: Any) -> str:
        if isinstance(task, dict):
            return str(task.get("id") or task.get("task_id") or "task")
        return str(getattr(task, "id", None) or getattr(task, "task_id", None) or "task")

    @staticmethod
    def _task_input_bindings(task: Any) -> dict[str, Any]:
        value = task.get("input_bindings") if isinstance(task, dict) else getattr(task, "input_bindings", None)
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _task_tags(task: Any) -> list[str]:
        value = task.get("tags") if isinstance(task, dict) else getattr(task, "tags", None)
        return [str(item) for item in value] if isinstance(value, (list, set, tuple)) else []

    @staticmethod
    def _task_target_refs(task: Any) -> list[Any]:
        value = task.get("target_refs") if isinstance(task, dict) else getattr(task, "target_refs", None)
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _task_approval_required(task: Any) -> bool:
        value = task.get("approval_required") if isinstance(task, dict) else getattr(task, "approval_required", False)
        return bool(value)

    @staticmethod
    def _task_gate_ids(task: Any) -> list[str]:
        value = task.get("gate_ids") if isinstance(task, dict) else getattr(task, "gate_ids", None)
        return [str(item) for item in value] if isinstance(value, (list, set, tuple)) else []

    @staticmethod
    def _task_float(task: Any, key: str) -> float:
        value = task.get(key) if isinstance(task, dict) else getattr(task, key, 0.0)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_opaque_runtime_id(value: str) -> bool:
        lowered = value.lower()
        if lowered.startswith(("host-", "svc-", "service-", "asset-", "kg-host::", "kg-service::")):
            return True
        try:
            ip_address(value)
            return False
        except ValueError:
            return "/" not in value and "." not in value and "://" not in value


__all__ = ["PolicyEngine"]
