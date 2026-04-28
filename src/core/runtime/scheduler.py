"""Runtime scheduler with phase-two policy controls.

This scheduler consumes TG scheduling fields together with Runtime State and
produces scheduling decisions. It does not perform real task execution or
worker RPC, but it now enforces the second-phase runtime guardrails.
"""

from __future__ import annotations

from datetime import timedelta
from ipaddress import ip_address, ip_network
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.models.runtime import RuntimeState, TaskRuntimeStatus, WorkerStatus, utc_now
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskStatus
from src.core.runtime.budgets import RuntimeBudgetManager
from src.core.runtime.credential_manager import RuntimeCredentialManager
from src.core.runtime.lease_manager import RuntimeLeaseManager
from src.core.runtime.locks import RuntimeLockManager
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager
from src.core.runtime.policy import RuntimePolicy, policy_from_runtime_state
from src.core.runtime.runtime_queries import RuntimeQueryService
from src.core.runtime.session_manager import RuntimeSessionManager


class SchedulingDecision(BaseModel):
    """One scheduler decision for one task candidate."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str = Field(min_length=1)
    worker_id: str | None = None
    session_id: str | None = None
    action: str = Field(min_length=1)
    accepted: bool = False
    reason: str | None = None
    required_resource_keys: list[str] = Field(default_factory=list)


class SchedulerTickResult(BaseModel):
    """Structured result returned by one scheduler tick."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    candidate_task_ids: list[str] = Field(default_factory=list)
    selected_task_ids: list[str] = Field(default_factory=list)
    decisions: list[SchedulingDecision] = Field(default_factory=list)
    blocked_task_ids: list[str] = Field(default_factory=list)
    skipped_task_ids: list[str] = Field(default_factory=list)


class RuntimeScheduler:
    """Minimal scheduler that produces assignment decisions from TG + Runtime."""

    def __init__(self) -> None:
        self.queries = RuntimeQueryService()
        self.budgets = RuntimeBudgetManager()
        self.locks = RuntimeLockManager()
        self.sessions = RuntimeSessionManager()
        self.leases = RuntimeLeaseManager()
        self.credentials = RuntimeCredentialManager()
        self.pivot_routes = RuntimePivotRouteManager()

    def tick(self, task_graph: TaskGraph, runtime_state: RuntimeState) -> SchedulerTickResult:
        """Run one scheduling tick and return a structured decision summary."""

        self._refresh_runtime_health(runtime_state)
        candidate_task_ids = self.select_schedulable_tasks(task_graph, runtime_state)
        decisions = self.assign_tasks(task_graph, runtime_state, candidate_task_ids)
        selected_task_ids = [decision.task_id for decision in decisions if decision.accepted]
        blocked_task_ids = [
            decision.task_id
            for decision in decisions
            if not decision.accepted and decision.reason in {"runtime lock conflict", "runtime budget exhausted", "no idle worker available"}
        ]
        skipped_task_ids = [
            decision.task_id
            for decision in decisions
            if not decision.accepted and decision.task_id not in blocked_task_ids
        ]
        return SchedulerTickResult(
            candidate_task_ids=candidate_task_ids,
            selected_task_ids=selected_task_ids,
            decisions=decisions,
            blocked_task_ids=blocked_task_ids,
            skipped_task_ids=skipped_task_ids,
        )

    def select_schedulable_tasks(self, task_graph: TaskGraph, runtime_state: RuntimeState) -> list[str]:
        """Return TG task IDs that satisfy minimal runtime scheduling checks."""

        selected: list[str] = []
        for node in task_graph.find_schedulable_tasks():
            if not isinstance(node, BaseTaskNode):
                continue
            if node.status != TaskStatus.READY:
                continue
            reason = self._admissibility_reason(task_graph, runtime_state, node)
            if reason is not None:
                self._audit(
                    runtime_state,
                    event_type="scheduler_reject",
                    task_id=node.id,
                    accepted=False,
                    reason=reason,
                )
                continue
            selected.append(node.id)
        return selected

    def assign_tasks(
        self,
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
        task_ids: list[str],
    ) -> list[SchedulingDecision]:
        """Build scheduler decisions for the provided candidate TG task IDs."""

        decisions: list[SchedulingDecision] = []
        available_worker_ids = [worker.worker_id for worker in self.queries.find_idle_workers(runtime_state)]
        reserved_workers: set[str] = set()
        reserved_hosts: dict[str, int] = {}
        reserved_subnets: dict[str, int] = {}

        for task_id in task_ids:
            task = task_graph.get_node(task_id)
            if not isinstance(task, BaseTaskNode):
                decisions.append(
                    SchedulingDecision(
                        task_id=task_id,
                        action="skip",
                        accepted=False,
                        reason="node is not a task",
                    )
                )
                continue
            if task.status != TaskStatus.READY:
                decisions.append(
                    SchedulingDecision(
                        task_id=task_id,
                        action="skip",
                        accepted=False,
                        reason="task is not ready in TG",
                        required_resource_keys=sorted(task.resource_keys),
                    )
                )
                continue
            reason = self._admissibility_reason(
                task_graph,
                runtime_state,
                task,
                reserved_workers=reserved_workers,
                reserved_hosts=reserved_hosts,
                reserved_subnets=reserved_subnets,
            )
            if reason is not None:
                decisions.append(
                    SchedulingDecision(
                        task_id=task_id,
                        action="block" if "conflict" in reason or "limit" in reason else "skip",
                        accepted=False,
                        reason=reason,
                        required_resource_keys=sorted(self._required_lock_keys(runtime_state, task)),
                    )
                )
                self._audit(
                    runtime_state,
                    event_type="scheduler_reject",
                    task_id=task_id,
                    accepted=False,
                    reason=reason,
                )
                continue

            worker_id = self._select_idle_worker(available_worker_ids, reserved_workers)
            if worker_id is None:
                decisions.append(
                    SchedulingDecision(
                        task_id=task_id,
                        action="defer",
                        accepted=False,
                        reason="no idle worker available",
                        required_resource_keys=sorted(task.resource_keys),
                    )
                )
                continue

            session_id = self._select_session_for_task(runtime_state, task)
            reserved_workers.add(worker_id)
            self._reserve_task_resources(
                runtime_state,
                task,
                reserved_hosts=reserved_hosts,
                reserved_subnets=reserved_subnets,
                session_id=session_id,
            )
            decisions.append(
                SchedulingDecision(
                    task_id=task_id,
                    worker_id=worker_id,
                    session_id=session_id,
                    action="assign",
                    accepted=True,
                    reason="ready in TG and admissible in runtime",
                    required_resource_keys=sorted(self._required_lock_keys(runtime_state, task)),
                )
            )
            self._audit(
                runtime_state,
                event_type="scheduler_assign",
                task_id=task_id,
                accepted=True,
                worker_id=worker_id,
                session_id=session_id,
                required_resource_keys=sorted(self._required_lock_keys(runtime_state, task)),
            )

        return decisions

    @staticmethod
    def _select_idle_worker(available_worker_ids: list[str], reserved_workers: set[str]) -> str | None:
        """Select one idle worker not already reserved in this tick."""

        for worker_id in available_worker_ids:
            if worker_id not in reserved_workers:
                return worker_id
        return None

    def _select_session_for_task(self, runtime_state: RuntimeState, task: BaseTaskNode) -> str | None:
        """Select one reusable runtime session for the task when possible.

        The current implementation is intentionally minimal. A task can hint its
        target host through `input_bindings["host_id"]` or through a `target_refs`
        entry. If no reusable session matches, the scheduler simply returns None.
        """

        bound_target = None
        if isinstance(task.input_bindings, dict):
            host_id = task.input_bindings.get("host_id")
            if isinstance(host_id, str):
                bound_target = host_id
        if bound_target is None:
            for ref in task.target_refs:
                if ref.ref_type == "Host":
                    bound_target = ref.ref_id
                    break
        sessions = self.queries.find_usable_sessions(runtime_state, bound_target=bound_target)
        return sessions[0].session_id if sessions else None

    # 中文注释：
    # 运行时策略统一放在 execution.metadata["runtime_policy"]，避免继续扩张模型字段。
    @staticmethod
    def _policy(runtime_state: RuntimeState) -> RuntimePolicy:
        return policy_from_runtime_state(runtime_state)

    # 中文注释：
    # 在每个调度 tick 前清理超时任务和过期锁，确保后续判定基于最新执行面状态。
    def _refresh_runtime_health(self, runtime_state: RuntimeState) -> None:
        self.locks.cleanup_expired_locks(runtime_state)
        self.sessions.cleanup_expired_sessions(runtime_state)
        self.leases.cleanup_expired_leases(runtime_state)
        now = utc_now()
        for session in runtime_state.sessions.values():
            if session.status != "expired":
                continue
            cleanup_reason = str(session.metadata.get("expiry_reason", "session_expired"))
            self.leases.release_leases_for_session(runtime_state, session.session_id, reason=cleanup_reason)
            self.credentials.expire_credentials_for_session(runtime_state, session.session_id, reason=cleanup_reason)
            self.pivot_routes.fail_routes_for_session(runtime_state, session.session_id, reason=cleanup_reason)
        timeout_sec = self._policy(runtime_state).default_task_timeout_sec
        for task in runtime_state.execution.tasks.values():
            if task.status not in {TaskRuntimeStatus.CLAIMED, TaskRuntimeStatus.RUNNING}:
                continue
            deadline = task.deadline
            if deadline is None and task.started_at is not None:
                deadline = task.started_at + timedelta(seconds=timeout_sec)
            if deadline is None or deadline > now:
                continue
            task.status = TaskRuntimeStatus.TIMED_OUT
            task.finished_at = now
            task.deadline = deadline
            task.last_error = "runtime task timed out"
            task.metadata["timeout_reason"] = "long_task_timeout"
            self.locks.release_all_for_owner(runtime_state, task.task_id)
            self.leases.release_leases_for_task(runtime_state, task.task_id, reason="task_timeout")
            session_id = self._task_session_binding(task)
            if session_id is not None and session_id in runtime_state.sessions:
                self.sessions.fail_session(runtime_state, session_id, reason="task_timeout")
                self.leases.release_leases_for_session(runtime_state, session_id, reason="task_timeout")
                self.credentials.expire_credentials_for_session(runtime_state, session_id, reason="task_timeout")
                self.pivot_routes.fail_routes_for_session(runtime_state, session_id, reason="task_timeout")
            if task.assigned_worker and task.assigned_worker in runtime_state.workers:
                worker = runtime_state.workers[task.assigned_worker]
                worker.current_task_id = None
                worker.status = WorkerStatus.IDLE
            self._audit(
                runtime_state,
                event_type="task_timeout",
                task_id=task.task_id,
                deadline=deadline.isoformat(),
                reason="long_task_timeout",
            )
        runtime_state.last_updated = now

    def _admissibility_reason(
        self,
        task_graph: TaskGraph,
        runtime_state: RuntimeState,
        task: BaseTaskNode,
        *,
        reserved_workers: set[str] | None = None,
        reserved_hosts: dict[str, int] | None = None,
        reserved_subnets: dict[str, int] | None = None,
    ) -> str | None:
        del task_graph
        policy = self._policy(runtime_state)
        target_hosts = self._task_host_ids(task)
        if policy.safety_stop:
            return "runtime safety stop active"
        if self._contains_blocked_host(task, target_hosts=target_hosts, policy=policy):
            return "target host blacklisted"
        if self._violates_authorization_scope(target_hosts=target_hosts, policy=policy):
            return "authorization scope denied"
        if self._violates_cidr_whitelist(task, policy=policy):
            return "cidr whitelist denied"
        if self._violates_egress_policy(task, policy=policy):
            return "egress denied by runtime policy"
        if self._requires_sensitive_approval(task, runtime_state, policy=policy):
            return "sensitive action approval required"
        required_keys = self._required_lock_keys(runtime_state, task)
        if self.queries.is_task_blocked_by_runtime(
            runtime_state,
            task_id=task.id,
            required_resource_keys=required_keys,
        ):
            return "runtime lock conflict"
        if self._has_unusable_bound_session(runtime_state, task):
            return "bound session unavailable"
        if self._has_session_policy_conflict(runtime_state, task):
            return "session policy conflict"
        if self._has_unusable_bound_credential(runtime_state, task):
            return "bound credential unavailable"
        if self._has_unusable_bound_pivot_route(runtime_state, task):
            return "pivot route unavailable"
        if self._exceeds_host_concurrency(runtime_state, task, reserved_hosts=reserved_hosts or {}, policy=policy):
            return "per-host concurrency limit reached"
        if self._exceeds_subnet_rate_limit(runtime_state, task, reserved_subnets=reserved_subnets or {}, policy=policy):
            return "per-subnet rate limit reached"
        if self._is_in_backoff_window(runtime_state, task):
            return "failure backoff active"
        if self.budgets.would_exceed_budget(
            runtime_state,
            operations=1,
            noise=task.estimated_noise,
            risk=task.estimated_risk,
        ):
            return "runtime budget exhausted"
        if reserved_workers is not None and self._select_idle_worker(
            [worker.worker_id for worker in self.queries.find_idle_workers(runtime_state)],
            reserved_workers,
        ) is None:
            return "no idle worker available"
        if reserved_workers is None and not self.queries.find_idle_workers(runtime_state):
            return "no idle worker available"
        return None

    def _required_lock_keys(self, runtime_state: RuntimeState, task: BaseTaskNode) -> set[str]:
        return self.locks.expand_policy_lock_keys(
            task.resource_keys,
            session_policies=self._session_policies(runtime_state, task),
        )

    def _session_policies(self, runtime_state: RuntimeState, task: BaseTaskNode) -> dict[str, str]:
        configured = {str(key): str(value) for key, value in self._policy(runtime_state).session_policies.items()}
        if isinstance(task.input_bindings, dict):
            session_id = task.input_bindings.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                configured.setdefault(session_id.strip(), str(task.input_bindings.get("session_policy", "exclusive")))
        return configured

    @staticmethod
    def _task_host_ids(task: BaseTaskNode) -> list[str]:
        hosts: set[str] = set()
        if isinstance(task.input_bindings, dict):
            for key in ("host_id", "target_host_id", "via_host", "source_host_id"):
                value = task.input_bindings.get(key)
                if isinstance(value, str) and value.strip():
                    hosts.add(value.strip())
        for ref in task.target_refs:
            if ref.ref_type == "Host":
                hosts.add(ref.ref_id)
        return sorted(hosts)

    @staticmethod
    def _task_session_ids(task: BaseTaskNode) -> list[str]:
        session_ids: set[str] = set()
        if isinstance(task.input_bindings, dict):
            value = task.input_bindings.get("session_id")
            if isinstance(value, str) and value.strip():
                session_ids.add(value.strip())
        for resource_key in task.resource_keys:
            if resource_key.startswith("session:"):
                session_ids.add(resource_key.split(":", 1)[1])
        return sorted(session_ids)

    @staticmethod
    def _task_credential_ids(task: BaseTaskNode) -> list[str]:
        credential_ids: set[str] = set()
        if isinstance(task.input_bindings, dict):
            value = task.input_bindings.get("credential_id")
            if isinstance(value, str) and value.strip():
                credential_ids.add(value.strip())
        for resource_key in task.resource_keys:
            if resource_key.startswith("credential:"):
                credential_ids.add(resource_key.split(":", 1)[1])
        return sorted(credential_ids)

    @staticmethod
    def _task_route_ids(task: BaseTaskNode) -> list[str]:
        route_ids: set[str] = set()
        if isinstance(task.input_bindings, dict):
            for key in ("route_id", "selected_route_id"):
                value = task.input_bindings.get(key)
                if isinstance(value, str) and value.strip():
                    route_ids.add(value.strip())
        return sorted(route_ids)

    @staticmethod
    def _task_session_binding(task: Any) -> str | None:
        metadata = getattr(task, "metadata", {})
        if isinstance(metadata, dict):
            value = metadata.get("session_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _task_subnets(self, task: BaseTaskNode) -> list[str]:
        subnets: set[str] = set()
        if isinstance(task.input_bindings, dict):
            for key in ("subnet_cidr", "cidr"):
                value = task.input_bindings.get(key)
                if isinstance(value, str) and value.strip():
                    subnets.add(value.strip())
            for key in ("address", "target_address"):
                value = task.input_bindings.get(key)
                if isinstance(value, str):
                    subnet = self._address_to_default_subnet(value)
                    if subnet is not None:
                        subnets.add(subnet)
        return sorted(subnets)

    @staticmethod
    def _address_to_default_subnet(address_text: str) -> str | None:
        try:
            address = ip_address(address_text.strip())
        except ValueError:
            return None
        if address.version == 4:
            return str(ip_network(f"{address}/24", strict=False))
        return str(ip_network(f"{address}/64", strict=False))

    @staticmethod
    def _contains_blocked_host(task: BaseTaskNode, *, target_hosts: list[str], policy: RuntimePolicy) -> bool:
        blocked = set(policy.blocked_hosts)
        if any(host in blocked for host in target_hosts):
            return True
        tags = {tag.lower() for tag in task.tags}
        return "blacklisted" in tags or "safety_blocked" in tags

    def _has_unusable_bound_session(self, runtime_state: RuntimeState, task: BaseTaskNode) -> bool:
        for session_id in self._task_session_ids(task):
            session = runtime_state.sessions.get(session_id)
            if session is None or not session.is_session_usable():
                return True
        return False

    def _has_unusable_bound_credential(self, runtime_state: RuntimeState, task: BaseTaskNode) -> bool:
        target_hosts = self._task_host_ids(task)
        bound_target = target_hosts[0] if target_hosts else None
        for credential_id in self._task_credential_ids(task):
            credential = runtime_state.credentials.get(credential_id)
            if credential is None:
                return True
            if not self.credentials.is_credential_usable_for_target(
                runtime_state,
                credential_id,
                target_id=bound_target,
            ):
                return True
        return False

    def _has_unusable_bound_pivot_route(self, runtime_state: RuntimeState, task: BaseTaskNode) -> bool:
        for route_id in self._task_route_ids(task):
            route = runtime_state.pivot_routes.get(route_id)
            if route is None or route.status.value != "active":
                return True
            if route.session_id is not None:
                session = runtime_state.sessions.get(route.session_id)
                if session is None or not session.is_session_usable():
                    return True
        return False

    @staticmethod
    def _violates_authorization_scope(*, target_hosts: list[str], policy: RuntimePolicy) -> bool:
        authorized_hosts = set(policy.authorized_hosts)
        if not authorized_hosts:
            return False
        return any(host not in authorized_hosts for host in target_hosts)

    def _violates_cidr_whitelist(self, task: BaseTaskNode, *, policy: RuntimePolicy) -> bool:
        whitelist = [ip_network(str(item), strict=False) for item in policy.cidr_whitelist]
        if not whitelist:
            return False
        for subnet_text in self._task_subnets(task):
            candidate = ip_network(subnet_text, strict=False)
            if not any(candidate.subnet_of(allowed) or candidate == allowed for allowed in whitelist):
                return True
        for key in ("address", "target_address"):
            value = task.input_bindings.get(key) if isinstance(task.input_bindings, dict) else None
            if not isinstance(value, str):
                continue
            try:
                address = ip_address(value)
            except ValueError:
                continue
            if not any(address in allowed for allowed in whitelist):
                return True
        return False

    def _violates_egress_policy(self, task: BaseTaskNode, *, policy: RuntimePolicy) -> bool:
        if not policy.deny_egress:
            return False
        whitelist = [ip_network(str(item), strict=False) for item in policy.cidr_whitelist]
        for key in ("address", "target_address"):
            value = task.input_bindings.get(key) if isinstance(task.input_bindings, dict) else None
            if not isinstance(value, str):
                continue
            try:
                address = ip_address(value)
            except ValueError:
                continue
            if whitelist and any(address in allowed for allowed in whitelist):
                continue
            return True
        return False

    @staticmethod
    def _requires_sensitive_approval(task: BaseTaskNode, runtime_state: RuntimeState, *, policy: RuntimePolicy) -> bool:
        sensitive_types = set(policy.sensitive_task_types)
        sensitive_tags = {str(item).lower() for item in policy.sensitive_tags}
        needs_approval = (
            task.approval_required
            or bool(task.gate_ids)
            or task.task_type.value in sensitive_types
            or any(tag.lower() in sensitive_tags for tag in task.tags)
        )
        if not needs_approval:
            return False
        return not bool(runtime_state.budgets.approval_cache.get(f"task:{task.id}:approved"))

    def _has_session_policy_conflict(self, runtime_state: RuntimeState, task: BaseTaskNode) -> bool:
        session_policies = self._session_policies(runtime_state, task)
        for session_id in self._task_session_ids(task):
            policy = session_policies.get(session_id, "exclusive")
            if self.locks.is_session_policy_conflict(
                runtime_state,
                session_id=session_id,
                policy=policy,
                owner_id=task.id,
            ):
                return True
        return False

    def _exceeds_host_concurrency(
        self,
        runtime_state: RuntimeState,
        task: BaseTaskNode,
        *,
        reserved_hosts: dict[str, int],
        policy: RuntimePolicy,
    ) -> bool:
        limits = dict(policy.max_concurrent_per_host)
        default_limit = int(limits.get("default", 1) or 1)
        active_by_host: dict[str, int] = {}
        for active in self.queries.find_active_tasks(runtime_state):
            for resource_key in active.resource_keys:
                if resource_key.startswith("host:"):
                    host_id = resource_key.split(":", 1)[1]
                    active_by_host[host_id] = active_by_host.get(host_id, 0) + 1
        for host_id in self._task_host_ids(task):
            limit = int(limits.get(host_id, default_limit) or default_limit)
            if active_by_host.get(host_id, 0) + reserved_hosts.get(host_id, 0) >= limit:
                return True
        return False

    def _exceeds_subnet_rate_limit(
        self,
        runtime_state: RuntimeState,
        task: BaseTaskNode,
        *,
        reserved_subnets: dict[str, int],
        policy: RuntimePolicy,
    ) -> bool:
        limits = dict(policy.rate_limit_per_subnet_per_min)
        if not limits:
            return False
        now = utc_now()
        counters = runtime_state.execution.metadata.setdefault("subnet_rate_limit", {})
        for subnet in self._task_subnets(task):
            if subnet not in limits:
                continue
            bucket = counters.setdefault(subnet, {"window_started_at": now.isoformat(), "count": 0})
            window_started = self._parse_time(bucket.get("window_started_at")) or now
            if now - window_started >= timedelta(minutes=1):
                bucket["window_started_at"] = now.isoformat()
                bucket["count"] = 0
            if int(bucket.get("count", 0)) + reserved_subnets.get(subnet, 0) >= int(limits[subnet]):
                return True
        return False

    def _is_in_backoff_window(self, runtime_state: RuntimeState, task: BaseTaskNode) -> bool:
        runtime_task = runtime_state.execution.tasks.get(task.id)
        if runtime_task is None or runtime_task.status not in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.TIMED_OUT}:
            return False
        if not runtime_task.is_task_retryable():
            return False
        base = int(task.retry_policy.backoff_seconds or self._policy(runtime_state).retry_backoff_base_sec or 0)
        if base <= 0 or runtime_task.finished_at is None:
            return False
        retry_after = runtime_task.finished_at + timedelta(seconds=base * max(runtime_task.attempt_count, 1))
        return retry_after > utc_now()

    # 中文注释：
    # 被本 tick 接受的任务要先占掉主机并发和网段速率名额，避免同一轮内部超发。
    def _reserve_task_resources(
        self,
        runtime_state: RuntimeState,
        task: BaseTaskNode,
        *,
        reserved_hosts: dict[str, int],
        reserved_subnets: dict[str, int],
        session_id: str | None,
    ) -> None:
        for host_id in self._task_host_ids(task):
            reserved_hosts[host_id] = reserved_hosts.get(host_id, 0) + 1
        now = utc_now()
        counters = runtime_state.execution.metadata.setdefault("subnet_rate_limit", {})
        for subnet in self._task_subnets(task):
            reserved_subnets[subnet] = reserved_subnets.get(subnet, 0) + 1
            bucket = counters.setdefault(subnet, {"window_started_at": now.isoformat(), "count": 0})
            window_started = self._parse_time(bucket.get("window_started_at")) or now
            if now - window_started >= timedelta(minutes=1):
                bucket["window_started_at"] = now.isoformat()
                bucket["count"] = 0
            bucket["count"] = int(bucket.get("count", 0)) + 1
        runtime_task = runtime_state.execution.tasks.get(task.id)
        if runtime_task is not None:
            runtime_task.resource_keys = set(self._required_lock_keys(runtime_state, task))
            if session_id is not None:
                runtime_task.metadata["session_id"] = session_id

    @staticmethod
    def _parse_time(value: Any) -> Any:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            from datetime import datetime

            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _audit(runtime_state: RuntimeState, *, event_type: str, **payload: Any) -> None:
        audit_log = runtime_state.execution.metadata.setdefault("audit_log", [])
        audit_log.append({"event_type": event_type, "at": utc_now().isoformat(), **payload})


__all__ = ["RuntimeScheduler", "SchedulerTickResult", "SchedulingDecision"]
