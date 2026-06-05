"""Compact graph context slices for LLM planning.

The graph context builder is intentionally read-only. It turns KG/AG/runtime
state into a bounded, JSON-serializable snapshot that can be sent to an LLM
without carrying raw tool output or long conversation history.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.graph.topology import NetworkTopology
from src.core.models.ag import (
    ActivationStatus,
    ActionNode,
    AttackGraph,
    ConstraintNode,
    GoalNode,
    GraphRef,
    StateNode,
    StateNodeType,
)
from src.core.models.attack_process import AttackProcessEdge, AttackProcessNodeType
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.kg import Evidence, Finding, NetworkZone, Observation
from src.core.models.runtime import RuntimeState, TaskRuntimeStatus


GraphContextRefScope = Literal["kg", "ag", "runtime", "query"]


class GraphContextBuilderConfig(BaseModel):
    """Bounds used while building a compact graph context."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_goals: int = Field(default=5, ge=1)
    max_services: int = Field(default=20, ge=1)
    max_frontier_actions: int = Field(default=20, ge=1)
    max_tasks_per_status: int = Field(default=10, ge=1)
    max_evidence_items: int = Field(default=20, ge=1)
    max_replan_requests: int = Field(default=10, ge=1)
    max_network_zones: int = Field(default=20, ge=1)
    max_reachability_paths: int = Field(default=30, ge=1)
    max_pivot_routes: int = Field(default=20, ge=1)
    max_policy_items: int = Field(default=30, ge=1)
    max_metadata_items: int = Field(default=12, ge=0)
    max_string_chars: int = Field(default=240, ge=40)
    max_evidence_chars: int = Field(default=500, ge=80)


class GraphContextRef(BaseModel):
    """Minimal reference to a graph or runtime object."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    graph: GraphContextRefScope
    ref_id: str = Field(min_length=1)
    ref_type: str | None = None
    label: str | None = None


class GraphContextGoal(BaseModel):
    """Goal node summary."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ref: GraphContextRef
    goal_type: str
    priority: int
    business_value: float
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    scope_refs: list[GraphContextRef] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphContextService(BaseModel):
    """Service-like state extracted from AG state nodes."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ref: GraphContextRef
    state_type: str
    truth_status: str
    confidence: float
    goal_relevance: float
    subject_refs: list[GraphContextRef] = Field(default_factory=list)
    host: str | None = None
    port: int | None = None
    protocol: str | None = None
    service_name: str | None = None
    version: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphContextAction(BaseModel):
    """Frontier action summary for planning."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ref: GraphContextRef
    action_type: str
    activation_status: str
    cost: float
    risk: float
    noise: float
    expected_value: float
    success_probability_prior: float
    goal_relevance: float
    approval_required: bool
    target_refs: list[GraphContextRef] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    resource_keys: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphContextTask(BaseModel):
    """Runtime task summary."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ref: GraphContextRef
    status: str
    task_type: str | None = None
    label: str | None = None
    execution_node_id: str | None = None
    source_action_id: str | None = None
    target_refs: list[GraphContextRef] = Field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 1
    last_error: str | None = None
    last_outcome_ref: str | None = None
    resource_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphContextEvidence(BaseModel):
    """Condensed event/outcome/evidence summary."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ref: GraphContextRef
    source: Literal["runtime_event", "outcome", "finding", "evidence"]
    summary: str
    task_id: str | None = None
    payload_ref: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphContextPolicy(BaseModel):
    """Policy and budget summary visible to planning."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    authorized_hosts: list[str] = Field(default_factory=list)
    cidr_whitelist: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    command_allowlist: list[str] = Field(default_factory=list)
    risk_policy: dict[str, Any] = Field(default_factory=dict)
    budget_summary: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class GraphContextNetworkZone(BaseModel):
    """Network zone summary visible to planning."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ref: GraphContextRef
    cidr: str | None = None
    zone_kind: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphContextReachabilityPath(BaseModel):
    """Reachability path summary visible to planning."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_host: str
    destination_host: str
    service_id: str | None = None
    via: str
    route_id: str | None = None
    session_id: str | None = None
    protocol: str | None = None
    port: int | None = None
    hops: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphContextPivotRoute(BaseModel):
    """Runtime pivot route summary visible to planning."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    route_id: str
    destination_host: str
    source_host: str | None = None
    via_host: str | None = None
    session_id: str | None = None
    status: str
    protocol: str | None = None
    allowed_ports: list[int] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=list)
    hop_count: int = 1
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphContext(BaseModel):
    """Compact context slice passed to graph-driven LLM planners."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    operation_id: str | None = None
    graph_versions: dict[str, Any] = Field(default_factory=dict)
    goals: list[GraphContextGoal] = Field(default_factory=list)
    known_services: list[GraphContextService] = Field(default_factory=list)
    frontier_actions: list[GraphContextAction] = Field(default_factory=list)
    tasks_by_status: dict[str, list[GraphContextTask]] = Field(default_factory=dict)
    evidence: list[GraphContextEvidence] = Field(default_factory=list)
    replan_requests: list[dict[str, Any]] = Field(default_factory=list)
    network_zones: list[GraphContextNetworkZone] = Field(default_factory=list)
    reachable_paths: list[GraphContextReachabilityPath] = Field(default_factory=list)
    pivot_routes: list[GraphContextPivotRoute] = Field(default_factory=list)
    policy: GraphContextPolicy = Field(default_factory=GraphContextPolicy)
    context_stats: dict[str, Any] = Field(default_factory=dict)


class GraphContextBuilder:
    """Build bounded LLM planning context from graph and runtime snapshots."""

    _SENSITIVE_SUBSTRINGS = (
        "raw",
        "stdout",
        "stderr",
        "output",
        "html",
        "body",
        "response",
        "content",
        "secret",
        "token",
        "api_key",
        "password",
    )

    def __init__(self, config: GraphContextBuilderConfig | None = None) -> None:
        self.config = config or GraphContextBuilderConfig()

    def build(
        self,
        *,
        knowledge_graph: KnowledgeGraph | None = None,
        attack_graph: AttackGraph | None = None,
        runtime_state: RuntimeState | None = None,
        policy_context: dict[str, Any] | None = None,
        findings: list[dict[str, Any]] | None = None,
        evidence: list[dict[str, Any]] | None = None,
    ) -> GraphContext:
        """Build a compact graph context.

        `findings` and `evidence` are accepted as already-condensed records.
        Raw tool output should be stored externally and referenced by
        `payload_ref`/artifact refs before it reaches this builder.
        """

        goals = self._build_goals(attack_graph)
        services = self._build_services(attack_graph)
        actions = self._build_frontier_actions(attack_graph)
        tasks = self._build_tasks(runtime_state=runtime_state)
        evidence_items = self._build_evidence(
            knowledge_graph=knowledge_graph,
            runtime_state=runtime_state,
            findings=findings or [],
            evidence=evidence or [],
        )
        replan_requests = self._build_replan_requests(runtime_state)
        network_zones = self._build_network_zones(knowledge_graph)
        reachable_paths = self._build_reachable_paths(knowledge_graph)
        pivot_routes = self._build_pivot_routes(runtime_state)
        policy = self._build_policy(policy_context or {}, runtime_state)

        context = GraphContext(
            operation_id=runtime_state.operation_id if runtime_state is not None else None,
            graph_versions=self._graph_versions(
                knowledge_graph=knowledge_graph,
                attack_graph=attack_graph,
            ),
            goals=goals,
            known_services=services,
            frontier_actions=actions,
            tasks_by_status=tasks,
            evidence=evidence_items,
            replan_requests=replan_requests,
            network_zones=network_zones,
            reachable_paths=reachable_paths,
            pivot_routes=pivot_routes,
            policy=policy,
        )
        return context.model_copy(update={"context_stats": self._stats(context)})

    def _build_goals(self, graph: AttackGraph | None) -> list[GraphContextGoal]:
        if graph is None:
            return []
        goals = sorted(graph.get_goal_nodes(), key=lambda item: (-item.priority, item.id))
        return [
            GraphContextGoal(
                ref=self._ag_node_ref(goal),
                goal_type=goal.goal_type.value,
                priority=goal.priority,
                business_value=goal.business_value,
                success_criteria=self._sanitize_mapping(goal.success_criteria),
                scope_refs=[self._graph_ref(ref) for ref in goal.scope_refs],
                properties=self._sanitize_mapping(goal.properties),
            )
            for goal in goals[: self.config.max_goals]
        ]

    def _build_services(self, graph: AttackGraph | None) -> list[GraphContextService]:
        if graph is None:
            return []
        service_state_types = {
            StateNodeType.SERVICE_KNOWN,
            StateNodeType.SERVICE_CONFIRMED,
            StateNodeType.REACHABILITY_VALIDATED,
            StateNodeType.PATH_CANDIDATE,
        }
        states = [
            node
            for node in graph.list_nodes()
            if isinstance(node, StateNode) and node.node_type in service_state_types
        ]
        states = sorted(states, key=lambda item: (-item.goal_relevance, -item.confidence, item.id))
        return [self._service_from_state(state) for state in states[: self.config.max_services]]

    def _service_from_state(self, state: StateNode) -> GraphContextService:
        props = self._sanitize_mapping(state.properties)
        return GraphContextService(
            ref=self._ag_node_ref(state),
            state_type=state.node_type.value,
            truth_status=state.truth_status.value,
            confidence=state.confidence,
            goal_relevance=state.goal_relevance,
            subject_refs=[self._graph_ref(ref) for ref in state.subject_refs],
            host=self._coerce_optional_str(props.get("host") or props.get("hostname") or props.get("address")),
            port=self._coerce_optional_int(props.get("port")),
            protocol=self._coerce_optional_str(props.get("protocol") or props.get("scheme")),
            service_name=self._coerce_optional_str(props.get("service") or props.get("service_name") or props.get("name")),
            version=self._coerce_optional_str(props.get("version") or props.get("product_version")),
            properties=props,
        )

    def _build_frontier_actions(self, graph: AttackGraph | None) -> list[GraphContextAction]:
        if graph is None:
            return []
        frontier_statuses = {
            ActivationStatus.ACTIVE,
            ActivationStatus.ACTIVATABLE,
            ActivationStatus.UNKNOWN,
            ActivationStatus.BLOCKED,
        }
        actions = [
            node
            for node in graph.list_nodes()
            if isinstance(node, ActionNode) and node.activation_status in frontier_statuses
        ]
        actions = sorted(
            actions,
            key=lambda item: (
                item.activation_status != ActivationStatus.ACTIVATABLE,
                -item.expected_value,
                item.risk,
                item.noise,
                item.id,
            ),
        )
        return [self._action_summary(action) for action in actions[: self.config.max_frontier_actions]]

    def _action_summary(self, action: ActionNode) -> GraphContextAction:
        blocked_reasons = [
            condition.reason
            for condition in action.activation_conditions
            if condition.reason and condition.status == ActivationStatus.BLOCKED
        ]
        return GraphContextAction(
            ref=self._ag_node_ref(action),
            action_type=action.action_type.value,
            activation_status=action.activation_status.value,
            cost=action.cost,
            risk=action.risk,
            noise=action.noise,
            expected_value=action.expected_value,
            success_probability_prior=action.success_probability_prior,
            goal_relevance=action.goal_relevance,
            approval_required=action.approval_required,
            target_refs=[self._graph_ref(ref) for ref in action.source_refs],
            required_capabilities=sorted(action.required_capabilities),
            resource_keys=sorted(action.resource_keys),
            blocked_reasons=[self._truncate(reason, self.config.max_string_chars) for reason in blocked_reasons],
            properties=self._sanitize_mapping(action.properties),
        )

    def _build_tasks(
        self,
        *,
        runtime_state: RuntimeState | None,
    ) -> dict[str, list[GraphContextTask]]:
        grouped: dict[str, list[GraphContextTask]] = {}
        if runtime_state is not None:
            for task in sorted(runtime_state.execution.tasks.values(), key=lambda item: item.task_id):
                task_metadata = dict(task.metadata)
                summary = GraphContextTask(
                    ref=GraphContextRef(graph="runtime", ref_id=task.task_id, ref_type="TaskRuntime"),
                    status=task.status.value,
                    task_type=self._coerce_optional_str(task_metadata.get("task_type") or task_metadata.get("stage_type")),
                    label=self._coerce_optional_str(task_metadata.get("label") or task_metadata.get("objective")),
                    execution_node_id=task.execution_node_id,
                    source_action_id=self._coerce_optional_str(task_metadata.get("source_action_id")),
                    target_refs=[],
                    attempt_count=task.attempt_count,
                    max_attempts=task.max_attempts,
                    last_error=self._truncate(task.last_error, self.config.max_string_chars),
                    last_outcome_ref=task.last_outcome_ref,
                    resource_keys=sorted(task.resource_keys),
                    metadata=self._sanitize_mapping(task_metadata),
                )
                grouped.setdefault(task.status.value, []).append(summary)

        return {
            status: items[: self.config.max_tasks_per_status]
            for status, items in sorted(grouped.items())
        }

    def _build_evidence(
        self,
        *,
        knowledge_graph: KnowledgeGraph | None,
        runtime_state: RuntimeState | None,
        findings: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[GraphContextEvidence]:
        items: list[GraphContextEvidence] = []
        items.extend(self._kg_evidence(knowledge_graph))
        if runtime_state is not None:
            for event in runtime_state.pending_events[-self.config.max_evidence_items :]:
                if event.summary:
                    items.append(
                        GraphContextEvidence(
                            ref=GraphContextRef(graph="runtime", ref_id=event.event_id, ref_type=event.event_type),
                            source="runtime_event",
                            summary=self._truncate(event.summary, self.config.max_evidence_chars) or "",
                            payload_ref=event.payload_ref,
                            created_at=event.created_at,
                            metadata=self._sanitize_mapping(event.metadata),
                        )
                    )
            for outcome in runtime_state.recent_outcomes[-self.config.max_evidence_items :]:
                items.append(
                    GraphContextEvidence(
                        ref=GraphContextRef(graph="runtime", ref_id=outcome.outcome_id, ref_type=outcome.outcome_type),
                        source="outcome",
                        summary=self._truncate(outcome.summary, self.config.max_evidence_chars) or "",
                        task_id=outcome.task_id,
                        payload_ref=outcome.payload_ref,
                        created_at=outcome.created_at,
                        metadata=self._sanitize_mapping(outcome.metadata),
                    )
                )
        items.extend(self._external_evidence(findings, source="finding"))
        items.extend(self._external_evidence(evidence, source="evidence"))
        return items[: self.config.max_evidence_items]

    def _kg_evidence(self, graph: KnowledgeGraph | None) -> list[GraphContextEvidence]:
        if graph is None:
            return []
        result: list[GraphContextEvidence] = []
        for node in graph.list_nodes():
            if not isinstance(node, (Evidence, Finding, Observation)):
                continue
            summary = getattr(node, "summary", None) or node.label
            source: Literal["finding", "evidence"] = "finding" if isinstance(node, Finding) else "evidence"
            result.append(
                GraphContextEvidence(
                    ref=GraphContextRef(graph="kg", ref_id=node.id, ref_type=node.type.value, label=node.label),
                    source=source,
                    summary=self._truncate(summary, self.config.max_evidence_chars) or node.label,
                    payload_ref=getattr(node, "content_ref", None),
                    created_at=node.last_seen,
                    metadata=self._sanitize_mapping(
                        {
                            "status": node.status.value,
                            "confidence": node.confidence,
                            "source_task_id": node.source_task_id,
                            **dict(node.properties),
                        }
                    ),
                )
            )
        return sorted(result, key=lambda item: item.created_at or datetime.min, reverse=True)

    def _external_evidence(
        self,
        records: list[dict[str, Any]],
        *,
        source: Literal["finding", "evidence"],
    ) -> list[GraphContextEvidence]:
        result: list[GraphContextEvidence] = []
        for index, record in enumerate(records):
            summary = (
                record.get("summary")
                or record.get("title")
                or record.get("description")
                or record.get("evidence_summary")
            )
            if not isinstance(summary, str) or not summary.strip():
                continue
            ref_id = str(record.get("finding_id") or record.get("evidence_id") or record.get("id") or f"{source}-{index}")
            result.append(
                GraphContextEvidence(
                    ref=GraphContextRef(graph="query", ref_id=ref_id, ref_type=source),
                    source=source,
                    summary=self._truncate(summary, self.config.max_evidence_chars) or "",
                    task_id=self._coerce_optional_str(record.get("task_id")),
                    payload_ref=self._coerce_optional_str(record.get("payload_ref") or record.get("artifact_ref")),
                    metadata=self._sanitize_mapping(record.get("metadata") if isinstance(record.get("metadata"), dict) else {}),
                )
            )
        return result

    def _build_replan_requests(self, runtime_state: RuntimeState | None) -> list[dict[str, Any]]:
        if runtime_state is None:
            return []
        requests = runtime_state.replan_requests[-self.config.max_replan_requests :]
        return [
            {
                "request_id": request.request_id,
                "reason": self._truncate(request.reason, self.config.max_string_chars),
                "task_ids": list(request.task_ids),
                "scope": request.scope,
                "metadata": self._sanitize_mapping(request.metadata),
            }
            for request in requests
        ]

    def _build_network_zones(self, graph: KnowledgeGraph | None) -> list[GraphContextNetworkZone]:
        if graph is None:
            return []
        zones = [node for node in graph.list_nodes() if isinstance(node, NetworkZone)]
        zones = sorted(zones, key=lambda item: item.id)[: self.config.max_network_zones]
        return [
            GraphContextNetworkZone(
                ref=GraphContextRef(graph="kg", ref_id=zone.id, ref_type=zone.type.value, label=zone.label),
                cidr=zone.cidr,
                zone_kind=zone.zone_kind,
                properties=self._sanitize_mapping(zone.properties),
            )
            for zone in zones
        ]

    def _build_reachable_paths(self, graph: KnowledgeGraph | None) -> list[GraphContextReachabilityPath]:
        if graph is None:
            return []
        topology = NetworkTopology(graph)
        paths = topology.reachable_paths()[: self.config.max_reachability_paths]
        return [
            GraphContextReachabilityPath(
                source_host=path.source_host,
                destination_host=path.destination_host,
                service_id=path.service_id,
                via=path.via,
                route_id=path.route_id,
                session_id=path.session_id,
                protocol=path.protocol,
                port=path.port,
                hops=path.hops,
                confidence=path.confidence,
                metadata=self._sanitize_mapping(path.metadata),
            )
            for path in paths
        ]

    def _build_pivot_routes(self, runtime_state: RuntimeState | None) -> list[GraphContextPivotRoute]:
        if runtime_state is None:
            return []
        routes = sorted(runtime_state.pivot_routes.values(), key=lambda item: item.route_id)[: self.config.max_pivot_routes]
        return [
            GraphContextPivotRoute(
                route_id=route.route_id,
                destination_host=route.destination_host,
                source_host=route.source_host,
                via_host=route.via_host,
                session_id=route.session_id,
                status=route.status.value,
                protocol=route.protocol,
                allowed_ports=sorted(route.allowed_ports),
                protocols=sorted(route.protocols),
                hop_count=route.hop_count,
                confidence=route.confidence,
                metadata=self._sanitize_mapping(route.metadata),
            )
            for route in routes
        ]

    def _build_policy(
        self,
        policy_context: dict[str, Any],
        runtime_state: RuntimeState | None,
    ) -> GraphContextPolicy:
        budget_summary = {}
        if runtime_state is not None:
            budget_summary = {
                "time_budget_used_sec": runtime_state.budgets.time_budget_used_sec,
                "time_budget_max_sec": runtime_state.budgets.time_budget_max_sec,
                "token_budget_used": runtime_state.budgets.token_budget_used,
                "token_budget_max": runtime_state.budgets.token_budget_max,
                "operation_budget_used": runtime_state.budgets.operation_budget_used,
                "operation_budget_max": runtime_state.budgets.operation_budget_max,
                "noise_budget_used": runtime_state.budgets.noise_budget_used,
                "noise_budget_max": runtime_state.budgets.noise_budget_max,
                "risk_budget_used": runtime_state.budgets.risk_budget_used,
                "risk_budget_max": runtime_state.budgets.risk_budget_max,
                "policy_flags": self._sanitize_mapping(runtime_state.budgets.policy_flags),
            }
        reserved = {
            "authorized_hosts",
            "cidr_whitelist",
            "disabled_tools",
            "command_allowlist",
            "risk_policy",
        }
        extra = {
            key: value
            for key, value in self._sanitize_mapping(policy_context).items()
            if key not in reserved
        }
        return GraphContextPolicy(
            authorized_hosts=self._coerce_string_list(policy_context.get("authorized_hosts")),
            cidr_whitelist=self._coerce_string_list(policy_context.get("cidr_whitelist")),
            disabled_tools=self._coerce_string_list(policy_context.get("disabled_tools")),
            command_allowlist=self._coerce_string_list(policy_context.get("command_allowlist")),
            risk_policy=self._sanitize_mapping(policy_context.get("risk_policy") if isinstance(policy_context.get("risk_policy"), dict) else {}),
            budget_summary=budget_summary,
            extra=dict(list(extra.items())[: self.config.max_policy_items]),
        )

    def _graph_versions(
        self,
        *,
        knowledge_graph: KnowledgeGraph | None,
        attack_graph: AttackGraph | None,
    ) -> dict[str, Any]:
        versions: dict[str, Any] = {}
        if knowledge_graph is not None:
            versions["kg_version"] = knowledge_graph.version
            versions["kg_last_patch_batch_id"] = knowledge_graph.last_patch_batch_id
        if attack_graph is not None:
            versions["ag_version"] = attack_graph.version
            versions["source_kg_version"] = attack_graph.source_kg_version
            versions["projection_batch_id"] = attack_graph.projection_batch_id
        return versions

    def _stats(self, context: GraphContext) -> dict[str, Any]:
        dumped = context.model_dump(mode="json", exclude={"context_stats"})
        return {
            "goal_count": len(context.goals),
            "known_service_count": len(context.known_services),
            "frontier_action_count": len(context.frontier_actions),
            "task_count": sum(len(items) for items in context.tasks_by_status.values()),
            "evidence_count": len(context.evidence),
            "network_zone_count": len(context.network_zones),
            "reachable_path_count": len(context.reachable_paths),
            "pivot_route_count": len(context.pivot_routes),
            "estimated_context_chars": len(str(dumped)),
            "large_artifacts_included": False,
        }

    def _sanitize_mapping(self, value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, raw_value in value.items():
            if len(result) >= self.config.max_metadata_items:
                break
            key_text = str(key)
            if self._is_sensitive_or_raw_key(key_text):
                if key_text.endswith("_ref") or key_text.endswith("ref"):
                    result[key_text] = self._sanitize_value(raw_value)
                continue
            sanitized = self._sanitize_value(raw_value)
            if sanitized is not None:
                result[key_text] = sanitized
        return result

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._truncate(value, self.config.max_string_chars)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, GraphRef):
            return self._graph_ref(value).model_dump(mode="json")
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value[: self.config.max_metadata_items]]
        if isinstance(value, set):
            return [self._sanitize_value(item) for item in sorted(value, key=str)[: self.config.max_metadata_items]]
        if isinstance(value, dict):
            return self._sanitize_mapping(value)
        return self._truncate(str(value), self.config.max_string_chars)

    def _is_sensitive_or_raw_key(self, key: str) -> bool:
        lowered = key.lower()
        return any(part in lowered for part in self._SENSITIVE_SUBSTRINGS)

    def _ag_node_ref(self, node: GoalNode | StateNode | ActionNode | ConstraintNode) -> GraphContextRef:
        return GraphContextRef(graph="ag", ref_id=node.id, ref_type=type(node).__name__, label=node.label)

    @staticmethod
    def _graph_ref(ref: GraphRef) -> GraphContextRef:
        return GraphContextRef(graph=ref.graph, ref_id=ref.ref_id, ref_type=ref.ref_type, label=ref.label)

    @staticmethod
    def _truncate(value: str | None, limit: int) -> str | None:
        if value is None:
            return None
        if len(value) <= limit:
            return value
        return f"{value[: max(0, limit - 15)]}...[truncated]"

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [str(item) for item in items if str(item).strip()]


class TwoGraphContextBuilder:
    """Build the KG/AG/runtime context used by PlannerAgent and StageAgent."""

    PROCESS_NODE_LIMIT = 20
    EVIDENCE_LIMIT = 20
    ASSET_LIMIT = 30
    SERVICE_LIMIT = 30
    FAILURE_LIMIT = 12
    STRING_LIMIT = 240

    def build(
        self,
        *,
        knowledge_graph: KnowledgeGraph,
        attack_graph: AttackGraph,
        runtime_state: RuntimeState,
        policy_context: dict[str, Any] | None = None,
        current_goal: str | None = None,
    ) -> dict[str, Any]:
        return {
            "kg_summary": self._kg_summary(knowledge_graph),
            "ag_process_summary": self._ag_process_summary(attack_graph),
            "runtime_summary": self._runtime_summary(runtime_state),
            "policy_summary": self._sanitize_mapping(policy_context or {}),
            "recent_evidence": self._recent_evidence(knowledge_graph, runtime_state),
            "known_assets": self._known_assets(knowledge_graph),
            "known_services": self._known_services(knowledge_graph, attack_graph),
            "active_sessions": self._active_sessions(runtime_state),
            "recent_attack_process_nodes": self._recent_attack_process_nodes(attack_graph),
            "recent_handoff_suggestions": self._recent_handoff_suggestions(attack_graph),
            "recent_failures": self._recent_failures(attack_graph, runtime_state),
            "current_goal": current_goal or runtime_state.execution.active_goal_id or runtime_state.execution.summary or "",
        }

    def _kg_summary(self, graph: KnowledgeGraph) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for node in graph.list_nodes():
            node_type = getattr(getattr(node, "type", None), "value", getattr(node, "type", None))
            counts[str(node_type)] = counts.get(str(node_type), 0) + 1
        return {
            "version": graph.version,
            "node_count": len(graph.list_nodes()),
            "edge_count": len(graph.list_edges()),
            "node_counts": counts,
        }

    def _ag_process_summary(self, graph: AttackGraph) -> dict[str, Any]:
        nodes = graph.find_process_nodes()
        counts: dict[str, int] = {}
        for node in nodes:
            counts[node.node_type.value] = counts.get(node.node_type.value, 0) + 1
        return {
            "version": graph.version,
            "process_node_count": len(nodes),
            "process_edge_count": len([edge for edge in graph.list_edges() if isinstance(edge, AttackProcessEdge)]),
            "node_counts": counts,
        }

    @staticmethod
    def _runtime_summary(state: RuntimeState) -> dict[str, Any]:
        return {
            "operation_id": state.operation_id,
            "operation_status": state.operation_status.value,
            "execution_status": state.execution.status.value,
            "task_count": len(state.execution.tasks),
            "session_count": len(state.sessions),
            "credential_count": len(state.credentials),
            "pivot_route_count": len(state.pivot_routes),
            "pending_event_count": len(state.pending_events),
            "recent_outcome_count": len(state.recent_outcomes),
            "replan_request_count": len(state.replan_requests),
            "budget_summary": {
                "operation_budget_used": state.budgets.operation_budget_used,
                "operation_budget_max": state.budgets.operation_budget_max,
                "noise_budget_used": state.budgets.noise_budget_used,
                "noise_budget_max": state.budgets.noise_budget_max,
                "risk_budget_used": state.budgets.risk_budget_used,
                "risk_budget_max": state.budgets.risk_budget_max,
                "policy_flags": dict(state.budgets.policy_flags),
            },
        }

    def _recent_evidence(self, graph: KnowledgeGraph, state: RuntimeState) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for node in graph.list_nodes():
            node_type = getattr(getattr(node, "type", None), "value", "")
            if node_type not in {"Evidence", "Finding", "Observation", "evidence", "finding", "observation"}:
                continue
            items.append(
                {
                    "ref": {"graph": "kg", "ref_id": node.id, "ref_type": str(node_type), "label": node.label},
                    "summary": self._truncate(getattr(node, "summary", None) or node.label),
                    "payload_ref": getattr(node, "content_ref", None),
                    "created_at": getattr(node, "last_seen", None) or getattr(node, "observed_at", None),
                }
            )
        for event in state.pending_events[-self.EVIDENCE_LIMIT :]:
            items.append(
                {
                    "ref": {"graph": "query", "ref_id": event.event_id, "ref_type": event.event_type},
                    "summary": self._truncate(event.summary),
                    "payload_ref": event.payload_ref,
                    "created_at": event.created_at,
                }
            )
        for outcome in state.recent_outcomes[-self.EVIDENCE_LIMIT :]:
            items.append(
                {
                    "ref": {"graph": "query", "ref_id": outcome.outcome_id, "ref_type": outcome.outcome_type},
                    "summary": self._truncate(outcome.summary),
                    "payload_ref": outcome.payload_ref,
                    "created_at": outcome.created_at,
                }
            )
        return [self._json_safe(item) for item in items[-self.EVIDENCE_LIMIT :]]

    def _known_assets(self, graph: KnowledgeGraph) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        for node in graph.list_nodes():
            node_type = str(getattr(getattr(node, "type", None), "value", getattr(node, "type", "")))
            if node_type not in {"Host", "DataAsset", "NetworkZone"}:
                continue
            assets.append(
                self._json_safe(
                    {
                        "ref": {"graph": "kg", "ref_id": node.id, "ref_type": node_type, "label": node.label},
                        "address": getattr(node, "address", None),
                        "hostname": getattr(node, "hostname", None),
                        "platform": getattr(node, "platform", None),
                        "asset_kind": getattr(node, "asset_kind", None),
                        "cidr": getattr(node, "cidr", None),
                    }
                )
            )
        return assets[: self.ASSET_LIMIT]

    def _known_services(self, graph: KnowledgeGraph, attack_graph: AttackGraph) -> list[dict[str, Any]]:
        services: list[dict[str, Any]] = []
        for node in graph.list_nodes():
            node_type = str(getattr(getattr(node, "type", None), "value", getattr(node, "type", "")))
            if node_type != "Service":
                continue
            services.append(
                self._json_safe(
                    {
                        "ref": {"graph": "kg", "ref_id": node.id, "ref_type": node_type, "label": node.label},
                        "service_name": getattr(node, "service_name", None),
                        "port": getattr(node, "port", None),
                        "protocol": getattr(node, "protocol", None),
                    }
                )
            )
        service_state_types = {StateNodeType.SERVICE_KNOWN, StateNodeType.SERVICE_CONFIRMED, StateNodeType.REACHABILITY_VALIDATED}
        for node in attack_graph.list_nodes():
            if not isinstance(node, StateNode) or node.node_type not in service_state_types:
                continue
            props = dict(node.properties)
            services.append(
                self._json_safe(
                    {
                        "ref": {"graph": "ag", "ref_id": node.id, "ref_type": node.node_type.value, "label": node.label},
                        "service_name": props.get("service") or props.get("service_name") or props.get("name"),
                        "host": props.get("host") or props.get("hostname") or props.get("address"),
                        "port": props.get("port"),
                        "protocol": props.get("protocol"),
                        "confidence": node.confidence,
                    }
                )
            )
        return services[: self.SERVICE_LIMIT]

    def _active_sessions(self, state: RuntimeState) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for session in sorted(state.sessions.values(), key=lambda item: item.session_id):
            status = getattr(getattr(session, "status", None), "value", getattr(session, "status", None))
            if status and str(status).lower() not in {"active", "ready", "available"}:
                continue
            sessions.append(self._json_safe(session.model_dump(mode="json")))
        return sessions[: self.ASSET_LIMIT]

    def _recent_attack_process_nodes(self, graph: AttackGraph) -> list[dict[str, Any]]:
        return [
            self._process_node_summary(node)
            for node in graph.recent_process_nodes(limit=self.PROCESS_NODE_LIMIT)
        ]

    def _recent_handoff_suggestions(self, graph: AttackGraph) -> list[dict[str, Any]]:
        handoffs = graph.find_process_nodes(node_type=AttackProcessNodeType.HANDOFF_SUGGESTION)
        handoffs = sorted(handoffs, key=lambda item: item.created_at, reverse=True)
        return [self._process_node_summary(node) for node in handoffs[: self.PROCESS_NODE_LIMIT]]

    def _recent_failures(self, graph: AttackGraph, state: RuntimeState) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for node in graph.recent_process_nodes(limit=self.PROCESS_NODE_LIMIT * 2):
            if str(node.status or "").lower() in {"failed", "blocked", "needs_replan", "need_more_info"}:
                failures.append(self._process_node_summary(node))
        for outcome in state.recent_outcomes[-self.FAILURE_LIMIT :]:
            status = str(outcome.metadata.get("status") or outcome.outcome_type).lower()
            if "fail" in status or "block" in status:
                failures.append(
                    self._json_safe(
                        {
                            "ref": {"graph": "query", "ref_id": outcome.outcome_id, "ref_type": outcome.outcome_type},
                            "summary": self._truncate(outcome.summary),
                            "task_id": outcome.task_id,
                        }
                    )
                )
        return failures[: self.FAILURE_LIMIT]

    def _process_node_summary(self, node: Any) -> dict[str, Any]:
        return self._json_safe(
            {
                "id": node.id,
                "node_type": node.node_type.value,
                "label": node.label,
                "cycle_index": node.cycle_index,
                "agent_name": node.agent_name,
                "stage_type": node.stage_type,
                "status": node.status,
                "summary": self._truncate(node.summary),
                "evidence_refs": list(node.evidence_refs),
                "created_at": node.created_at,
                "properties": self._sanitize_mapping(node.properties),
            }
        )

    def _sanitize_mapping(self, value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, raw in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(part in lowered for part in ("stdout", "stderr", "raw", "body", "content", "secret", "token", "password")):
                if not lowered.endswith("_ref") and not lowered.endswith("ref"):
                    continue
            result[key_text] = self._sanitize_value(raw)
            if len(result) >= 20:
                break
        return result

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._truncate(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value[:20]]
        if isinstance(value, dict):
            return self._sanitize_mapping(value)
        return self._truncate(str(value))

    @classmethod
    def _truncate(cls, value: Any) -> str:
        text = "" if value is None else str(value)
        if len(text) <= cls.STRING_LIMIT:
            return text
        return f"{text[: cls.STRING_LIMIT - 15]}...[truncated]"

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: TwoGraphContextBuilder._json_safe(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [TwoGraphContextBuilder._json_safe(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        return value


__all__ = [
    "GraphContext",
    "GraphContextAction",
    "GraphContextBuilder",
    "GraphContextBuilderConfig",
    "GraphContextEvidence",
    "GraphContextGoal",
    "GraphContextNetworkZone",
    "GraphContextPivotRoute",
    "GraphContextPolicy",
    "GraphContextReachabilityPath",
    "GraphContextRef",
    "GraphContextService",
    "GraphContextTask",
    "TwoGraphContextBuilder",
]
