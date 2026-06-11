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
from src.core.models.ag import AttackGraph
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
            "known_services": self._known_services(knowledge_graph),
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

    def _known_services(self, graph: KnowledgeGraph) -> list[dict[str, Any]]:
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
