"""Initial KG / AG / TG construction for a new operation."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import AttackGraphTaskBuilder, TaskGenerationRequest
from src.core.models.ag import ActionNode, ActionNodeType, AttackGraph, stable_node_id
from src.core.models.kg import BelongsToZoneEdge, Goal, Host, NetworkZone, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.models.tg import BaseTaskNode, TaskGraph


@dataclass(frozen=True)
class InitialTarget:
    """Normalized user-supplied target for graph initialization."""

    raw: str
    kind: str
    host_value: str
    label: str
    address: str | None = None
    hostname: str | None = None
    url: str | None = None
    scheme: str | None = None
    port: int | None = None


@dataclass(frozen=True)
class GraphInitializationResult:
    """Graphs and selected initial work produced for one operation."""

    operation_id: str
    target: InitialTarget
    kg: KnowledgeGraph
    ag: AttackGraph
    tg: TaskGraph
    host_id: str
    goal_id: str
    scope_id: str
    initial_action_ids: list[str]
    initial_task_ids: list[str]


class GraphInitializer:
    """Create and persist the initial graph memory state for an operation."""

    def __init__(
        self,
        store: GraphMemoryStore | None = None,
        *,
        projector: AttackGraphProjector | None = None,
        task_builder: AttackGraphTaskBuilder | None = None,
    ) -> None:
        self._store = store or GraphMemoryStore()
        self._projector = projector or AttackGraphProjector()
        self._task_builder = task_builder or AttackGraphTaskBuilder()

    def initialize(
        self,
        *,
        operation_id: str,
        target: str,
        goal_label: str | None = None,
        goal_description: str | None = None,
        goal_category: str = "context",
        persist: bool = True,
    ) -> GraphInitializationResult:
        """Build initial KG / AG / TG from a user target and optionally persist it."""

        normalized_target = normalize_initial_target(target)
        kg = self._build_initial_kg(
            operation_id=operation_id,
            target=normalized_target,
            goal_label=goal_label,
            goal_description=goal_description,
            goal_category=goal_category,
        )
        ag = self._projector.project(kg, goal_context={"goal_ids": [self._goal_id(operation_id)]})
        initial_actions = self._select_initial_actions(ag, self._host_id(operation_id, normalized_target))
        tg = self._build_initial_tg(ag, initial_actions)
        initial_task_ids = [
            node.id
            for node in tg.list_nodes()
            if isinstance(node, BaseTaskNode) and node.source_action_id in {action.id for action in initial_actions}
        ]

        if persist:
            self._store.save_kg(operation_id, kg)
            self._store.save_ag(operation_id, ag)
            self._store.save_tg(operation_id, tg)

        return GraphInitializationResult(
            operation_id=operation_id,
            target=normalized_target,
            kg=kg,
            ag=ag,
            tg=tg,
            host_id=self._host_id(operation_id, normalized_target),
            goal_id=self._goal_id(operation_id),
            scope_id=self._scope_id(operation_id, normalized_target),
            initial_action_ids=[action.id for action in initial_actions],
            initial_task_ids=sorted(initial_task_ids),
        )

    def _build_initial_kg(
        self,
        *,
        operation_id: str,
        target: InitialTarget,
        goal_label: str | None,
        goal_description: str | None,
        goal_category: str,
    ) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        host_id = self._host_id(operation_id, target)
        goal_id = self._goal_id(operation_id)
        scope_id = self._scope_id(operation_id, target)

        kg.add_node(
            Host(
                id=host_id,
                label=target.label,
                address=target.address,
                hostname=target.hostname,
                status=EntityStatus.OBSERVED,
                confidence=0.9,
                properties={
                    "operation_id": operation_id,
                    "target_kind": target.kind,
                    "target": target.raw,
                    **self._target_properties(target),
                },
                tags={"initial_target", target.kind},
            )
        )
        kg.add_node(
            NetworkZone(
                id=scope_id,
                label=f"Scope: {target.label}",
                cidr=target.raw if target.kind == "cidr" else None,
                zone_kind="operation_scope",
                status=EntityStatus.OBSERVED,
                confidence=1.0,
                properties={
                    "operation_id": operation_id,
                    "target": target.raw,
                    "target_kind": target.kind,
                    "allowed": True,
                },
                tags={"scope", "initial_scope"},
            )
        )
        kg.add_node(
            Goal(
                id=goal_id,
                label=goal_label or f"Validate target context: {target.label}",
                category=goal_category,
                description=goal_description or f"Build initial context for {target.raw}",
                status=EntityStatus.OBSERVED,
                confidence=0.9,
                properties={
                    "operation_id": operation_id,
                    "priority": 80,
                    "business_value": 0.8,
                    "target": target.raw,
                    "target_host_id": host_id,
                    "scope_id": scope_id,
                },
                tags={"initial_goal"},
            )
        )
        kg.add_edge(
            BelongsToZoneEdge(
                id=stable_node_id("kg-edge", {"type": "BELONGS_TO_ZONE", "source": host_id, "target": scope_id}),
                label="belongs_to_zone",
                source=host_id,
                target=scope_id,
                status=EntityStatus.OBSERVED,
                confidence=1.0,
            )
        )
        kg.add_edge(
            TargetsEdge(
                id=stable_node_id("kg-edge", {"type": "TARGETS", "source": goal_id, "target": host_id}),
                label="targets",
                source=goal_id,
                target=host_id,
                status=EntityStatus.OBSERVED,
                confidence=1.0,
            )
        )
        return kg

    def _select_initial_actions(self, ag: AttackGraph, host_id: str) -> list[ActionNode]:
        actions = [
            action
            for action in ag.find_actions(ActionNodeType.ENUMERATE_HOST)
            if any(ref.graph == "kg" and ref.ref_id == host_id for ref in action.source_refs)
        ]
        return sorted(actions, key=lambda action: action.id)

    def _build_initial_tg(self, ag: AttackGraph, actions: list[ActionNode]) -> TaskGraph:
        result = self._task_builder.build_candidates(
            ag,
            TaskGenerationRequest(
                action_ids=[action.id for action in actions],
                include_evidence_tasks=False,
                group_label="Initial Probe",
            ),
        )
        if result.task_graph is None:
            return TaskGraph()
        return TaskGraph.from_dict(result.task_graph)

    @staticmethod
    def _target_properties(target: InitialTarget) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "url": target.url,
                "scheme": target.scheme,
                "port": target.port,
            }.items()
            if value is not None
        }

    @staticmethod
    def _host_id(operation_id: str, target: InitialTarget) -> str:
        return stable_node_id(
            "kg-host",
            {
                "operation_id": operation_id,
                "target": target.host_value,
                "kind": target.kind,
            },
        )

    @staticmethod
    def _goal_id(operation_id: str) -> str:
        return stable_node_id("kg-goal", {"operation_id": operation_id, "kind": "initial_goal"})

    @staticmethod
    def _scope_id(operation_id: str, target: InitialTarget) -> str:
        return stable_node_id(
            "kg-scope",
            {
                "operation_id": operation_id,
                "target": target.raw,
                "kind": target.kind,
            },
        )


def initialize_graph_memory(
    *,
    operation_id: str,
    target: str,
    store: GraphMemoryStore | None = None,
    root_dir: str | Path | None = None,
    goal_label: str | None = None,
    goal_description: str | None = None,
    goal_category: str = "context",
) -> GraphInitializationResult:
    """Convenience function for creating and saving initial graph memory."""

    if store is not None and root_dir is not None:
        raise ValueError("provide either store or root_dir, not both")
    resolved_store = store or GraphMemoryStore(root_dir or "runtime-store")
    return GraphInitializer(resolved_store).initialize(
        operation_id=operation_id,
        target=target,
        goal_label=goal_label,
        goal_description=goal_description,
        goal_category=goal_category,
        persist=True,
    )


def normalize_initial_target(target: str) -> InitialTarget:
    """Normalize an IP, hostname or URL into the initial KG target shape."""

    raw = target.strip()
    if not raw:
        raise ValueError("target must be non-empty")

    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        host = parsed.hostname
        if not host:
            raise ValueError("URL target must include a host")
        return InitialTarget(
            raw=raw,
            kind="url",
            host_value=host,
            label=host,
            address=_address_or_none(host),
            hostname=None if _is_ip_address(host) else host,
            url=raw,
            scheme=parsed.scheme,
            port=parsed.port,
        )

    if "/" in raw:
        return InitialTarget(
            raw=raw,
            kind="cidr",
            host_value=raw,
            label=raw,
        )

    if _is_ip_address(raw):
        return InitialTarget(
            raw=raw,
            kind="ip",
            host_value=raw,
            label=raw,
            address=raw,
        )

    return InitialTarget(
        raw=raw,
        kind="hostname",
        host_value=raw,
        label=raw,
        hostname=raw,
    )


def _is_ip_address(value: str) -> bool:
    try:
        ip_address(value)
    except ValueError:
        return False
    return True


def _address_or_none(value: str) -> str | None:
    return value if _is_ip_address(value) else None


__all__ = [
    "GraphInitializationResult",
    "GraphInitializer",
    "InitialTarget",
    "initialize_graph_memory",
    "normalize_initial_target",
]
