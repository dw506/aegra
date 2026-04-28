"""KG -> AG projection utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import (
    AGEdgeType,
    ActivationCondition,
    ActivationStatus,
    ActionNode,
    ActionNodeType,
    AttackGraph,
    BaseAGEdge,
    ConstraintNode,
    ConstraintNodeType,
    GoalNode,
    GoalNodeType,
    GraphRef,
    ProjectionTrace,
    StateNode,
    StateNodeType,
    TruthStatus,
    stable_node_id,
)
from src.core.models.kg import BaseEdge, BaseNode, Credential, DataAsset, Goal, Host, Identity, PrivilegeState, Service, Session
from src.core.models.kg_enums import EdgeType, EntityStatus, NodeType


class AttackGraphProjector:
    """Project stable planning states and actions from the Knowledge Graph."""

    def project(
        self,
        kg: KnowledgeGraph,
        goal_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
    ) -> AttackGraph:
        """Build a fresh Attack Graph from the current KG snapshot."""

        ag = AttackGraph()
        goal_refs = self._build_goal_nodes(kg, ag, goal_context)
        self._build_constraint_nodes(ag, policy_context)
        self._build_state_nodes(kg, ag, goal_refs)
        self._build_action_nodes(kg, ag)
        self._connect_enablement(ag)
        self._connect_constraints(ag)
        self._refresh_goal_state_nodes(ag)
        self._refresh_action_activation(ag)
        # 这里把 AG 快照显式锚定到来源 KG 版本，便于后续 TG 构建和 checkpoint 跟踪。
        ag.set_projection_metadata(
            source_kg_version=kg.version,
            projection_batch_id=kg.last_patch_batch_id,
            metadata={
                "goal_context": dict(goal_context or {}),
                "policy_context": dict(policy_context or {}),
                "source_change_count": kg.delta.change_count,
            },
            version=max(ag.version, kg.version),
        )
        return ag

    def project_incremental(
        self,
        kg: KnowledgeGraph,
        existing_graph: AttackGraph | None,
        changed_refs: list[str] | None = None,
        goal_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
    ) -> AttackGraph:
        """Replace only AG slices affected by the changed KG refs."""

        fresh = self.project(kg, goal_context=goal_context, policy_context=policy_context)
        if existing_graph is None or not changed_refs:
            return fresh

        merged = AttackGraph.from_dict(existing_graph.to_dict())
        affected_old = self._collect_affected_node_ids(existing_graph, changed_refs)
        affected_new = self._collect_affected_node_ids(fresh, changed_refs)

        for node_id in sorted(affected_old):
            if node_id in merged._nodes:
                merged.remove_node(node_id)
        for node_id in sorted(affected_new):
            if node_id not in merged._nodes:
                merged.add_node(deepcopy(fresh.get_node(node_id)))
        for edge in fresh.list_edges():
            if edge.source in affected_new and edge.target in affected_new:
                if edge.id not in merged._edges:
                    merged.add_edge(deepcopy(edge))
            elif edge.source in merged._nodes and edge.target in merged._nodes and edge.id not in merged._edges:
                merged.add_edge(deepcopy(edge))
        self._refresh_action_activation(merged)
        merged.set_projection_metadata(
            source_kg_version=fresh.source_kg_version,
            projection_batch_id=fresh.projection_batch_id,
            metadata=fresh.metadata,
            version=max(merged.version, fresh.version),
        )
        return merged

    def _build_goal_nodes(
        self,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        goal_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        goal_refs: dict[str, set[str]] = {}
        requested_goal_ids = set((goal_context or {}).get("goal_ids", []))
        for goal in [node for node in kg.list_nodes(type=NodeType.GOAL) if isinstance(node, Goal)]:
            if requested_goal_ids and goal.id not in requested_goal_ids:
                continue
            related_entities = kg.get_goal_related_entities(goal.id)
            goal_refs[goal.id] = {entity.id for entity in related_entities}
            goal_node = GoalNode(
                id=stable_node_id("ag-goal", {"goal": goal.id}),
                goal_type=self._goal_type_for_goal(goal),
                label=goal.label,
                success_criteria={
                    "goal_id": goal.id,
                    "required_state_types": [
                        StateNodeType.GOAL_RELEVANT_DATA_LOCATED.value,
                        StateNodeType.GOAL_STATE_SATISFIED.value,
                    ],
                },
                priority=int(goal.properties.get("priority", 50)),
                business_value=float(goal.properties.get("business_value", 0.8)),
                scope_refs=[self._kg_node_ref(goal)],
                source_refs=[self._kg_node_ref(goal)],
                projection_traces=[self._trace("goal-node", [self._kg_node_ref(goal)])],
                tags=set(goal.tags),
                properties={"kg_goal_id": goal.id},
            )
            ag.add_node(goal_node)
            self._ensure_state(
                ag,
                state_type=StateNodeType.GOAL_STATE_SATISFIED,
                label=f"Goal state satisfied: {goal.label}",
                subject_refs=[self._kg_node_ref(goal)],
                created_from=[self._kg_node_ref(goal)],
                confidence=goal.confidence,
                truth_status=self._truth_from_status(goal.status, goal.confidence),
                goal_relevance=1.0,
                properties={"goal_id": goal.id},
            )
        return goal_refs

    def _build_constraint_nodes(
        self,
        ag: AttackGraph,
        policy_context: dict[str, Any] | None,
    ) -> None:
        for raw in (policy_context or {}).get("constraints", []):
            constraint_type = ConstraintNodeType(raw["constraint_type"])
            applies_to = [GraphRef.model_validate(ref) for ref in raw.get("applies_to", [])]
            payload = {
                "constraint_type": constraint_type.value,
                "label": raw.get("label", constraint_type.value),
                "applies_to": [ref.model_dump(mode="json") for ref in applies_to],
            }
            constraint = ConstraintNode(
                id=stable_node_id("ag-constraint", payload),
                constraint_type=constraint_type,
                label=raw.get("label", constraint_type.value),
                hard_or_soft=raw.get("hard_or_soft", "hard"),
                budget_value=raw.get("budget_value"),
                current_usage=raw.get("current_usage"),
                applies_to=applies_to,
                source_refs=applies_to,
                properties=dict(raw.get("properties", {})),
                tags=set(raw.get("tags", [])),
            )
            if constraint.id not in ag._nodes:
                ag.add_node(constraint)

    def _build_state_nodes(
        self,
        kg: KnowledgeGraph,
        ag: AttackGraph,
        goal_refs: dict[str, set[str]],
    ) -> None:
        goal_related_entities = {entity_id for related in goal_refs.values() for entity_id in related}

        for host in [node for node in kg.list_nodes(type=NodeType.HOST) if isinstance(node, Host)]:
            host_ref = self._kg_node_ref(host)
            self._ensure_state(
                ag,
                state_type=StateNodeType.HOST_KNOWN,
                label=f"Host known: {host.label}",
                subject_refs=[host_ref],
                created_from=[host_ref],
                confidence=host.confidence,
                truth_status=self._truth_from_status(host.status, host.confidence),
                goal_relevance=self._goal_relevance(host.id, goal_related_entities),
                properties={"hostname": host.hostname, "platform": host.platform},
            )
            if host.status == EntityStatus.VALIDATED:
                self._ensure_state(
                    ag,
                    state_type=StateNodeType.HOST_VALIDATED,
                    label=f"Host validated: {host.label}",
                    subject_refs=[host_ref],
                    created_from=[host_ref],
                    confidence=host.confidence,
                    truth_status=TruthStatus.VALIDATED,
                    goal_relevance=self._goal_relevance(host.id, goal_related_entities),
                    properties={"hostname": host.hostname},
                )

        for service_edge in kg.list_edges(type=EdgeType.HOSTS):
            host = kg.get_node(service_edge.source)
            service = kg.get_node(service_edge.target)
            if not isinstance(host, Host) or not isinstance(service, Service):
                continue
            service_ref = self._kg_node_ref(service)
            host_ref = self._kg_node_ref(host)
            edge_ref = self._kg_edge_ref(service_edge)
            self._ensure_state(
                ag,
                state_type=StateNodeType.SERVICE_KNOWN,
                label=f"Service known: {service.label} on {host.label}",
                subject_refs=[host_ref, service_ref],
                created_from=[host_ref, service_ref, edge_ref],
                confidence=min(host.confidence, service.confidence, service_edge.confidence),
                truth_status=self._truth_from_status(service.status, service.confidence),
                goal_relevance=max(
                    self._goal_relevance(host.id, goal_related_entities),
                    self._goal_relevance(service.id, goal_related_entities),
                ),
                properties={"host_id": host.id, "service_id": service.id, "port": service.port},
            )
            if service.status == EntityStatus.VALIDATED or kg.get_supporting_evidence(service.id):
                self._ensure_state(
                    ag,
                    state_type=StateNodeType.SERVICE_CONFIRMED,
                    label=f"Service confirmed: {service.label} on {host.label}",
                    subject_refs=[host_ref, service_ref],
                    created_from=[host_ref, service_ref, edge_ref],
                    confidence=max(service.confidence, 0.85),
                    truth_status=TruthStatus.VALIDATED if service.status == EntityStatus.VALIDATED else TruthStatus.ACTIVE,
                    goal_relevance=self._goal_relevance(service.id, goal_related_entities),
                    properties={"host_id": host.id, "service_id": service.id},
                )

        for reach_edge in kg.list_edges(type=EdgeType.CAN_REACH):
            source = kg.get_node(reach_edge.source)
            target = kg.get_node(reach_edge.target)
            source_ref = self._kg_node_ref(source)
            target_ref = self._kg_node_ref(target)
            edge_ref = self._kg_edge_ref(reach_edge)
            state_type = (
                StateNodeType.REACHABILITY_VALIDATED
                if reach_edge.status == EntityStatus.VALIDATED or reach_edge.confidence >= 0.85
                else StateNodeType.PATH_CANDIDATE
            )
            truth_status = (
                TruthStatus.VALIDATED
                if state_type == StateNodeType.REACHABILITY_VALIDATED
                else self._truth_from_status(reach_edge.status, reach_edge.confidence)
            )
            self._ensure_state(
                ag,
                state_type=state_type,
                label=f"Reachability {state_type.value.lower()}: {source.label} -> {target.label}",
                subject_refs=[source_ref, target_ref],
                created_from=[source_ref, target_ref, edge_ref],
                confidence=reach_edge.confidence,
                truth_status=truth_status,
                goal_relevance=max(
                    self._goal_relevance(source.id, goal_related_entities),
                    self._goal_relevance(target.id, goal_related_entities),
                ),
                properties={"source_id": source.id, "target_id": target.id},
            )

        for identity in [node for node in kg.list_nodes(type=NodeType.IDENTITY) if isinstance(node, Identity)]:
            identity_ref = self._kg_node_ref(identity)
            self._ensure_state(
                ag,
                state_type=StateNodeType.IDENTITY_KNOWN,
                label=f"Identity known: {identity.label}",
                subject_refs=[identity_ref],
                created_from=[identity_ref],
                confidence=identity.confidence,
                truth_status=self._truth_from_status(identity.status, identity.confidence),
                goal_relevance=self._goal_relevance(identity.id, goal_related_entities),
                properties={"username": identity.username},
            )

        for auth_edge in kg.list_edges(type=EdgeType.AUTHENTICATES_AS):
            credential = kg.get_node(auth_edge.source)
            identity = kg.get_node(auth_edge.target)
            if not isinstance(credential, Credential) or not isinstance(identity, Identity):
                continue
            credential_ref = self._kg_node_ref(credential)
            identity_ref = self._kg_node_ref(identity)
            edge_ref = self._kg_edge_ref(auth_edge)
            self._ensure_state(
                ag,
                state_type=StateNodeType.CREDENTIAL_USABLE,
                label=f"Credential usable: {credential.label} as {identity.label}",
                subject_refs=[credential_ref, identity_ref],
                created_from=[credential_ref, identity_ref, edge_ref],
                confidence=min(credential.confidence, identity.confidence, auth_edge.confidence),
                truth_status=self._truth_from_status(auth_edge.status, auth_edge.confidence),
                goal_relevance=self._goal_relevance(identity.id, goal_related_entities),
                properties={"credential_id": credential.id, "identity_id": identity.id},
            )

        for availability_edge in kg.list_edges(type=EdgeType.IDENTITY_AVAILABLE_ON):
            identity = kg.get_node(availability_edge.source)
            host = kg.get_node(availability_edge.target)
            if not isinstance(identity, Identity) or not isinstance(host, Host):
                continue
            identity_ref = self._kg_node_ref(identity)
            host_ref = self._kg_node_ref(host)
            edge_ref = self._kg_edge_ref(availability_edge)
            self._ensure_state(
                ag,
                state_type=StateNodeType.IDENTITY_AVAILABLE_ON_HOST,
                label=f"Identity available on host: {identity.label} -> {host.label}",
                subject_refs=[identity_ref, host_ref],
                created_from=[identity_ref, host_ref, edge_ref],
                confidence=min(identity.confidence, host.confidence, availability_edge.confidence),
                truth_status=self._truth_from_status(availability_edge.status, availability_edge.confidence),
                goal_relevance=max(
                    self._goal_relevance(identity.id, goal_related_entities),
                    self._goal_relevance(host.id, goal_related_entities),
                ),
                properties={"identity_id": identity.id, "host_id": host.id},
            )

        for reuse_edge in kg.list_edges(type=EdgeType.REUSES_CREDENTIAL):
            credential = kg.get_node(reuse_edge.source)
            host = kg.get_node(reuse_edge.target)
            if not isinstance(credential, Credential) or not isinstance(host, Host):
                continue
            credential_ref = self._kg_node_ref(credential)
            host_ref = self._kg_node_ref(host)
            edge_ref = self._kg_edge_ref(reuse_edge)
            self._ensure_state(
                ag,
                state_type=StateNodeType.CREDENTIAL_REUSABLE_ON_HOST,
                label=f"Credential reusable on host: {credential.label} -> {host.label}",
                subject_refs=[credential_ref, host_ref],
                created_from=[credential_ref, host_ref, edge_ref],
                confidence=min(credential.confidence, host.confidence, reuse_edge.confidence),
                truth_status=self._truth_from_status(reuse_edge.status, reuse_edge.confidence),
                goal_relevance=max(
                    self._goal_relevance(credential.id, goal_related_entities),
                    self._goal_relevance(host.id, goal_related_entities),
                ),
                properties={"credential_id": credential.id, "host_id": host.id},
            )

        for session in [node for node in kg.list_nodes(type=NodeType.SESSION) if isinstance(node, Session)]:
            session_ref = self._kg_node_ref(session)
            self._ensure_state(
                ag,
                state_type=StateNodeType.MANAGED_SESSION_AVAILABLE,
                label=f"Managed session available: {session.label}",
                subject_refs=[session_ref],
                created_from=[session_ref],
                confidence=session.confidence,
                truth_status=self._truth_from_status(session.status, session.confidence),
                goal_relevance=self._goal_relevance(session.id, goal_related_entities),
                properties={"session_kind": session.session_kind},
            )

        for session_edge in kg.list_edges(type=EdgeType.SESSION_ON):
            session = kg.get_node(session_edge.source)
            host = kg.get_node(session_edge.target)
            if not isinstance(session, Session) or not isinstance(host, Host):
                continue
            session_ref = self._kg_node_ref(session)
            host_ref = self._kg_node_ref(host)
            edge_ref = self._kg_edge_ref(session_edge)
            self._ensure_state(
                ag,
                state_type=StateNodeType.SESSION_ACTIVE_ON_HOST,
                label=f"Session active on host: {session.label} -> {host.label}",
                subject_refs=[session_ref, host_ref],
                created_from=[session_ref, host_ref, edge_ref],
                confidence=min(session.confidence, host.confidence, session_edge.confidence),
                truth_status=self._truth_from_status(session_edge.status, session_edge.confidence),
                goal_relevance=max(
                    self._goal_relevance(session.id, goal_related_entities),
                    self._goal_relevance(host.id, goal_related_entities),
                ),
                properties={"session_id": session.id, "host_id": host.id},
            )

        for privilege in [node for node in kg.list_nodes(type=NodeType.PRIVILEGE_STATE) if isinstance(node, PrivilegeState)]:
            privilege_ref = self._kg_node_ref(privilege)
            self._ensure_state(
                ag,
                state_type=StateNodeType.PRIVILEGE_VALIDATED,
                label=f"Privilege validated: {privilege.label}",
                subject_refs=[privilege_ref],
                created_from=[privilege_ref],
                confidence=privilege.confidence,
                truth_status=self._truth_from_status(privilege.status, privilege.confidence),
                goal_relevance=self._goal_relevance(privilege.id, goal_related_entities),
                properties={"privilege_level": privilege.privilege_level},
            )

        for source_edge in kg.list_edges(type=EdgeType.PRIVILEGE_SOURCE):
            privilege = kg.get_node(source_edge.source)
            source_node = kg.get_node(source_edge.target)
            if not isinstance(privilege, PrivilegeState) or not isinstance(source_node, BaseNode):
                continue
            privilege_ref = self._kg_node_ref(privilege)
            source_ref = self._kg_node_ref(source_node)
            edge_ref = self._kg_edge_ref(source_edge)
            self._ensure_state(
                ag,
                state_type=StateNodeType.PRIVILEGE_SOURCE_KNOWN,
                label=f"Privilege source known: {privilege.label} via {source_node.label}",
                subject_refs=[privilege_ref, source_ref],
                created_from=[privilege_ref, source_ref, edge_ref],
                confidence=min(privilege.confidence, source_node.confidence, source_edge.confidence),
                truth_status=self._truth_from_status(source_edge.status, source_edge.confidence),
                goal_relevance=max(
                    self._goal_relevance(privilege.id, goal_related_entities),
                    self._goal_relevance(source_node.id, goal_related_entities),
                ),
                properties={"privilege_id": privilege.id, "source_id": source_node.id},
            )

        for pivot_edge in kg.list_edges(type=EdgeType.PIVOTS_TO):
            source_host = kg.get_node(pivot_edge.source)
            target_host = kg.get_node(pivot_edge.target)
            if not isinstance(source_host, Host) or not isinstance(target_host, Host):
                continue
            source_ref = self._kg_node_ref(source_host)
            target_ref = self._kg_node_ref(target_host)
            edge_ref = self._kg_edge_ref(pivot_edge)
            self._ensure_state(
                ag,
                state_type=StateNodeType.PIVOT_HOST_AVAILABLE,
                label=f"Pivot host available: {source_host.label} -> {target_host.label}",
                subject_refs=[source_ref, target_ref],
                created_from=[source_ref, target_ref, edge_ref],
                confidence=min(source_host.confidence, target_host.confidence, pivot_edge.confidence),
                truth_status=self._truth_from_status(pivot_edge.status, pivot_edge.confidence),
                goal_relevance=max(
                    self._goal_relevance(source_host.id, goal_related_entities),
                    self._goal_relevance(target_host.id, goal_related_entities),
                ),
                properties={"source_id": source_host.id, "target_id": target_host.id},
            )

        lateral_source_states = [
            *ag.find_states(StateNodeType.SERVICE_CONFIRMED),
            *ag.find_states(StateNodeType.SERVICE_KNOWN),
        ]
        for state in lateral_source_states:
            host_id = str(state.properties.get("host_id") or "")
            service_id = str(state.properties.get("service_id") or "")
            if not host_id or not service_id:
                continue
            host = kg.get_node(host_id)
            service = kg.get_node(service_id)
            if not isinstance(host, Host) or not isinstance(service, Service):
                continue
            self._ensure_state(
                ag,
                state_type=StateNodeType.LATERAL_SERVICE_EXPOSED,
                label=f"Lateral service exposed: {service.label} on {host.label}",
                subject_refs=[self._kg_node_ref(host), self._kg_node_ref(service)],
                created_from=list(state.created_from),
                confidence=state.confidence,
                truth_status=state.truth_status,
                goal_relevance=state.goal_relevance,
                properties={"host_id": host_id, "service_id": service_id},
            )

        for asset in [node for node in kg.list_nodes(type=NodeType.DATA_ASSET) if isinstance(node, DataAsset)]:
            asset_ref = self._kg_node_ref(asset)
            self._ensure_state(
                ag,
                state_type=StateNodeType.DATA_ASSET_KNOWN,
                label=f"Data asset known: {asset.label}",
                subject_refs=[asset_ref],
                created_from=[asset_ref],
                confidence=asset.confidence,
                truth_status=self._truth_from_status(asset.status, asset.confidence),
                goal_relevance=self._goal_relevance(asset.id, goal_related_entities),
                properties={"asset_kind": asset.asset_kind},
            )
            for goal_id, related_ids in goal_refs.items():
                if asset.id in related_ids:
                    goal = kg.get_node(goal_id)
                    goal_ref = self._kg_node_ref(goal)
                    self._ensure_state(
                        ag,
                        state_type=StateNodeType.GOAL_RELEVANT_DATA_LOCATED,
                        label=f"Goal-relevant data located: {asset.label}",
                        subject_refs=[asset_ref, goal_ref],
                        created_from=[asset_ref, goal_ref],
                        confidence=max(asset.confidence, 0.8),
                        truth_status=self._truth_from_status(asset.status, asset.confidence),
                        goal_relevance=1.0,
                        properties={"goal_id": goal_id, "asset_id": asset.id},
                    )

    def _build_action_nodes(self, kg: KnowledgeGraph, ag: AttackGraph) -> None:
        for state in ag.find_states(StateNodeType.HOST_KNOWN):
            host_ref = state.subject_refs[0]
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.HOST_VALIDATED,
                label=f"Host validated: {state.label}",
                subject_refs=[host_ref],
                created_from=[host_ref],
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"host_id": host_ref.ref_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.ENUMERATE_HOST,
                label=f"Enumerate host: {state.label}",
                bound_args={"host_id": host_ref.ref_id},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.15,
                risk=0.05,
                noise=0.15,
                expected_value=0.65,
                success_probability_prior=0.85,
                parallelizable=True,
                resource_keys={f"host:{host_ref.ref_id}"},
                source_refs=[host_ref],
            )

        for state in ag.find_states(StateNodeType.SERVICE_KNOWN):
            host_id = str(state.properties["host_id"])
            service_id = str(state.properties["service_id"])
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.SERVICE_CONFIRMED,
                label=f"Service confirmed: {service_id} on {host_id}",
                subject_refs=list(state.subject_refs),
                created_from=list(state.created_from),
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"host_id": host_id, "service_id": service_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.VALIDATE_SERVICE,
                label=f"Validate service: {service_id}",
                bound_args={"host_id": host_id, "service_id": service_id},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.2,
                risk=0.08,
                noise=0.2,
                expected_value=0.7,
                success_probability_prior=0.8,
                parallelizable=True,
                resource_keys={f"service:{service_id}", f"host:{host_id}"},
                source_refs=list(state.subject_refs),
            )

        for state in ag.find_states(StateNodeType.PATH_CANDIDATE):
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.REACHABILITY_VALIDATED,
                label=f"Reachability validated: {state.label}",
                subject_refs=list(state.subject_refs),
                created_from=list(state.created_from),
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties=dict(state.properties),
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.VALIDATE_REACHABILITY,
                label=f"Validate reachability: {state.label}",
                bound_args=dict(state.properties),
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.25,
                risk=0.05,
                noise=0.1,
                expected_value=0.75,
                success_probability_prior=0.78,
                parallelizable=True,
                resource_keys={f"path:{state.properties['source_id']}->{state.properties['target_id']}"},
                source_refs=list(state.subject_refs),
            )

        for state in ag.find_states(StateNodeType.IDENTITY_KNOWN):
            identity_ref = state.subject_refs[0]
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.IDENTITY_CONTEXT_KNOWN,
                label=f"Identity context known: {state.label}",
                subject_refs=[identity_ref],
                created_from=[identity_ref],
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"identity_id": identity_ref.ref_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.ENUMERATE_IDENTITY_CONTEXT,
                label=f"Enumerate identity context: {state.label}",
                bound_args={"identity_id": identity_ref.ref_id},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.12,
                risk=0.04,
                noise=0.08,
                expected_value=0.55,
                success_probability_prior=0.85,
                parallelizable=True,
                resource_keys={f"identity:{identity_ref.ref_id}"},
                source_refs=[identity_ref],
            )

        for state in ag.find_states(StateNodeType.HOST_VALIDATED):
            host_ref = state.subject_refs[0]
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.MANAGED_SESSION_AVAILABLE,
                label=f"Managed session available on {host_ref.ref_id}",
                subject_refs=[host_ref],
                created_from=[host_ref],
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"host_id": host_ref.ref_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.ESTABLISH_MANAGED_SESSION,
                label=f"Establish managed session: {host_ref.ref_id}",
                bound_args={"host_id": host_ref.ref_id},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.4,
                risk=0.18,
                noise=0.15,
                expected_value=0.7,
                success_probability_prior=0.6,
                parallelizable=False,
                approval_required=True,
                resource_keys={f"host:{host_ref.ref_id}"},
                source_refs=[host_ref],
            )

        for state in ag.find_states(StateNodeType.MANAGED_SESSION_AVAILABLE):
            session_subjects = list(state.subject_refs)
            target_privilege = self._ensure_state(
                ag,
                state_type=StateNodeType.PRIVILEGE_VALIDATED,
                label=f"Privilege validated from {state.label}",
                subject_refs=session_subjects,
                created_from=session_subjects,
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"session_source": state.id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.VALIDATE_PRIVILEGE_STATE,
                label=f"Validate privilege state: {state.label}",
                bound_args={"session_source": state.id},
                required_states=[state],
                produced_states=[target_privilege],
                goal_relevance=state.goal_relevance,
                cost=0.35,
                risk=0.15,
                noise=0.1,
                expected_value=0.78,
                success_probability_prior=0.7,
                parallelizable=False,
                resource_keys={f"session:{state.id}"},
                source_refs=session_subjects,
            )

        for state in ag.find_states(StateNodeType.PIVOT_HOST_AVAILABLE):
            source_host = str(state.properties.get("source_id") or "")
            target_host = str(state.properties.get("target_id") or "")
            if not source_host or not target_host:
                continue
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.REACHABILITY_VALIDATED,
                label=f"Reachability validated via pivot: {source_host} -> {target_host}",
                subject_refs=list(state.subject_refs),
                created_from=list(state.created_from),
                confidence=max(state.confidence, 0.8),
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"source_id": source_host, "target_id": target_host, "via_pivot": True},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.ESTABLISH_PIVOT_ROUTE,
                label=f"Establish pivot route: {source_host} -> {target_host}",
                bound_args={"source_host": source_host, "target_host": target_host},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.45,
                risk=0.28,
                noise=0.18,
                expected_value=0.8,
                success_probability_prior=0.62,
                parallelizable=False,
                resource_keys={f"host:{source_host}", f"host:{target_host}", f"pivot:{source_host}->{target_host}"},
                source_refs=list(state.subject_refs),
            )

        for state in ag.find_states(StateNodeType.CREDENTIAL_REUSABLE_ON_HOST):
            host_id = str(state.properties.get("host_id") or "")
            credential_id = str(state.properties.get("credential_id") or "")
            if not host_id or not credential_id:
                continue
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.SESSION_ACTIVE_ON_HOST,
                label=f"Session active on host via credential: {credential_id} -> {host_id}",
                subject_refs=list(state.subject_refs),
                created_from=list(state.created_from),
                confidence=max(state.confidence, 0.75),
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"host_id": host_id, "credential_id": credential_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.REUSE_CREDENTIAL_ON_HOST,
                label=f"Reuse credential on host: {credential_id} -> {host_id}",
                bound_args={"credential_id": credential_id, "host_id": host_id},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.4,
                risk=0.24,
                noise=0.16,
                expected_value=0.76,
                success_probability_prior=0.65,
                parallelizable=False,
                resource_keys={f"credential:{credential_id}", f"host:{host_id}"},
                source_refs=list(state.subject_refs),
            )

        for state in ag.find_states(StateNodeType.LATERAL_SERVICE_EXPOSED):
            host_id = str(state.properties.get("host_id") or "")
            service_id = str(state.properties.get("service_id") or "")
            if not host_id or not service_id:
                continue
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.SESSION_ACTIVE_ON_HOST,
                label=f"Session active after lateral service access: {service_id} on {host_id}",
                subject_refs=list(state.subject_refs),
                created_from=list(state.created_from),
                confidence=max(state.confidence, 0.72),
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"host_id": host_id, "service_id": service_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.EXPLOIT_LATERAL_SERVICE,
                label=f"Exploit lateral service: {service_id} on {host_id}",
                bound_args={"host_id": host_id, "service_id": service_id},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=state.goal_relevance,
                cost=0.5,
                risk=0.35,
                noise=0.3,
                expected_value=0.7,
                success_probability_prior=0.58,
                parallelizable=False,
                resource_keys={f"host:{host_id}", f"service:{service_id}"},
                source_refs=list(state.subject_refs),
            )

        for state in ag.find_states(StateNodeType.SESSION_ACTIVE_ON_HOST):
            host_id = str(state.properties.get("host_id") or "")
            if not host_id:
                continue
            identity_context = self._ensure_state(
                ag,
                state_type=StateNodeType.IDENTITY_CONTEXT_KNOWN,
                label=f"Identity context known on host: {host_id}",
                subject_refs=list(state.subject_refs),
                created_from=list(state.created_from),
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=state.goal_relevance,
                properties={"host_id": host_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.ENUMERATE_IDENTITY_CONTEXT,
                label=f"Enumerate identity context on host: {host_id}",
                bound_args={"host_id": host_id},
                required_states=[state],
                produced_states=[identity_context],
                goal_relevance=state.goal_relevance,
                cost=0.22,
                risk=0.08,
                noise=0.08,
                expected_value=0.74,
                success_probability_prior=0.8,
                parallelizable=False,
                resource_keys={f"host:{host_id}", f"session:{state.id}"},
                source_refs=list(state.subject_refs),
            )

        for state in ag.find_states(StateNodeType.DATA_ASSET_KNOWN):
            if state.goal_relevance <= 0:
                continue
            asset_id = str(state.subject_refs[0].ref_id)
            produced = self._ensure_state(
                ag,
                state_type=StateNodeType.GOAL_RELEVANT_DATA_LOCATED,
                label=f"Goal-relevant data located: {state.label}",
                subject_refs=list(state.subject_refs),
                created_from=list(state.created_from),
                confidence=state.confidence,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=max(state.goal_relevance, 0.9),
                properties={"asset_id": asset_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.LOCATE_GOAL_RELEVANT_DATA,
                label=f"Locate goal-relevant data: {asset_id}",
                bound_args={"asset_id": asset_id},
                required_states=[state],
                produced_states=[produced],
                goal_relevance=max(state.goal_relevance, 0.9),
                cost=0.22,
                risk=0.06,
                noise=0.1,
                expected_value=0.82,
                success_probability_prior=0.8,
                parallelizable=True,
                resource_keys={f"asset:{asset_id}"},
                source_refs=list(state.subject_refs),
            )

        for goal_node in ag.get_goal_nodes():
            goal_ref = goal_node.scope_refs[0]
            prerequisite_states = [
                state
                for state in ag.find_states(active_only=False)
                if state.node_type == StateNodeType.GOAL_RELEVANT_DATA_LOCATED
                and any(ref.ref_id == goal_ref.ref_id for ref in state.subject_refs)
            ]
            target_state = self._ensure_state(
                ag,
                state_type=StateNodeType.GOAL_STATE_SATISFIED,
                label=f"Goal state satisfied: {goal_node.label}",
                subject_refs=[goal_ref],
                created_from=[goal_ref],
                confidence=goal_node.business_value,
                truth_status=TruthStatus.CANDIDATE,
                goal_relevance=1.0,
                properties={"goal_id": goal_ref.ref_id},
            )
            self._bind_action(
                ag,
                action_type=ActionNodeType.VALIDATE_GOAL_CONDITION,
                label=f"Validate goal condition: {goal_node.label}",
                bound_args={"goal_id": goal_ref.ref_id},
                required_states=prerequisite_states or [target_state],
                produced_states=[target_state],
                goal_relevance=1.0,
                cost=0.1,
                risk=0.03,
                noise=0.03,
                expected_value=goal_node.business_value,
                success_probability_prior=0.9,
                parallelizable=True,
                resource_keys={f"goal:{goal_ref.ref_id}"},
                source_refs=[goal_ref],
            )

    def _connect_enablement(self, ag: AttackGraph) -> None:
        requires = {}
        for edge in ag.list_edges(AGEdgeType.REQUIRES):
            requires.setdefault(edge.source, set()).add(edge.target)
        for produced_edge in ag.list_edges(AGEdgeType.PRODUCES):
            state_id = produced_edge.target
            for action_id in requires.get(state_id, set()):
                self._add_edge(
                    ag,
                    edge_type=AGEdgeType.ENABLES,
                    source=state_id,
                    target=action_id,
                    label="enables",
                )

        for goal_node in ag.get_goal_nodes():
            goal_ref = goal_node.scope_refs[0]
            for state in ag.find_states():
                if state.node_type not in {
                    StateNodeType.GOAL_RELEVANT_DATA_LOCATED,
                    StateNodeType.GOAL_STATE_SATISFIED,
                }:
                    continue
                if any(ref.ref_id == goal_ref.ref_id for ref in state.subject_refs):
                    self._add_edge(
                        ag,
                        edge_type=AGEdgeType.ENABLES,
                        source=state.id,
                        target=goal_node.id,
                        label="enables_goal",
                    )

    def _connect_constraints(self, ag: AttackGraph) -> None:
        constraints = ag.get_constraint_nodes()
        for action in ag.find_actions():
            for constraint in constraints:
                if self._constraint_applies(action, constraint):
                    self._add_edge(
                        ag,
                        edge_type=AGEdgeType.BLOCKED_BY,
                        source=action.id,
                        target=constraint.id,
                        label="blocked_by",
                    )

    def _refresh_goal_state_nodes(self, ag: AttackGraph) -> None:
        for goal_node in ag.get_goal_nodes():
            target_states = [
                state
                for state in ag.find_states(StateNodeType.GOAL_RELEVANT_DATA_LOCATED)
                if any(ref.ref_id == goal_node.scope_refs[0].ref_id for ref in state.subject_refs)
            ]
            goal_state = next(
                (
                    state
                    for state in ag.find_states(StateNodeType.GOAL_STATE_SATISFIED)
                    if state.properties.get("goal_id") == goal_node.scope_refs[0].ref_id
                ),
                None,
            )
            if goal_state is None:
                continue
            if target_states:
                goal_state.truth_status = TruthStatus.ACTIVE
                goal_state.confidence = max(goal_state.confidence, max(state.confidence for state in target_states))

    def _refresh_action_activation(self, ag: AttackGraph) -> None:
        for action in ag.find_actions():
            required_state_ids = [
                edge.source
                for edge in ag.list_edges(AGEdgeType.REQUIRES)
                if edge.target == action.id
            ]
            produced_state_ids = [
                edge.target
                for edge in ag.list_edges(AGEdgeType.PRODUCES)
                if edge.source == action.id
            ]
            blocked = any(
                self._constraint_is_currently_blocking(ag.get_node(edge.target))
                for edge in ag.list_edges(AGEdgeType.BLOCKED_BY)
                if edge.source == action.id and isinstance(ag.get_node(edge.target), ConstraintNode)
            )
            all_required_active = all(
                isinstance(ag.get_node(state_id), StateNode)
                and ag.get_node(state_id).truth_status in {TruthStatus.ACTIVE, TruthStatus.VALIDATED}
                for state_id in required_state_ids
            )
            all_outputs_ready = produced_state_ids and all(
                isinstance(ag.get_node(state_id), StateNode)
                and ag.get_node(state_id).truth_status in {TruthStatus.ACTIVE, TruthStatus.VALIDATED}
                for state_id in produced_state_ids
            )
            if blocked:
                action.activation_status = ActivationStatus.BLOCKED
            elif all_outputs_ready:
                action.activation_status = ActivationStatus.SATISFIED
            elif all_required_active or not required_state_ids:
                action.activation_status = ActivationStatus.ACTIVATABLE
            else:
                action.activation_status = ActivationStatus.DORMANT
            action.activation_conditions = [
                ActivationCondition(
                    key="required_states",
                    required_refs=[GraphRef(graph="ag", ref_id=state_id, ref_type="StateNode") for state_id in required_state_ids],
                    status=action.activation_status,
                )
            ]

    def _bind_action(
        self,
        ag: AttackGraph,
        action_type: ActionNodeType,
        label: str,
        bound_args: dict[str, Any],
        required_states: list[StateNode],
        produced_states: list[StateNode],
        goal_relevance: float,
        cost: float,
        risk: float,
        noise: float,
        expected_value: float,
        success_probability_prior: float,
        parallelizable: bool,
        resource_keys: set[str],
        source_refs: list[GraphRef],
        approval_required: bool = False,
    ) -> ActionNode:
        action = ActionNode(
            id=stable_node_id(
                "ag-action",
                {"action_type": action_type.value, "bound_args": bound_args},
            ),
            action_type=action_type,
            label=label,
            bound_args=bound_args,
            required_inputs=sorted(bound_args.keys()),
            precondition_schema={"state_types": sorted({state.node_type.value for state in required_states})},
            postcondition_schema={"state_types": sorted({state.node_type.value for state in produced_states})},
            required_capabilities={"planner-managed"},
            cost=cost,
            risk=risk,
            noise=noise,
            expected_value=expected_value,
            success_probability_prior=success_probability_prior,
            goal_relevance=goal_relevance,
            parallelizable=parallelizable,
            cooldown_seconds=0,
            retry_policy={"max_retries": 1},
            approval_required=approval_required,
            resource_keys=resource_keys,
            source_refs=source_refs,
            projection_traces=[self._trace(f"action:{action_type.value.lower()}", source_refs)],
            tags={action_type.value.lower()},
        )
        if action.id not in ag._nodes:
            ag.add_node(action)
        else:
            action = ag.get_node(action.id)  # type: ignore[assignment]
        for state in required_states:
            self._add_edge(ag, AGEdgeType.REQUIRES, state.id, action.id, "requires")
        for state in produced_states:
            self._add_edge(ag, AGEdgeType.PRODUCES, action.id, state.id, "produces")
        return action

    def _ensure_state(
        self,
        ag: AttackGraph,
        state_type: StateNodeType,
        label: str,
        subject_refs: list[GraphRef],
        created_from: list[GraphRef],
        confidence: float,
        truth_status: TruthStatus,
        goal_relevance: float,
        properties: dict[str, Any],
    ) -> StateNode:
        state_id = stable_node_id(
            "ag-state",
            {
                "state_type": state_type.value,
                "subject_refs": sorted(ref.key() for ref in subject_refs),
                "properties": properties,
            },
        )
        if state_id in ag._nodes:
            existing = ag.get_node(state_id)
            assert isinstance(existing, StateNode)
            if self._truth_rank(truth_status) > self._truth_rank(existing.truth_status):
                existing.truth_status = truth_status
            existing.confidence = max(existing.confidence, confidence)
            existing.goal_relevance = max(existing.goal_relevance, goal_relevance)
            return existing

        state = StateNode(
            id=state_id,
            node_type=state_type,
            label=label,
            subject_refs=subject_refs,
            properties=properties,
            truth_status=truth_status,
            confidence=confidence,
            goal_relevance=goal_relevance,
            created_from=created_from,
            source_refs=created_from,
            projection_traces=[self._trace(f"state:{state_type.value.lower()}", created_from)],
            tags={state_type.value.lower()},
        )
        ag.add_node(state)
        return state

    def _add_edge(
        self,
        ag: AttackGraph,
        edge_type: AGEdgeType,
        source: str,
        target: str,
        label: str,
    ) -> BaseAGEdge:
        edge_id = stable_node_id(
            "ag-edge",
            {"edge_type": edge_type.value, "source": source, "target": target},
        )
        if edge_id in ag._edges:
            return ag.get_edge(edge_id)
        edge = BaseAGEdge(
            id=edge_id,
            edge_type=edge_type,
            source=source,
            target=target,
            label=label,
        )
        ag.add_edge(edge)
        return edge

    def _constraint_applies(self, action: ActionNode, constraint: ConstraintNode) -> bool:
        if constraint.constraint_type == ConstraintNodeType.APPROVAL_GATE:
            return action.approval_required
        if not constraint.applies_to:
            return constraint.constraint_type in {
                ConstraintNodeType.CONCURRENCY_LIMIT,
                ConstraintNodeType.NOISE_BUDGET,
                ConstraintNodeType.RISK_BUDGET,
                ConstraintNodeType.TIME_BUDGET,
                ConstraintNodeType.TOKEN_BUDGET,
            }
        for ref in constraint.applies_to:
            if ref.graph == "kg" and any(source_ref.ref_id == ref.ref_id for source_ref in action.source_refs):
                return True
            if ref.graph == "ag" and ref.ref_id in action.resource_keys:
                return True
        return False

    @staticmethod
    def _constraint_is_currently_blocking(node: ConstraintNode) -> bool:
        if node.constraint_type == ConstraintNodeType.APPROVAL_GATE:
            return not bool(node.properties.get("approved", False))
        if node.budget_value is None or node.current_usage is None:
            return node.hard_or_soft == "hard"
        return float(node.current_usage) >= float(node.budget_value)

    @staticmethod
    def _goal_type_for_goal(goal: Goal) -> GoalNodeType:
        category = str(goal.category or goal.properties.get("category", "")).lower()
        if "profile" in category:
            return GoalNodeType.HOST_PROFILE_SUFFICIENT
        if "context" in category:
            return GoalNodeType.TARGET_CONTEXT_VALIDATED
        if "data" in category:
            return GoalNodeType.GOAL_RELEVANT_DATA_PRESENT
        return GoalNodeType.OBJECTIVE_SATISFIED

    @staticmethod
    def _goal_relevance(entity_id: str, goal_related_entities: set[str]) -> float:
        return 1.0 if entity_id in goal_related_entities else 0.2

    @staticmethod
    def _truth_from_status(status: EntityStatus, confidence: float) -> TruthStatus:
        if status == EntityStatus.REVOKED:
            return TruthStatus.REVOKED
        if status == EntityStatus.STALE:
            return TruthStatus.STALE
        if status == EntityStatus.VALIDATED:
            return TruthStatus.VALIDATED
        if confidence >= 0.7:
            return TruthStatus.ACTIVE
        return TruthStatus.CANDIDATE

    @staticmethod
    def _truth_rank(status: TruthStatus) -> int:
        order = {
            TruthStatus.CANDIDATE: 0,
            TruthStatus.ACTIVE: 1,
            TruthStatus.VALIDATED: 2,
            TruthStatus.STALE: -1,
            TruthStatus.REVOKED: -2,
        }
        return order[status]

    @staticmethod
    def _kg_node_ref(node: BaseNode) -> GraphRef:
        return GraphRef(graph="kg", ref_id=node.id, ref_type=node.type.value, label=node.label)

    @staticmethod
    def _kg_edge_ref(edge: BaseEdge) -> GraphRef:
        return GraphRef(graph="kg", ref_id=edge.id, ref_type=edge.type.value, label=edge.label)

    @staticmethod
    def _trace(rule: str, refs: list[GraphRef]) -> ProjectionTrace:
        return ProjectionTrace(rule=rule, input_refs=refs)

    def _collect_affected_node_ids(self, ag: AttackGraph, changed_refs: list[str]) -> set[str]:
        changed = set(changed_refs)
        affected = {
            node.id
            for node in ag.list_nodes()
            if any(ref.ref_id in changed for ref in AttackGraph._refs_for_index(node))
        }
        closure = set(affected)
        for edge in ag.list_edges():
            if edge.source in affected or edge.target in affected:
                closure.add(edge.source)
                closure.add(edge.target)
        return closure
