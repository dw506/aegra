"""Phase-two result applier for worker-originated execution results.

This module consumes compatibility-layer ``AgentTaskResult`` objects and routes
their side effects through the existing ownership boundaries:

- observations / evidence -> State Writer -> KG state deltas
- fact write requests -> KG structural state deltas
- projection requests -> Graph Projection
- runtime requests / hints -> Runtime managers and runtime event queue

Workers remain unable to write KG / AG / TG stores directly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.agent_models import EvidenceRecord, ObservationRecord, StateDeltaRecord
from src.core.agents.agent_protocol import (
    AgentContext,
    AgentExecutionResult,
    AgentInput,
    AgentKind,
    AgentOutput,
    GraphRef,
    GraphScope,
)
from src.core.agents.graph_projection import GraphProjectionAgent
from src.core.agents.kg_events import KGDeltaEvent, KGDeltaEventType, KGEventBatch
from src.core.agents.state_writer import StateWriterAgent
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.tg_builder import AttackGraphTaskBuilder, TaskCandidate, TaskGenerationRequest, TaskGenerationResult, TaskGraphBuilder
from src.core.graph.tg_merge import merge_task_graphs
from src.core.models.ag import ActionNode, AttackGraph
from src.core.models.events import (
    AgentResultAdapter,
    AgentResultStatus,
    AgentTaskResult,
    FactWriteKind,
    FactWriteRequest,
    ProjectionRequest,
    RuntimeControlRequest,
    RuntimeControlType,
)
from src.core.models.finding import EvidenceArtifactRecord, Finding
from src.core.models.runtime import OutcomeCacheEntry, ReplanRequest, RuntimeEventRef, RuntimeState, TaskRuntimeStatus, WorkerStatus
from src.core.models.runtime import utc_now
from src.core.models.tg import BaseTaskNode, TaskGraph, TaskStatus
from src.core.runtime.budgets import RuntimeBudgetManager
from src.core.runtime.checkpoint_store import RuntimeCheckpointManager
from src.core.runtime.credential_manager import RuntimeCredentialManager
from src.core.runtime.events import (
    BudgetConsumedEvent,
    CheckpointCreatedEvent,
    LockAcquiredEvent,
    LockReleasedEvent,
    ReplanRequestedEvent,
    SessionExpiredEvent,
    SessionHeartbeatEvent,
    SessionOpenedEvent,
    event_to_ref,
)
from src.core.runtime.lease_manager import RuntimeLeaseManager
from src.core.runtime.locks import RuntimeLockManager
from src.core.runtime.observability import append_audit_log
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager
from src.core.runtime.reachability import ReachabilityPropagator
from src.core.runtime.risk_scoring import RiskScorer
from src.core.runtime.session_manager import RuntimeSessionManager


class PhaseTwoApplyResult(BaseModel):
    """Structured summary of one phase-two result application pass."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    runtime_event_refs: list[RuntimeEventRef] = Field(default_factory=list)
    kg_state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    ag_state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    kg_apply_result: dict[str, Any] | None = None
    ag_graph: dict[str, Any] | None = None
    tg_graph: dict[str, Any] | None = None
    tg_task_candidates: list[dict[str, Any]] = Field(default_factory=list)
    tg_created_task_ids: list[str] = Field(default_factory=list)
    kg_event_batch: KGEventBatch | None = None
    state_writer_result: AgentExecutionResult | None = None
    graph_projection_result: AgentExecutionResult | None = None
    logs: list[str] = Field(default_factory=list)


class PhaseTwoResultApplier:
    """Apply one worker result using existing phase-two owners and managers."""

    def __init__(
        self,
        *,
        state_writer: StateWriterAgent | None = None,
        graph_projection: GraphProjectionAgent | None = None,
        session_manager: RuntimeSessionManager | None = None,
        credential_manager: RuntimeCredentialManager | None = None,
        lease_manager: RuntimeLeaseManager | None = None,
        pivot_route_manager: RuntimePivotRouteManager | None = None,
        budget_manager: RuntimeBudgetManager | None = None,
        checkpoint_manager: RuntimeCheckpointManager | None = None,
        lock_manager: RuntimeLockManager | None = None,
        attack_graph_projector: AttackGraphProjector | None = None,
        task_graph_builder: AttackGraphTaskBuilder | None = None,
        reachability_propagator: ReachabilityPropagator | None = None,
    ) -> None:
        self._state_writer = state_writer or StateWriterAgent()
        self._graph_projection = graph_projection or GraphProjectionAgent()
        self._attack_graph_projector = attack_graph_projector or AttackGraphProjector()
        self._task_graph_builder = task_graph_builder or AttackGraphTaskBuilder()
        self._session_manager = session_manager or RuntimeSessionManager()
        self._credential_manager = credential_manager or RuntimeCredentialManager()
        self._lease_manager = lease_manager or RuntimeLeaseManager()
        self._pivot_route_manager = pivot_route_manager or RuntimePivotRouteManager()
        self._reachability_propagator = reachability_propagator or ReachabilityPropagator(self._pivot_route_manager)
        self._budget_manager = budget_manager or RuntimeBudgetManager()
        self._checkpoint_manager = checkpoint_manager or RuntimeCheckpointManager()
        self._lock_manager = lock_manager or RuntimeLockManager()

    def apply(
        self,
        result: AgentTaskResult | AgentExecutionResult | AgentOutput,
        state: RuntimeState,
        *,
        agent_input: AgentInput | None = None,
        agent_name: str | None = None,
        agent_kind: AgentKind | None = None,
        kg_ref: GraphRef | None = None,
        kg_store: KnowledgeGraph | None = None,
        attack_graph: AttackGraph | None = None,
        task_graph: TaskGraph | None = None,
        goal_context: dict[str, Any] | None = None,
        policy_context: dict[str, Any] | None = None,
    ) -> PhaseTwoApplyResult:
        """Apply one worker result through runtime, KG and AG ownership paths.

        中文注释：
        这里统一接受 worker 的多种运行时返回形态，但会在入口立即收敛为
        `AgentTaskResult`。这样 result applier 内部不再关心 `AgentOutput`
        还是 `AgentExecutionResult`，只处理 canonical result。
        """

        canonical_result = AgentResultAdapter.to_task_result(
            result,
            agent_input=agent_input,
            agent_name=agent_name,
            agent_kind=agent_kind,
        )

        if canonical_result.operation_id != state.operation_id:
            raise ValueError("result.operation_id must match RuntimeState.operation_id")

        resolved_kg_ref = kg_ref or self._resolve_kg_ref(canonical_result)
        apply_result = PhaseTwoApplyResult()

        runtime_event_refs = self._apply_runtime_effects(result=canonical_result, state=state)
        apply_result.runtime_event_refs.extend(runtime_event_refs)
        self._sync_runtime_views_from_result(state=state, result=canonical_result)
        self._apply_task_lifecycle(state=state, result=canonical_result)
        if task_graph is not None and self._sync_task_graph_lifecycle(task_graph=task_graph, result=canonical_result):
            apply_result.tg_graph = task_graph.to_dict()
            apply_result.logs.append(f"synced TG task {canonical_result.tg_node_id} lifecycle from runtime result")
        self._record_recent_outcome(state=state, result=canonical_result)
        self._audit_tool_invocations(state=state, result=canonical_result)
        self._audit_tool_execution_from_result(state=state, result=canonical_result)
        self._record_evidence_and_findings(state=state, result=canonical_result)

        state_writer_input: AgentInput | None = None
        state_writer_result, state_writer_input = self._run_state_writer(result=canonical_result, state=state, kg_ref=resolved_kg_ref)
        if state_writer_result is not None:
            apply_result.state_writer_result = state_writer_result
            apply_result.kg_state_deltas.extend(state_writer_result.output.state_deltas)
            apply_result.logs.extend(state_writer_result.output.logs)

        fact_deltas = self._fact_state_deltas(result=canonical_result)
        apply_result.kg_state_deltas.extend(fact_deltas)
        if fact_deltas:
            self._audit_fact_writes(state=state, result=canonical_result)
            apply_result.logs.append(f"converted {len(fact_deltas)} fact write request(s) into KG state delta(s)")

        if apply_result.kg_state_deltas:
            apply_result.kg_state_deltas = self._order_kg_state_deltas(apply_result.kg_state_deltas)
            if kg_store is not None:
                apply_result.kg_apply_result = self._apply_kg_deltas_to_store(
                    kg_store=kg_store,
                    kg_ref=resolved_kg_ref,
                    state=state,
                    state_deltas=apply_result.kg_state_deltas,
                    state_writer_input=state_writer_input,
                )
                apply_result.logs.append(
                    f"applied {len(apply_result.kg_state_deltas)} KG state delta(s) to KG version "
                    f"{apply_result.kg_apply_result['kg_version']}"
                )
            batch = self._kg_batch(state_deltas=apply_result.kg_state_deltas)
            apply_result.kg_event_batch = batch
            apply_result.logs.append(f"built KG event batch with {len(batch.events)} event(s)")

            if canonical_result.projection_requests:
                projection_result = self._run_graph_projection(
                    state=state,
                    kg_ref=resolved_kg_ref,
                    batch=batch,
                    goal_context=goal_context,
                    policy_context=policy_context,
                    projection_requests=canonical_result.projection_requests,
                )
                apply_result.graph_projection_result = projection_result
                apply_result.ag_state_deltas.extend(projection_result.output.state_deltas)
                apply_result.logs.extend(projection_result.output.logs)

            if kg_store is not None:
                projected_ag = self._project_attack_graph(
                    kg_store=kg_store,
                    existing_graph=attack_graph,
                    state_deltas=apply_result.kg_state_deltas,
                    goal_context=goal_context,
                    policy_context=policy_context,
                )
                apply_result.ag_graph = projected_ag.to_dict()
                apply_result.logs.append(
                    f"re-projected AG from KG version {kg_store.version} into AG version {projected_ag.version}"
                )
                generated_tg = self._generate_candidate_task_graph(projected_ag)
                apply_result.tg_task_candidates = [
                    candidate.model_dump(mode="json") for candidate in generated_tg.candidates
                ]
                apply_result.tg_created_task_ids = list(generated_tg.created_task_ids)
                if generated_tg.task_graph is not None and generated_tg.candidates:
                    generated_task_graph = TaskGraph.from_dict(generated_tg.task_graph)
                    merged_task_graph = merge_task_graphs(task_graph, generated_task_graph)
                    self._sync_task_graph_lifecycle(task_graph=merged_task_graph, result=canonical_result)
                    apply_result.tg_graph = merged_task_graph.to_dict()
                    apply_result.logs.append(
                        f"generated {len(apply_result.tg_task_candidates)} TG candidate task(s) from updated AG"
                    )
                elif task_graph is not None:
                    apply_result.tg_graph = task_graph.to_dict()
                    apply_result.logs.append("updated AG produced no activatable TG candidates; kept existing TG snapshot")
                else:
                    apply_result.logs.append("updated AG produced no activatable TG candidates")

        worker_generated_tg = self._generate_worker_candidate_task_graph(canonical_result)
        if worker_generated_tg.task_graph is not None and worker_generated_tg.candidates:
            existing_task_graph = TaskGraph.from_dict(apply_result.tg_graph) if apply_result.tg_graph is not None else task_graph
            generated_task_graph = TaskGraph.from_dict(worker_generated_tg.task_graph)
            merged_task_graph = merge_task_graphs(existing_task_graph, generated_task_graph)
            self._sync_task_graph_lifecycle(task_graph=merged_task_graph, result=canonical_result)
            apply_result.tg_graph = merged_task_graph.to_dict()
            apply_result.tg_task_candidates.extend(
                candidate.model_dump(mode="json") for candidate in worker_generated_tg.candidates
            )
            apply_result.tg_created_task_ids.extend(worker_generated_tg.created_task_ids)
            apply_result.logs.append(
                f"generated {len(worker_generated_tg.candidates)} TG candidate task(s) from worker result"
            )

        return apply_result

    @staticmethod
    def _order_kg_state_deltas(state_deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply KG entity upserts before relations that may reference them."""

        def rank(delta: dict[str, Any]) -> tuple[int, int]:
            delta_type = str(delta.get("delta_type") or "")
            patch_kind = str((delta.get("payload") or {}).get("patch_kind") or "")
            if delta_type == "upsert_entity" or patch_kind == "entity":
                return (0, 0)
            if delta_type == "upsert_relation" or patch_kind == "relation":
                return (2, 0)
            return (1, 0)

        return sorted(list(state_deltas), key=rank)

    def _run_state_writer(
        self,
        *,
        result: AgentTaskResult,
        state: RuntimeState,
        kg_ref: GraphRef,
    ) -> tuple[AgentExecutionResult | None, AgentInput | None]:
        observations = [
            ObservationRecord(
                id=record.observation_id,
                source_agent=result.agent_role.value,
                confidence=record.confidence,
                summary=record.summary,
                refs=[self._to_protocol_ref(ref) for ref in record.refs],
                payload=dict(record.payload),
            )
            for record in result.observations
        ]
        evidence = [
            EvidenceRecord(
                id=record.evidence_id,
                source_agent=result.agent_role.value,
                confidence=1.0,
                summary=record.summary,
                refs=[self._to_protocol_ref(ref) for ref in record.refs],
                payload=dict(record.metadata),
                payload_ref=record.payload_ref,
            )
            for record in result.evidence
        ]
        if not observations and not evidence:
            return None, None
        agent_input = AgentInput(
            graph_refs=[kg_ref],
            task_ref=result.task_id,
            context=AgentContext(
                operation_id=state.operation_id,
                runtime_state_ref=state.operation_id,
            ),
            raw_payload={
                "observations": [item.model_dump(mode="json") for item in observations],
                "evidences": [item.model_dump(mode="json") for item in evidence],
            },
        )
        return self._state_writer.run(agent_input), agent_input

    def _apply_kg_deltas_to_store(
        self,
        *,
        kg_store: KnowledgeGraph,
        kg_ref: GraphRef,
        state: RuntimeState,
        state_deltas: list[dict[str, Any]],
        state_writer_input: AgentInput | None,
    ) -> dict[str, Any]:
        agent_input = state_writer_input or AgentInput(
            graph_refs=[kg_ref],
            context=AgentContext(
                operation_id=state.operation_id,
                runtime_state_ref=state.operation_id,
            ),
            raw_payload={"kg_version": kg_store.version},
        )
        apply_request = self._state_writer.build_store_apply_request(
            kg_ref=kg_ref,
            state_deltas=state_deltas,
            agent_input=agent_input,
            base_kg_version=kg_store.version,
        )
        return self._state_writer.apply_to_store(store=kg_store, apply_request=apply_request)

    def _project_attack_graph(
        self,
        *,
        kg_store: KnowledgeGraph,
        existing_graph: AttackGraph | None,
        state_deltas: list[dict[str, Any]],
        goal_context: dict[str, Any] | None,
        policy_context: dict[str, Any] | None,
    ) -> AttackGraph:
        changed_refs = [
            GraphRef.model_validate(delta["target_ref"]).ref_id
            for delta in state_deltas
            if isinstance(delta, dict) and isinstance(delta.get("target_ref"), dict)
        ]
        return self._attack_graph_projector.project_incremental(
            kg_store,
            existing_graph=existing_graph,
            changed_refs=changed_refs,
            goal_context=goal_context,
            policy_context=policy_context,
        )

    def _generate_candidate_task_graph(self, attack_graph: AttackGraph) -> TaskGenerationResult:
        action_ids = [
            action.id
            for action in attack_graph.find_actions(activatable_only=True)
            if isinstance(action, ActionNode)
        ]
        if not action_ids:
            action_ids = [action.id for action in attack_graph.find_actions() if isinstance(action, ActionNode)]
        return self._task_graph_builder.build_candidates(
            attack_graph,
            TaskGenerationRequest(
                action_ids=sorted(action_ids),
                include_evidence_tasks=False,
                group_label="Updated Candidate Tasks",
            ),
        )

    @staticmethod
    def _generate_worker_candidate_task_graph(result: AgentTaskResult) -> TaskGenerationResult:
        raw_candidates = []
        payload_candidates = result.outcome_payload.get("task_candidates")
        if isinstance(payload_candidates, list):
            raw_candidates.extend(payload_candidates)
        nested_payload = result.outcome_payload.get("payload")
        if isinstance(nested_payload, dict) and isinstance(nested_payload.get("task_candidates"), list):
            raw_candidates.extend(nested_payload["task_candidates"])
        metadata_candidates = result.metadata.get("task_candidates")
        if isinstance(metadata_candidates, list):
            raw_candidates.extend(metadata_candidates)
        candidates: list[TaskCandidate] = []
        seen: set[str] = set()
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                continue
            try:
                candidate = TaskCandidate.model_validate(raw)
            except Exception:
                continue
            key = f"{candidate.source_action_id}:{candidate.task_type.value}:{candidate.input_bindings}"
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        if not candidates:
            return TaskGenerationResult()
        return TaskGraphBuilder().build_from_candidates(
            TaskGenerationRequest(
                candidates=candidates,
                include_evidence_tasks=False,
                group_label="Worker Proposed Tasks",
            )
        )

    def _run_graph_projection(
        self,
        *,
        state: RuntimeState,
        kg_ref: GraphRef,
        batch: KGEventBatch,
        goal_context: dict[str, Any] | None,
        policy_context: dict[str, Any] | None,
        projection_requests: list[ProjectionRequest],
    ) -> AgentExecutionResult:
        agent_input = AgentInput(
            graph_refs=[kg_ref],
            context=AgentContext(
                operation_id=state.operation_id,
                runtime_state_ref=state.operation_id,
            ),
            raw_payload={
                "kg_event_batch": batch.model_dump(mode="json"),
                "goal_context": dict(goal_context or {}),
                "policy_context": dict(policy_context or {}),
                "projection_requests": [item.model_dump(mode="json") for item in projection_requests],
            },
        )
        return self._graph_projection.run(agent_input)

    def _apply_runtime_effects(
        self,
        *,
        result: AgentTaskResult,
        state: RuntimeState,
    ) -> list[RuntimeEventRef]:
        refs: list[RuntimeEventRef] = []
        for request in result.runtime_requests:
            refs.extend(self._apply_runtime_request(state=state, request=request))
        for hint in result.checkpoint_hints:
            checkpoint = self._checkpoint_manager.create_checkpoint(
                state=state,
                checkpoint_id=hint.checkpoint_id,
                created_after_tasks=list(hint.created_after_tasks or [result.task_id]),
                summary=hint.summary,
            )
            refs.append(
                self._push_runtime_event(
                    state,
                    CheckpointCreatedEvent(
                        operation_id=state.operation_id,
                        checkpoint_id=checkpoint.checkpoint_id,
                        created_after_tasks=list(checkpoint.created_after_tasks),
                        summary=checkpoint.summary,
                    ),
                )
            )
        for hint in result.replan_hints:
            self._checkpoint_manager.add_replan_marker(
                state,
                {
                    "source_task_id": hint.source_task_id,
                    "scope": hint.scope.value,
                    "reason": hint.reason,
                    "task_ids": list(hint.task_ids),
                    "invalidated_ref_keys": list(hint.invalidated_ref_keys),
                    "metadata": dict(hint.metadata),
                },
            )
            request = ReplanRequest(
                request_id=f"replan-hint-{hint.hint_id}",
                reason=hint.reason,
                task_ids=list(hint.task_ids or [hint.source_task_id]),
                scope=hint.scope.value,
                metadata={
                    "hint_id": hint.hint_id,
                    "invalidated_ref_keys": list(hint.invalidated_ref_keys),
                    **dict(hint.metadata),
                },
            )
            state.request_replan(request)
            refs.append(
                self._push_runtime_event(
                    state,
                    ReplanRequestedEvent(
                        operation_id=state.operation_id,
                        request_id=request.request_id,
                        reason=request.reason,
                        task_ids=list(request.task_ids),
                        scope=request.scope,
                        payload={"metadata": dict(request.metadata)},
                    ),
                )
            )
        if result.critic_signals:
            bucket = state.execution.metadata.setdefault("critic_signals", [])
            bucket.extend(signal.model_dump(mode="json") for signal in result.critic_signals)
        return refs

    def _apply_runtime_request(
        self,
        *,
        state: RuntimeState,
        request: RuntimeControlRequest,
    ) -> list[RuntimeEventRef]:
        if request.request_type == RuntimeControlType.OPEN_SESSION:
            session_id = request.session_id or f"session-{request.request_id}"
            session = self._session_manager.open_session(
                state=state,
                session_id=session_id,
                bound_identity=self._string(request.metadata.get("bound_identity")),
                bound_target=self._string(request.metadata.get("bound_target")),
                lease_seconds=int(request.lease_seconds or 300),
                reusability=str(request.reuse_policy or "exclusive"),
            )
            self._session_manager.bind_task_to_session(state, request.source_task_id, session.session_id)
            lease_id = self._lease_id(session.session_id, request.source_task_id)
            lease = self._lease_manager.create_lease(
                state=state,
                lease_id=lease_id,
                session_id=session.session_id,
                owner_task_id=request.source_task_id,
                lease_seconds=int(request.lease_seconds or 300),
                reuse_policy=str(request.reuse_policy or "exclusive"),
                metadata={"created_by": "result_applier", "request_id": request.request_id},
            )
            self._lease_manager.bind_lease_to_task_or_session(
                state,
                lease.lease_id,
                task_id=request.source_task_id,
                session_id=session.session_id,
            )
            execution_endpoint = self._mapping(request.metadata.get("execution_endpoint"))
            if execution_endpoint:
                session.metadata["execution_endpoint"] = execution_endpoint
            capabilities = self._mapping(request.metadata.get("capabilities"))
            if capabilities:
                session.metadata["capabilities"] = capabilities
            return [
                self._push_runtime_event(
                    state,
                    SessionOpenedEvent(
                        operation_id=state.operation_id,
                        session_id=session.session_id,
                        bound_identity=session.bound_identity,
                        bound_target=session.bound_target,
                        lease_expiry=session.lease_expiry,
                        reusability=session.reusability,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.REGISTER_PIVOT_ROUTE:
            route_id = self._string(request.metadata.get("route_id")) or f"route-{request.request_id}"
            destination_host = self._string(request.metadata.get("destination_host") or request.metadata.get("target_host"))
            if destination_host is None:
                raise ValueError("register_pivot_route runtime request requires metadata.destination_host")
            route = self._pivot_route_manager.register_candidate(
                state,
                route_id,
                destination_host,
                source_host=self._string(request.metadata.get("source_host")),
                via_host=self._string(request.metadata.get("via_host")),
                session_id=request.session_id or self._string(request.metadata.get("session_id")),
                protocol=self._string(request.metadata.get("protocol")),
                destination_zone=self._string(request.metadata.get("destination_zone")),
                destination_cidr=self._string(request.metadata.get("destination_cidr")),
                allowed_ports=self._list_or_set(request.metadata.get("allowed_ports") or request.metadata.get("port")),
                protocols=self._list_or_set(request.metadata.get("protocols")),
                hop_count=self._int(request.metadata.get("hop_count")),
                confidence=self._float(request.metadata.get("confidence")),
                metadata={
                    "created_by": "result_applier",
                    "request_id": request.request_id,
                    "transport": self._mapping(request.metadata.get("transport")),
                },
            )
            if bool(request.metadata.get("active", False)):
                route = self._pivot_route_manager.activate_route(state, route.route_id)
            return [self._runtime_control_event(state, request, summary=f"registered pivot route {route.route_id}")]
        if request.request_type == RuntimeControlType.VERIFY_PIVOT_ROUTE:
            route_id = self._string(request.metadata.get("route_id")) or self._string(request.session_id)
            if route_id is None:
                raise ValueError("verify_pivot_route runtime request requires metadata.route_id")
            reachable = bool(request.metadata.get("reachable", True))
            if reachable:
                self._pivot_route_manager.activate_route(state, route_id)
            else:
                self._pivot_route_manager.fail_route(state, route_id, reason=request.reason or "pivot_verification_failed")
            return [self._runtime_control_event(state, request, summary=f"verified pivot route {route_id}")]
        if request.request_type == RuntimeControlType.OPEN_TUNNEL:
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                route = state.pivot_routes[route_id]
                transport = dict(route.metadata.get("transport", {})) if isinstance(route.metadata.get("transport"), dict) else {}
                transport.update(
                    {
                        "kind": "tcp_tunnel",
                        "tunnel_endpoint": self._string(request.metadata.get("tunnel_endpoint") or request.metadata.get("endpoint")),
                        "health": "ready",
                    }
                )
                route.metadata["transport"] = {key: value for key, value in transport.items() if value is not None}
            session_id = request.session_id or self._string(request.metadata.get("session_id"))
            if session_id is not None and session_id in state.sessions:
                endpoint = dict(state.sessions[session_id].metadata.get("execution_endpoint", {})) if isinstance(state.sessions[session_id].metadata.get("execution_endpoint"), dict) else {}
                endpoint.update({"kind": "tunnel", "adapter": "tcp_tunnel", "tunnel_endpoint": self._string(request.metadata.get("tunnel_endpoint") or request.metadata.get("endpoint"))})
                state.sessions[session_id].metadata["execution_endpoint"] = {key: value for key, value in endpoint.items() if value is not None}
            return [self._runtime_control_event(state, request, summary="opened tunnel")]
        if request.request_type == RuntimeControlType.CLOSE_TUNNEL:
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                route = state.pivot_routes[route_id]
                transport = dict(route.metadata.get("transport", {})) if isinstance(route.metadata.get("transport"), dict) else {}
                transport["health"] = "closed"
                route.metadata["transport"] = transport
            return [self._runtime_control_event(state, request, summary="closed tunnel")]
        if request.request_type == RuntimeControlType.ATTACH_NETWORK_NAMESPACE:
            namespace = self._string(request.metadata.get("network_namespace") or request.metadata.get("namespace"))
            if namespace is None:
                raise ValueError("attach_network_namespace runtime request requires metadata.network_namespace")
            for session_id in [request.session_id, self._string(request.metadata.get("session_id"))]:
                if session_id is not None and session_id in state.sessions:
                    endpoint = dict(state.sessions[session_id].metadata.get("execution_endpoint", {})) if isinstance(state.sessions[session_id].metadata.get("execution_endpoint"), dict) else {}
                    endpoint.update({"adapter": "netns_shell", "namespace": namespace})
                    state.sessions[session_id].metadata["execution_endpoint"] = endpoint
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                transport = dict(state.pivot_routes[route_id].metadata.get("transport", {})) if isinstance(state.pivot_routes[route_id].metadata.get("transport"), dict) else {}
                transport.update({"kind": "netns", "namespace": namespace, "health": "ready"})
                state.pivot_routes[route_id].metadata["transport"] = transport
            return [self._runtime_control_event(state, request, summary=f"attached network namespace {namespace}")]
        if request.request_type == RuntimeControlType.DETACH_NETWORK_NAMESPACE:
            route_id = self._string(request.metadata.get("route_id"))
            if route_id is not None and route_id in state.pivot_routes:
                transport = dict(state.pivot_routes[route_id].metadata.get("transport", {})) if isinstance(state.pivot_routes[route_id].metadata.get("transport"), dict) else {}
                transport.pop("namespace", None)
                if transport.get("kind") == "netns":
                    transport["health"] = "closed"
                state.pivot_routes[route_id].metadata["transport"] = transport
            return [self._runtime_control_event(state, request, summary="detached network namespace")]
        if request.request_type == RuntimeControlType.EXTEND_SESSION:
            if not request.session_id:
                raise ValueError("extend_session runtime request requires session_id")
            session = self._session_manager.extend_lease(
                state=state,
                session_id=request.session_id,
                extra_seconds=int(request.lease_seconds or 60),
            )
            for lease in self._lease_manager.list_leases_for_session(state, request.session_id, active_only=True):
                self._lease_manager.extend_lease(
                    state,
                    lease.lease_id,
                    extra_seconds=int(request.lease_seconds or 60),
                )
            return [
                self._push_runtime_event(
                    state,
                    SessionHeartbeatEvent(
                        operation_id=state.operation_id,
                        session_id=session.session_id,
                        heartbeat_at=session.heartbeat_at,
                        lease_expiry=session.lease_expiry,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.EXPIRE_SESSION:
            if not request.session_id:
                raise ValueError("expire_session runtime request requires session_id")
            session = self._session_manager.expire_session(
                state=state,
                session_id=request.session_id,
                reason=request.reason,
            )
            self._cleanup_session_family(
                state,
                session_id=session.session_id,
                reason=request.reason or "runtime_request_expire_session",
                close_routes=False,
            )
            return [
                self._push_runtime_event(
                    state,
                    SessionExpiredEvent(
                        operation_id=state.operation_id,
                        session_id=session.session_id,
                        reason=request.reason,
                        failure_count=session.failure_count,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.ACQUIRE_LOCKS:
            owner_type = str(request.metadata.get("owner_type", "task"))
            owner_id = self._string(request.metadata.get("owner_id")) or request.source_task_id
            results = self._lock_manager.acquire_many(
                state=state,
                lock_keys=list(request.lock_keys),
                owner_type=owner_type,
                owner_id=owner_id,
                ttl_seconds=request.lease_seconds,
            )
            return [
                self._push_runtime_event(
                    state,
                    LockAcquiredEvent(
                        operation_id=state.operation_id,
                        lock_key=item.lock_key,
                        owner_type=item.owner_type,
                        owner_id=item.owner_id,
                        expires_at=item.expires_at,
                    ),
                )
                for item in results
            ]
        if request.request_type == RuntimeControlType.RELEASE_LOCKS:
            owner_id = self._string(request.metadata.get("owner_id"))
            refs: list[RuntimeEventRef] = []
            for lock_key in request.lock_keys:
                if self._lock_manager.release_lock(state=state, lock_key=lock_key, owner_id=owner_id):
                    refs.append(
                        self._push_runtime_event(
                            state,
                            LockReleasedEvent(
                                operation_id=state.operation_id,
                                lock_key=lock_key,
                                owner_id=owner_id or request.source_task_id,
                                reason=request.reason,
                            ),
                        )
                    )
            return refs
        if request.request_type == RuntimeControlType.CONSUME_BUDGET:
            delta = request.budget_delta
            if delta.time_sec:
                self._budget_manager.consume_time(state, delta.time_sec)
            if delta.tokens:
                self._budget_manager.consume_tokens(state, delta.tokens)
            if delta.operations:
                self._budget_manager.consume_operations(state, delta.operations)
            if delta.noise:
                self._budget_manager.consume_noise(state, delta.noise)
            if delta.risk:
                self._budget_manager.consume_risk(state, delta.risk)
            approval_updates = {
                str(key): bool(value)
                for key, value in dict(request.metadata.get("approval_updates", {})).items()
            }
            for key, value in approval_updates.items():
                self._budget_manager.set_approval(state, key, value)
            policy_updates = {
                str(key): value
                for key, value in dict(request.metadata.get("policy_flag_updates", {})).items()
            }
            for key, value in policy_updates.items():
                self._budget_manager.set_policy_flag(state, key, value)
            return [
                self._push_runtime_event(
                    state,
                    BudgetConsumedEvent(
                        operation_id=state.operation_id,
                        time_budget_used_sec_delta=delta.time_sec,
                        token_budget_used_delta=delta.tokens,
                        operation_budget_used_delta=delta.operations,
                        noise_budget_used_delta=delta.noise,
                        risk_budget_used_delta=delta.risk,
                        approval_updates=approval_updates,
                        policy_flag_updates=policy_updates,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.CREATE_CHECKPOINT:
            checkpoint_id = request.checkpoint_id or f"checkpoint-{request.request_id}"
            checkpoint = self._checkpoint_manager.create_checkpoint(
                state=state,
                checkpoint_id=checkpoint_id,
                created_after_tasks=[request.source_task_id],
                summary=request.reason,
            )
            return [
                self._push_runtime_event(
                    state,
                    CheckpointCreatedEvent(
                        operation_id=state.operation_id,
                        checkpoint_id=checkpoint.checkpoint_id,
                        created_after_tasks=list(checkpoint.created_after_tasks),
                        summary=checkpoint.summary,
                    ),
                )
            ]
        if request.request_type == RuntimeControlType.REQUEST_REPLAN:
            replan_request = ReplanRequest(
                request_id=f"runtime-replan-{request.request_id}",
                reason=request.reason or "runtime control requested replanning",
                task_ids=[request.source_task_id],
                scope=str(request.metadata.get("scope", "local")),
                metadata=dict(request.metadata),
            )
            state.request_replan(replan_request)
            self._checkpoint_manager.add_replan_marker(
                state,
                {
                    "request_id": replan_request.request_id,
                    "reason": replan_request.reason,
                    "scope": replan_request.scope,
                    "task_ids": list(replan_request.task_ids),
                },
            )
            return [
                self._push_runtime_event(
                    state,
                    ReplanRequestedEvent(
                        operation_id=state.operation_id,
                        request_id=replan_request.request_id,
                        reason=replan_request.reason,
                        task_ids=list(replan_request.task_ids),
                        scope=replan_request.scope,
                        payload={"metadata": dict(replan_request.metadata)},
                    ),
                )
            ]
        raise ValueError(f"unsupported runtime request type: {request.request_type.value}")

    # 中文注释：
    # worker 只给出结果和线索，真正写回 RuntimeState 的生命周期转移统一在这里完成。
    def _apply_task_lifecycle(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        task = state.execution.tasks.get(result.task_id)
        if task is None:
            return
        now = utc_now()
        runtime_setup_block = self._is_runtime_setup_block(result)
        if not runtime_setup_block:
            task.attempt_count = min(task.attempt_count + 1, task.max_attempts)
        task.last_outcome_ref = f"runtime://results/{result.result_id}"
        task.metadata["last_result_status"] = result.status.value
        task.metadata["last_result_summary"] = result.summary
        task.metadata["last_result_at"] = now.isoformat()
        if result.error_message is not None:
            task.last_error = result.error_message
        elif result.status.value in {"failed", "blocked", "needs_replan"}:
            task.last_error = result.summary
        else:
            task.last_error = None

        if runtime_setup_block:
            task.status = TaskRuntimeStatus.PENDING
            task.finished_at = None
            task.started_at = None
            task.deadline = None
            task.metadata["runtime_blocked_reason"] = self._string(result.outcome_payload.get("blocked_on")) or result.summary
        elif result.status.value == "succeeded":
            task.status = TaskRuntimeStatus.SUCCEEDED
            task.finished_at = now
            task.metadata.pop("runtime_blocked_reason", None)
        elif result.status.value == "noop":
            task.status = TaskRuntimeStatus.SKIPPED
            task.finished_at = now
            task.metadata.pop("runtime_blocked_reason", None)
        elif result.status.value == "needs_replan":
            task.status = TaskRuntimeStatus.BLOCKED
            task.finished_at = now
            task.metadata["requires_replan"] = True
        elif result.status.value == "blocked":
            task.status = TaskRuntimeStatus.BLOCKED
            task.finished_at = now
            task.metadata["runtime_blocked_reason"] = self._string(result.outcome_payload.get("blocked_on")) or result.summary
        else:
            task.status = TaskRuntimeStatus.FAILED
            task.finished_at = now

        self._release_task_execution_resources(state=state, task_id=result.task_id)
        if runtime_setup_block:
            return

        lease_cleanup_reason = result.error_message or result.summary or f"task_{task.status.value}"
        self._lease_manager.release_leases_for_task(state, result.task_id, reason=lease_cleanup_reason)
        session_id = self._result_session_id(result=result, task=task)
        if task.status == TaskRuntimeStatus.SUCCEEDED:
            self._close_single_use_session(state=state, session_id=session_id, task_id=result.task_id)
            return
        if task.status in {TaskRuntimeStatus.FAILED, TaskRuntimeStatus.TIMED_OUT, TaskRuntimeStatus.BLOCKED}:
            if session_id is not None and (
                task.status == TaskRuntimeStatus.FAILED
                or result.status.value == "needs_replan"
                or self._string(result.outcome_payload.get("blocked_on")) == "reachability"
            ):
                self._session_manager.fail_session(
                    state,
                    session_id,
                    reason=result.error_message or result.summary,
                )
                self._cleanup_session_family(
                    state,
                    session_id=session_id,
                    reason=result.error_message or result.summary or f"task_{task.status.value}",
                    close_routes=False,
                )

    def _sync_task_graph_lifecycle(self, *, task_graph: TaskGraph, result: AgentTaskResult) -> bool:
        """Mirror successful Runtime task completion into the Task Graph."""

        if result.status != AgentResultStatus.SUCCEEDED:
            return False
        for task_id in (result.tg_node_id, result.task_id):
            if task_id not in task_graph._nodes:
                continue
            node = task_graph.get_node(task_id)
            if not isinstance(node, BaseTaskNode):
                continue
            updated = task_graph.mark_task_status(task_id, TaskStatus.SUCCEEDED, reason=result.summary)
            updated.last_outcome_ref = f"runtime://results/{result.result_id}"
            return True
        return False

    def _sync_runtime_views_from_result(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        self._sync_credential_view(state=state, result=result)
        self._sync_pivot_route_view(state=state, result=result)

    def _record_evidence_and_findings(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        normalized_evidence = self._normalize_evidence_records(result)
        if normalized_evidence:
            bucket = state.execution.metadata.setdefault("evidence_artifacts", [])
            existing_indexes = {
                str(item.get("evidence_id")): index
                for index, item in enumerate(bucket)
                if isinstance(item, dict) and item.get("evidence_id")
            }
            for evidence in normalized_evidence:
                payload = evidence.model_dump(mode="json")
                index = existing_indexes.get(evidence.evidence_id)
                if index is None:
                    existing_indexes[evidence.evidence_id] = len(bucket)
                    bucket.append(payload)
                else:
                    bucket[index] = payload

        validation = self._dict(result.outcome_payload.get("validation"))
        if not validation and result.outcome_payload.get("outcome_type") == "vulnerability_validation":
            validation = dict(result.outcome_payload)
        status = self._string(validation.get("status"))
        if status == "blocked":
            self._record_finding_audit(
                state=state,
                result=result,
                event_type="validation_blocked",
                validation=validation,
                reason=self._string(validation.get("failure_reason")) or result.error_message or result.summary,
            )
            return
        if status == "not_detected":
            return
        if status not in {"validated", "suspected"}:
            return

        finding = self._finding_from_validation(
            state=state,
            result=result,
            validation=validation,
            evidence_refs=[item.evidence_id for item in normalized_evidence],
        )
        bucket = state.execution.metadata.setdefault("findings", [])
        existing_index = next(
            (
                index
                for index, item in enumerate(bucket)
                if isinstance(item, dict) and item.get("finding_id") == finding.finding_id
            ),
            None,
        )
        payload = finding.model_dump(mode="json")
        if existing_index is None:
            bucket.append(payload)
            event_type = "finding_created"
        else:
            original_created_at = bucket[existing_index].get("created_at") if isinstance(bucket[existing_index], dict) else None
            if original_created_at:
                payload["created_at"] = original_created_at
            bucket[existing_index] = payload
            event_type = "finding_updated"
        self._record_finding_audit(
            state=state,
            result=result,
            event_type=event_type,
            validation=validation,
            finding_id=finding.finding_id,
            evidence_refs=list(finding.evidence_refs),
        )

    def _normalize_evidence_records(self, result: AgentTaskResult) -> list[EvidenceArtifactRecord]:
        records: list[EvidenceArtifactRecord] = []
        for evidence in result.evidence:
            records.append(
                EvidenceArtifactRecord(
                    evidence_id=evidence.evidence_id,
                    kind=evidence.kind,
                    summary=evidence.summary,
                    payload_ref=evidence.payload_ref,
                    task_ref=result.task_id,
                    tool_output_ref=evidence.tool_output_ref or evidence.payload_ref,
                    refs=[ref.model_dump(mode="json") for ref in evidence.refs],
                    metadata={
                        **dict(evidence.metadata),
                        "operation_id": result.operation_id,
                        "worker_result_id": result.result_id,
                        "source_agent": result.agent_role.value,
                        "tg_node_id": result.tg_node_id,
                        "timestamp": evidence.created_at.isoformat(),
                    },
                    created_at=evidence.created_at,
                )
            )
        return records

    def _finding_from_validation(
        self,
        *,
        state: RuntimeState,
        result: AgentTaskResult,
        validation: dict[str, Any],
        evidence_refs: list[str],
    ) -> Finding:
        service_ref = self._string(result.outcome_payload.get("service_id")) or self._ref_id_for_type(result, "Service") or "unknown-service"
        vulnerability_ref = (
            self._string(result.outcome_payload.get("vulnerability_id"))
            or self._string(validation.get("vulnerability_id"))
            or self._ref_id_for_type(result, "Vulnerability")
            or "unknown-vulnerability"
        )
        title = self._string(validation.get("vulnerability_name")) or self._string(validation.get("summary")) or vulnerability_ref
        cvss = self._coerce_float(validation.get("cvss"))
        epss = self._coerce_float(validation.get("epss"))
        kev = bool(validation.get("kev", False))
        confidence = self._coerce_float(result.outcome_payload.get("confidence")) or self._coerce_float(validation.get("confidence")) or 0.5
        requires_auth = bool(
            validation.get("requires_auth")
            or validation.get("requires_authentication")
            or result.outcome_payload.get("requires_auth")
        )
        public_exposed = self._is_publicly_exposed(state=state, service_ref=service_ref, validation=validation)
        critical_asset = self._is_critical_asset(state=state, service_ref=service_ref, validation=validation)
        validated = str(validation.get("status")) == "validated"
        risk_score = RiskScorer.score(
            cvss=cvss,
            epss=epss,
            kev=kev,
            public_exposed=public_exposed,
            validated=validated,
            requires_auth=requires_auth,
            critical_asset=critical_asset,
            confidence=confidence,
        )
        remediation = self._string(validation.get("remediation")) or self._default_remediation(validation)
        return Finding(
            finding_id=f"finding::{vulnerability_ref}::{service_ref}",
            title=title,
            affected_asset_refs=self._affected_asset_refs(state=state, service_ref=service_ref, result=result),
            service_ref=service_ref,
            vulnerability_ref=vulnerability_ref,
            evidence_refs=evidence_refs,
            validation_status=str(validation.get("status")),  # type: ignore[arg-type]
            severity=risk_score.severity,
            cvss=cvss,
            epss=epss,
            kev=kev,
            confidence=confidence,
            false_positive_risk=round(1.0 - confidence, 3),
            remediation=remediation,
            risk_score=risk_score,
            provenance={
                "operation_id": state.operation_id,
                "task_ref": result.task_id,
                "tg_node_id": result.tg_node_id,
                "worker_result_id": result.result_id,
                "tool_output_refs": [item.payload_ref for item in result.evidence],
                "timestamp": result.created_at.isoformat(),
            },
        )

    def _record_finding_audit(self, *, state: RuntimeState, result: AgentTaskResult, event_type: str, validation: dict[str, Any], **payload: Any) -> None:
        entry = {
            "event_type": event_type,
            "at": utc_now().isoformat(),
            "source_task_id": result.task_id,
            "worker_result_id": result.result_id,
            "validation_status": validation.get("status"),
            "vulnerability_id": validation.get("vulnerability_id") or result.outcome_payload.get("vulnerability_id"),
            "service_id": result.outcome_payload.get("service_id"),
            **payload,
        }
        finding_audit = state.execution.metadata.setdefault("finding_audit", [])
        finding_audit.append(dict(entry))
        self._append_audit_log(state, entry)

    def _sync_credential_view(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        raw = self._dict(result.outcome_payload.get("credential_validation"))
        credential_id = self._string(raw.get("credential_id"))
        if credential_id is None:
            return
        principal = self._string(raw.get("principal")) or self._string(raw.get("username")) or "unknown-principal"
        if credential_id not in state.credentials:
            self._credential_manager.upsert_credential(
                state,
                credential_id,
                principal,
                kind=str(raw.get("kind", "password")),
                secret_ref=self._string(raw.get("secret_ref")),
                source_session_id=self._result_session_id(result=result, task=state.execution.tasks.get(result.task_id)),
                metadata={"created_by": "result_applier"},
            )
        status = self._string(raw.get("status")) or ("valid" if bool(raw.get("validated")) else "unknown")
        target_id = self._string(raw.get("target_id")) or self._string(raw.get("bound_target"))
        self._credential_manager.record_validation(
            state,
            credential_id,
            status=status,
            target_id=target_id,
            metadata={"validator_output": raw, "source_task_id": result.task_id},
        )

    def _sync_pivot_route_view(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        self._reachability_propagator.sync_from_task_result(state=state, result=result)

    def _release_task_execution_resources(self, *, state: RuntimeState, task_id: str) -> None:
        task = state.execution.tasks.get(task_id)
        if task is None:
            return
        self._lock_manager.release_all_for_owner(state, task_id)
        worker_id = task.assigned_worker
        if worker_id is not None and worker_id in state.workers:
            worker = state.workers[worker_id]
            worker.current_task_id = None
            if worker.status not in {WorkerStatus.LOST, WorkerStatus.UNAVAILABLE}:
                worker.status = WorkerStatus.IDLE
        task.assigned_worker = None

    def _cleanup_session_family(
        self,
        state: RuntimeState,
        *,
        session_id: str,
        reason: str,
        close_routes: bool,
    ) -> None:
        self._lease_manager.release_leases_for_session(state, session_id, reason=reason)
        self._credential_manager.expire_credentials_for_session(state, session_id, reason=reason)
        if close_routes:
            self._pivot_route_manager.close_routes_for_session(state, session_id, reason=reason)
        else:
            self._pivot_route_manager.fail_routes_for_session(state, session_id, reason=reason)

    def _close_single_use_session(self, *, state: RuntimeState, session_id: str | None, task_id: str) -> None:
        if session_id is None or session_id not in state.sessions:
            return
        session = state.sessions[session_id]
        if session.reusability != "single_use":
            self._session_manager.unbind_task_from_session(state, task_id, session_id)
            return
        self._session_manager.close_session(state, session_id, reason="single_use_session_completed")
        self._cleanup_session_family(
            state,
            session_id=session_id,
            reason="single_use_session_completed",
            close_routes=True,
        )

    @staticmethod
    def _is_runtime_setup_block(result: AgentTaskResult) -> bool:
        if result.status.value != "blocked":
            return False
        return any(request.request_type == RuntimeControlType.OPEN_SESSION for request in result.runtime_requests)

    @staticmethod
    def _lease_id(session_id: str, task_id: str) -> str:
        return f"lease::{session_id}::{task_id}"

    @staticmethod
    def _ref_id_for_type(result: AgentTaskResult, ref_type: str) -> str | None:
        expected = ref_type.lower()
        for collection in (result.observations, result.evidence):
            for item in collection:
                for ref in getattr(item, "refs", []):
                    if str(getattr(ref, "ref_type", "")).lower() == expected:
                        return str(getattr(ref, "ref_id"))
        for request in result.fact_write_requests:
            for ref in (request.subject_ref, request.object_ref):
                if ref is not None and str(getattr(ref, "ref_type", "")).lower() == expected:
                    return str(getattr(ref, "ref_id"))
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _default_remediation(validation: dict[str, Any]) -> str:
        cve = validation.get("cve")
        if cve:
            return f"Review vendor guidance for {cve}, apply the relevant patch, and validate compensating controls."
        return "Review vendor guidance, apply the relevant patch, and validate compensating controls."

    def _affected_asset_refs(self, *, state: RuntimeState, service_ref: str, result: AgentTaskResult) -> list[str]:
        refs = [service_ref]
        for ref_type in ("Host", "DataAsset"):
            ref_id = self._ref_id_for_type(result, ref_type)
            if ref_id is not None:
                refs.append(ref_id)
        for target in self._target_inventory(state):
            target_id = self._string(target.get("asset_id")) or self._string(target.get("value")) or self._string(target.get("address"))
            if target_id and (target_id in service_ref or service_ref in target_id):
                refs.append(target_id)
        return list(dict.fromkeys(refs))

    def _is_publicly_exposed(self, *, state: RuntimeState, service_ref: str, validation: dict[str, Any]) -> bool:
        explicit = validation.get("public_exposed")
        if explicit is not None:
            return bool(explicit)
        for target in self._target_inventory(state):
            text = " ".join(str(target.get(key, "")) for key in ("value", "address", "hostname", "url"))
            if text and (text in service_ref or service_ref in text):
                tags = {str(item).lower() for item in target.get("tags", []) if item is not None}
                metadata = self._dict(target.get("metadata"))
                return bool(metadata.get("public_exposed") or metadata.get("internet_exposed") or tags & {"public", "internet", "external"})
        return False

    def _is_critical_asset(self, *, state: RuntimeState, service_ref: str, validation: dict[str, Any]) -> bool:
        explicit = validation.get("critical_asset")
        if explicit is not None:
            return bool(explicit)
        for target in self._target_inventory(state):
            text = " ".join(str(target.get(key, "")) for key in ("value", "address", "hostname", "url"))
            if text and (text in service_ref or service_ref in text):
                tags = {str(item).lower() for item in target.get("tags", []) if item is not None}
                metadata = self._dict(target.get("metadata"))
                return bool(metadata.get("critical_asset") or metadata.get("business_critical") or tags & {"critical", "crown-jewel"})
        return False

    @staticmethod
    def _target_inventory(state: RuntimeState) -> list[dict[str, Any]]:
        value = state.execution.metadata.get("target_inventory", [])
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, dict)]

    @staticmethod
    def _dict(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _result_session_id(self, *, result: AgentTaskResult, task: Any) -> str | None:
        for request in result.runtime_requests:
            if request.request_type in {RuntimeControlType.OPEN_SESSION, RuntimeControlType.EXTEND_SESSION, RuntimeControlType.EXPIRE_SESSION}:
                if request.session_id:
                    return request.session_id
        if task is not None:
            session_id = self._string(task.metadata.get("session_id"))
            if session_id is not None:
                return session_id
        return self._string(result.outcome_payload.get("session_id"))

    @staticmethod
    def _result_target_host_id(result: AgentTaskResult) -> str | None:
        for collection in (result.observations, result.evidence):
            for item in collection:
                for ref in getattr(item, "refs", []):
                    if str(getattr(ref, "ref_type", "")).lower() == "host":
                        return str(getattr(ref, "ref_id"))
        for request in result.fact_write_requests:
            for ref in (request.subject_ref, request.object_ref):
                if ref is not None and str(getattr(ref, "ref_type", "")).lower() == "host":
                    return str(getattr(ref, "ref_id"))
        return None

    def _fact_state_deltas(self, *, result: AgentTaskResult) -> list[dict[str, Any]]:
        deltas: list[dict[str, Any]] = []
        for request in result.fact_write_requests:
            if request.kind == FactWriteKind.RELATION_UPSERT or (
                request.object_ref is not None and request.relation_type
            ):
                deltas.append(self._relation_delta(result=result, request=request))
            else:
                deltas.append(self._entity_delta(result=result, request=request))
        return deltas

    def _entity_delta(self, *, result: AgentTaskResult, request: FactWriteRequest) -> dict[str, Any]:
        if request.subject_ref is None:
            raise ValueError("entity/assertion fact writes require subject_ref")
        subject_ref = self._to_protocol_ref(request.subject_ref)
        evidence_chain = self._evidence_chain(result=result, request=request)
        patch = {
            "patch_id": request.proposal_id,
            "entity_id": subject_ref.ref_id,
            "operation": "upsert",
            "entity_kind": "node",
            "entity_type": subject_ref.ref_type or "entity",
            "label": request.summary,
            "attributes": {
                **dict(request.attributes),
                "confidence": request.confidence,
                "evidence_ids": list(request.evidence_ids),
                "evidence_chain": evidence_chain,
                "source_task_id": request.source_task_id,
                "fact_kind": request.kind.value,
            },
            "source_refs": [subject_ref.model_dump(mode="json")],
            "provenance": {
                "worker_result_id": result.result_id,
                "source_task_id": request.source_task_id,
                "source_agent": result.agent_role.value,
                "evidence_chain": evidence_chain,
            },
        }
        return (
            StateDeltaRecord(
                id=f"delta-{request.proposal_id}",
                source_agent=result.agent_role.value,
                summary=request.summary,
                confidence=request.confidence,
                refs=[subject_ref],
                payload={"patch_kind": "entity", "fact_kind": request.kind.value},
                graph_scope=GraphScope.KG,
                delta_type="upsert_entity",
                target_ref=subject_ref,
                patch=patch,
            ).to_agent_output_fragment()
            | {"write_type": "structural"}
        )

    def _relation_delta(self, *, result: AgentTaskResult, request: FactWriteRequest) -> dict[str, Any]:
        if request.subject_ref is None or request.object_ref is None or not request.relation_type:
            raise ValueError("relation fact writes require subject_ref, object_ref and relation_type")
        source_ref = self._to_protocol_ref(request.subject_ref)
        object_ref = self._to_protocol_ref(request.object_ref)
        relation_id = f"{request.relation_type.lower()}::{source_ref.ref_id}::{object_ref.ref_id}"
        evidence_chain = self._evidence_chain(result=result, request=request)
        patch = {
            "patch_id": request.proposal_id,
            "relation_id": relation_id,
            "operation": "upsert",
            "entity_kind": "edge",
            "relation_type": request.relation_type,
            "source": source_ref.ref_id,
            "target": object_ref.ref_id,
            "label": request.summary,
            "attributes": {
                **dict(request.attributes),
                "confidence": request.confidence,
                "evidence_ids": list(request.evidence_ids),
                "evidence_chain": evidence_chain,
                "source_task_id": request.source_task_id,
                "fact_kind": request.kind.value,
            },
            "source_refs": [
                source_ref.model_dump(mode="json"),
                object_ref.model_dump(mode="json"),
            ],
            "provenance": {
                "worker_result_id": result.result_id,
                "source_task_id": request.source_task_id,
                "source_agent": result.agent_role.value,
                "evidence_chain": evidence_chain,
            },
        }
        target_ref = GraphRef(graph=GraphScope.KG, ref_id=relation_id, ref_type=request.relation_type)
        return (
            StateDeltaRecord(
                id=f"delta-{request.proposal_id}",
                source_agent=result.agent_role.value,
                summary=request.summary,
                confidence=request.confidence,
                refs=[source_ref, object_ref],
                payload={"patch_kind": "relation", "fact_kind": request.kind.value},
                graph_scope=GraphScope.KG,
                delta_type="upsert_relation",
                target_ref=target_ref,
                patch=patch,
            ).to_agent_output_fragment()
            | {"write_type": "structural"}
        )

    def _kg_batch(self, *, state_deltas: list[dict[str, Any]]) -> KGEventBatch:
        events: list[KGDeltaEvent] = []
        for delta in state_deltas:
            target_ref = GraphRef.model_validate(delta["target_ref"])
            patch = dict(delta.get("patch", {}))
            delta_type = str(delta.get("delta_type", ""))
            events.append(
                KGDeltaEvent(
                    event_type=self._kg_event_type(delta_type=delta_type, patch=patch),
                    source_agent=str(delta.get("source_agent", "phase_two_result_applier")),
                    target_ref=target_ref,
                    patch=patch,
                    metadata={"state_delta_id": delta.get("id")},
                )
            )
        return KGEventBatch.from_events(events, metadata={"source": self.__class__.__name__})

    # 中文注释：
    # 所有结构化事实写入都带上证据链，便于后续审计、回放和人工复核。
    @staticmethod
    def _evidence_chain(*, result: AgentTaskResult, request: FactWriteRequest) -> dict[str, Any]:
        return {
            "evidence_ids": list(request.evidence_ids),
            "worker_result_id": result.result_id,
            "source_task_id": request.source_task_id,
            "source_agent": result.agent_role.value,
            "summary": request.summary,
        }

    # 中文注释：
    # 工具调用审计统一写入 runtime metadata，不让 worker 直接操作审计存储。
    def _audit_tool_invocations(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for evidence in result.evidence:
            metadata = dict(evidence.metadata)
            tool = metadata.get("tool")
            command = metadata.get("command") or metadata.get("tool_command")
            adapter = metadata.get("adapter")
            if tool is None and command is None and adapter is None:
                continue
            self._append_audit_log(
                state,
                {
                    "event_type": "tool_invocation",
                    "source_task_id": result.task_id,
                    "worker_result_id": result.result_id,
                    "evidence_id": evidence.evidence_id,
                    "tool": tool,
                    "command": command,
                    "adapter": adapter,
                    "payload_ref": evidence.payload_ref,
                },
            )

    def _audit_tool_execution_from_result(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        seen: set[tuple[tuple[str, str], ...]] = set()
        for tool_execution in self._tool_execution_candidates(result):
            normalized = self._normalize_tool_execution(tool_execution)
            if not normalized:
                continue
            fingerprint = tuple(sorted((key, str(value)) for key, value in normalized.items() if value is not None))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            success = bool(normalized.get("success", False))
            self._append_audit_log(
                state,
                {
                    "event_type": "tool_execution_recorded" if success else "tool_execution_failed",
                    "source_task_id": result.task_id,
                    "worker_result_id": result.result_id,
                    "adapter": normalized.get("adapter"),
                    "tool": normalized.get("tool"),
                    "success": success,
                    "exit_code": normalized.get("exit_code"),
                    "command_id": normalized.get("command_id"),
                    "payload_ref": normalized.get("payload_ref"),
                    "stdout_excerpt": normalized.get("stdout_excerpt"),
                    "stderr_excerpt": normalized.get("stderr_excerpt"),
                },
            )

    @staticmethod
    def _tool_execution_candidates(result: AgentTaskResult) -> list[Any]:
        candidates: list[Any] = [
            result.outcome_payload.get("tool_execution"),
            result.metadata.get("tool_execution"),
        ]
        for evidence in result.evidence:
            candidates.append(evidence.metadata.get("tool_execution"))
        return [candidate for candidate in candidates if candidate is not None]

    @classmethod
    def _normalize_tool_execution(cls, tool_execution: Any) -> dict[str, Any]:
        if isinstance(tool_execution, BaseModel):
            data = tool_execution.model_dump(mode="json")
        elif isinstance(tool_execution, dict):
            data = dict(tool_execution)
        else:
            return {}
        return {
            "adapter": cls._string(data.get("adapter")) or "unknown_adapter",
            "tool": cls._string(data.get("tool")) or "unknown_tool",
            "success": bool(data.get("success", False)),
            "exit_code": data.get("exit_code"),
            "command_id": cls._string(data.get("command_id")),
            "payload_ref": cls._string(data.get("payload_ref")),
            "stdout_excerpt": cls._excerpt(data.get("stdout")),
            "stderr_excerpt": cls._excerpt(data.get("stderr")),
        }

    @staticmethod
    def _excerpt(value: Any, limit: int = 500) -> str:
        if value is None:
            return ""
        return str(value)[:limit]

    def _audit_fact_writes(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        for request in result.fact_write_requests:
            evidence_chain = self._evidence_chain(result=result, request=request)
            self._append_audit_log(
                state,
                {
                    "event_type": "fact_write",
                    "source_task_id": request.source_task_id,
                    "worker_result_id": result.result_id,
                    "proposal_id": request.proposal_id,
                    "fact_kind": request.kind.value,
                    "summary": request.summary,
                    "evidence_chain": evidence_chain,
                },
            )
            self._append_audit_log(
                state,
                {
                    "event_type": "evidence_chain",
                    "source_task_id": request.source_task_id,
                    "worker_result_id": result.result_id,
                    "proposal_id": request.proposal_id,
                    "subject_ref": request.subject_ref.model_dump(mode="json") if request.subject_ref is not None else None,
                    "object_ref": request.object_ref.model_dump(mode="json") if request.object_ref is not None else None,
                    "evidence_chain": evidence_chain,
                },
            )

    def _record_recent_outcome(self, *, state: RuntimeState, result: AgentTaskResult) -> None:
        payload_ref = self._result_payload_ref(result)
        state.record_outcome(
            OutcomeCacheEntry(
                outcome_id=self._string(result.outcome_payload.get("id")) or f"outcome::{result.task_id}",
                task_id=result.task_id,
                outcome_type=self._string(result.outcome_payload.get("outcome_type"))
                or self._string(result.outcome_payload.get("kind"))
                or "worker_outcome",
                summary=self._string(result.outcome_payload.get("summary")) or result.summary,
                payload_ref=payload_ref,
                metadata={
                    "status": result.status.value,
                    "source_agent": result.agent_role.value,
                    "worker_result_id": result.result_id,
                },
            )
        )

    @staticmethod
    def _kg_event_type(*, delta_type: str, patch: dict[str, Any]) -> KGDeltaEventType:
        operation = str(patch.get("operation") or "upsert").lower()
        if delta_type == "upsert_relation":
            return KGDeltaEventType.RELATION_UPDATED if operation == "update" else KGDeltaEventType.RELATION_ADDED
        if delta_type == "upsert_entity":
            return KGDeltaEventType.ENTITY_UPDATED if operation == "update" else KGDeltaEventType.ENTITY_ADDED
        if "confidence" in patch:
            return KGDeltaEventType.CONFIDENCE_CHANGED
        return KGDeltaEventType.ENTITY_UPDATED

    @staticmethod
    def _resolve_kg_ref(result: AgentTaskResult) -> GraphRef:
        for collection in (result.observations, result.evidence, result.fact_write_requests):
            for item in collection:
                refs = getattr(item, "refs", None)
                if refs is None and isinstance(item, FactWriteRequest):
                    refs = [ref for ref in (item.subject_ref, item.object_ref) if ref is not None]
                for ref in refs or []:
                    graph = getattr(ref, "graph", None)
                    graph_value = graph.value if hasattr(graph, "value") else str(graph or "").lower()
                    if graph_value == GraphScope.KG.value:
                        return PhaseTwoResultApplier._to_protocol_ref(ref)
        return GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph")

    @staticmethod
    def _to_protocol_ref(ref: Any) -> GraphRef:
        graph = getattr(ref, "graph", None)
        graph_value = graph.value if hasattr(graph, "value") else str(graph or "").lower()
        metadata = dict(getattr(ref, "metadata", {}) or {})
        label = getattr(ref, "label", None)
        if label is not None and "label" not in metadata:
            metadata["label"] = label
        return GraphRef(
            graph=GraphScope(graph_value),
            ref_id=str(getattr(ref, "ref_id")),
            ref_type=getattr(ref, "ref_type", None),
            metadata=metadata,
        )

    @staticmethod
    def _push_runtime_event(state: RuntimeState, event: Any) -> RuntimeEventRef:
        ref = event_to_ref(event, cursor=state.event_cursor + 1)
        state.push_event(ref)
        return ref

    @staticmethod
    def _append_audit_log(state: RuntimeState, entry: dict[str, Any]) -> None:
        append_audit_log(state, entry)

    def _runtime_control_event(
        self,
        state: RuntimeState,
        request: RuntimeControlRequest,
        *,
        summary: str,
    ) -> RuntimeEventRef:
        self._append_audit_log(
            state,
            {
                "event_type": request.request_type.value,
                "source_task_id": request.source_task_id,
                "request_id": request.request_id,
                "summary": summary,
                "metadata": dict(request.metadata),
            },
        )
        ref = RuntimeEventRef(
            event_id=request.request_id,
            event_type=request.request_type.value,
            cursor=state.event_cursor + 1,
            summary=summary,
            metadata={"source_task_id": request.source_task_id, **dict(request.metadata)},
        )
        state.push_event(ref)
        return ref

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _list_or_set(value: Any) -> list[Any] | set[Any] | None:
        if value is None:
            return None
        if isinstance(value, (list, set)):
            return value
        return [value]

    @staticmethod
    def _int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _result_payload_ref(result: AgentTaskResult) -> str:
        raw_result_ref = result.outcome_payload.get("raw_result_ref")
        if raw_result_ref is not None:
            return str(raw_result_ref)
        if result.evidence:
            return str(result.evidence[0].payload_ref)
        return f"runtime://worker-results/{result.task_id}"


__all__ = ["PhaseTwoApplyResult", "PhaseTwoResultApplier"]
