#把 ExecutionAgent 的 ExecutionResult 写回到 Runtime、KG、AG 和审计日志中

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.attack_process import AttackProcessEdge, AttackProcessEdgeType, AttackProcessNodeType, AttackStepNode, stable_node_id
from src.core.models.runtime import OutcomeCacheEntry, ReplanRequest, RuntimeEventRef, RuntimeState, utc_now
from src.core.planning.models import PlannerOutcome
from src.core.runtime.credential_manager import RuntimeCredentialManager
from src.core.runtime.events import ReplanRequestedEvent, SessionOpenedEvent, event_to_ref
from src.core.runtime.observability import append_audit_log
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager
from src.core.runtime.session_manager import RuntimeSessionManager
from src.core.runtime.tool_trace_fact_extractor import ToolTraceFactExtractor
from src.core.execution.models import ExecutionResult


class PhaseTwoApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_event_refs: list[RuntimeEventRef] = Field(default_factory=list)
    kg_state_deltas: list[dict[str, Any]] = Field(default_factory=list)
    kg_apply_result: dict[str, Any] | None = None
    kg_write_diagnostics: dict[str, Any] = Field(default_factory=dict)
    ag_graph: dict[str, Any] | None = None
    logs: list[str] = Field(default_factory=list)


class PhaseTwoResultApplier:
    """The sole v3 writeback owner.  It accepts no worker compatibility protocol."""

    #初始化了四个管理器，管理会话，凭证，路由和事实
    def __init__(self) -> None:
        self._sessions = RuntimeSessionManager()
        self._credentials = RuntimeCredentialManager()
        self._routes = RuntimePivotRouteManager()
        self._facts = ToolTraceFactExtractor()

    #记录 Planner 决策
    def apply_planner_outcome(self, outcome: PlannerOutcome, state: RuntimeState, kg_store: KnowledgeGraph, attack_graph: AttackGraph) -> PhaseTwoApplyResult:
        #显式删除当前函数作用域里的 kg_store 这个局部变量引用
        del kg_store     

        #把上一轮 PlannerOutcome 写入 Runtime metadata，写一条 audit log
        state.execution.metadata["last_planner_outcome"] = outcome.model_dump(mode="json")
        append_audit_log(state, {"event_type": "planner_outcome_applied", "planner_outcome": outcome.model_dump(mode="json")})
        return PhaseTwoApplyResult(ag_graph=attack_graph.to_dict(), logs=[f"recorded planner outcome {outcome.action}"])

    def apply_execution_result(self, execution_result: ExecutionResult, state: RuntimeState, kg_store: KnowledgeGraph, attack_graph: AttackGraph) -> PhaseTwoApplyResult:
       
        if execution_result.operation_id != state.operation_id:
            raise ValueError("execution_result.operation_id must match RuntimeState.operation_id")
       
        harvested = self._harvest_runtime_facts(execution_result)
        result = PhaseTwoApplyResult()
        result.runtime_event_refs.extend(self._apply_runtime(execution_result, state, harvested))
        self._record_runtime_metadata(execution_result, state, harvested)
        deltas = self._fact_deltas(execution_result)
        result.kg_state_deltas = self._ordered(deltas)
        if deltas:
            result.kg_apply_result = kg_store.apply_patch_batch({"operation_id": state.operation_id, "base_kg_version": kg_store.version, "state_deltas": result.kg_state_deltas})
            result.kg_write_diagnostics = self._write_summary(len(deltas), result.kg_apply_result)
        else:
            result.kg_write_diagnostics = {"status": "no_deltas", "reason": "stage produced no writable facts", "delta_count": 0}
        self._record_attack_step(execution_result, attack_graph, self._kg_refs(result.kg_state_deltas))
        result.ag_graph = attack_graph.to_dict()
        self._audit(execution_result, state)
        return result

    def _apply_runtime(self, stage: ExecutionResult, state: RuntimeState, harvested: dict[str, list[dict[str, Any]]]) -> list[RuntimeEventRef]:
        refs: list[RuntimeEventRef] = []
        for item in harvested["sessions"]:
            if not isinstance(item, dict): continue
            session_id = self._string(item.get("session_id")) or f"session::{stage.execution_id}"
            session = self._sessions.open_session(state, session_id, self._string(item.get("bound_identity") or item.get("identity")), self._string(item.get("bound_target") or item.get("target_id") or item.get("host_id")), lease_seconds=int(item.get("lease_seconds") or 300), reusability=str(item.get("reuse_policy") or "shared"))
            self._sessions.bind_task_to_session(state, stage.execution_id, session_id)
            refs.append(self._push(state, SessionOpenedEvent(operation_id=state.operation_id, session_id=session.session_id, bound_identity=session.bound_identity, bound_target=session.bound_target, lease_expiry=session.lease_expiry, reusability=session.reusability)))
        for item in harvested["pivot_routes"]:
            if not isinstance(item, dict): continue
            destination = self._string(item.get("destination_host") or item.get("target_host"))
            if not destination: continue
            route = self._routes.register_candidate(state, self._string(item.get("route_id")) or f"route::{stage.execution_id}", destination, source_host=self._string(item.get("source_host")), via_host=self._string(item.get("via_host")), session_id=self._string(item.get("session_id")), protocol=self._string(item.get("protocol")), destination_zone=self._string(item.get("destination_zone") or item.get("zone_ref")), destination_cidr=self._string(item.get("destination_cidr")), allowed_ports=self._as_set(item.get("allowed_ports") or item.get("port")), protocols=self._as_set(item.get("protocols")), confidence=self._float(item.get("confidence")), metadata={"source_task_id": stage.execution_id})
            if item.get("active") or item.get("reachable"): self._routes.activate_route(state, route.route_id)
        if stage.status in {"blocked", "need_more_info", "needs_replan"}:
            request = ReplanRequest(request_id=f"replan::{stage.result_id}", reason=stage.replan_recommendation or stage.summary, task_ids=[stage.execution_id], scope="local")
            state.request_replan(request)
            refs.append(self._push(state, ReplanRequestedEvent(operation_id=state.operation_id, request_id=request.request_id, reason=request.reason, task_ids=request.task_ids, scope=request.scope)))
        return refs

    def _record_runtime_metadata(self, stage: ExecutionResult, state: RuntimeState, harvested: dict[str, list[dict[str, Any]]]) -> None:
        hints = stage.runtime_hints
        if "goal_satisfied" in hints:
            state.execution.metadata["goal_state"] = {"goal_satisfied": bool(hints["goal_satisfied"]), "goal_summary": hints.get("goal_summary", stage.summary), "goal_evidence_refs": list(hints.get("goal_evidence_refs") or []), "source_task_id": stage.execution_id}
        if stage.findings: state.execution.metadata.setdefault("findings", []).extend(dict(v) for v in stage.findings if isinstance(v, dict))
        for item in harvested["credentials"]:
            if not isinstance(item, dict): continue
            credential_id = self._string(item.get("credential_id") or item.get("id"))
            if not credential_id: continue
            if credential_id not in state.credentials:
                self._credentials.upsert_credential(state, credential_id, self._string(item.get("principal") or item.get("username")) or "unknown-principal", kind=str(item.get("kind") or "password"), secret_ref=self._string(item.get("secret_ref")), metadata={"source_task_id": stage.execution_id})
            status = self._string(item.get("status"))
            if status: self._credentials.record_validation(state, credential_id, status=status, target_id=self._string(item.get("target_id")), metadata={"source_task_id": stage.execution_id})
        state.record_outcome(OutcomeCacheEntry(outcome_id=stage.result_id, task_id=stage.execution_id, outcome_type=stage.capability, summary=stage.summary, payload_ref=f"runtime://execution-results/{stage.execution_id}", metadata={"status": stage.status, "agent": stage.agent_name}))

    def _harvest_runtime_facts(self, stage: ExecutionResult) -> dict[str, list[dict[str, Any]]]:
        """Derive runtime facts (sessions / pivot routes / credentials) SOLELY from
        successful tool traces — channel ①, the tool is the authority.

        The former channel-② self-report fields (ExecutionResult.sessions/
        pivot_routes/credentials) are gone; this single deterministic bridge is now
        the only path that materializes runtime objects. Tools emit the facts as
        ``parsed_output.runtime_hints`` (session_open/register_pivot_route/
        credential_id), so canned and real tools both feed it identically."""

        sessions: list[dict[str, Any]] = []
        routes: list[dict[str, Any]] = []
        credentials: list[dict[str, Any]] = []
        seen_sessions: set[str] = set()
        seen_routes: set[str] = set()
        seen_creds: set[str] = set()
        for trace in stage.tool_trace:
            hints = trace.parsed_output.get("runtime_hints") if trace.success else None
            if not isinstance(hints, dict) or hints.get("blocked_by"): continue
            session_id = self._string(hints.get("session_id"))
            if session_id and session_id not in seen_sessions:
                sessions.append({k: hints[k] for k in ("session_id", "bound_identity", "bound_target", "lease_seconds", "reuse_policy") if k in hints}); seen_sessions.add(session_id)
            route_id = self._string(hints.get("route_id"))
            if hints.get("register_pivot_route") and route_id and route_id not in seen_routes:
                routes.append(dict(hints)); seen_routes.add(route_id)
            credential_id = self._string(hints.get("credential_id"))
            if credential_id and credential_id not in seen_creds:
                credentials.append({
                    "credential_id": credential_id,
                    "principal": hints.get("principal"),
                    "kind": hints.get("kind") or "password",
                    "secret_ref": hints.get("secret_ref"),
                    "status": hints.get("credential_status") or hints.get("status"),
                    "target_id": hints.get("bind_target") or hints.get("target_service_id"),
                })
                seen_creds.add(credential_id)
        return {"sessions": sessions, "pivot_routes": routes, "credentials": credentials}

    def _fact_deltas(self, stage: ExecutionResult) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = [dict(x) for x in stage.discovered_entities if isinstance(x, dict)]
        # Single KG machine-fact source: the deterministic ToolTraceFactExtractor.
        # The tools' self-declared parsed.entities envelope is NOT re-ingested here
        # (it used inconsistent id fields — candidate_id/capability_id/session_id —
        # so typed nodes were silently dropped or duplicated against the extractor's
        # synthesized ids). Every tool that yields a contract-relevant node has a
        # dedicated _extract_* function; per-tool extractors own the id scheme.
        for trace_result in self._facts.extract_all(stage.tool_trace):
            for fact in trace_result.facts:
                records.append({"id": f"{fact.entity_type.lower()}::{self._hash({'label': fact.label, 'tool': fact.source_tool})}", "type": fact.entity_type, "label": fact.label, **fact.properties, "confidence": fact.confidence})
        for observation in stage.observations:
            if not isinstance(observation, dict): continue
            records.extend(self._structured_records(observation, stage))
        for item in stage.findings:
            if isinstance(item, dict): records.append({"id": item.get("finding_id") or item.get("id"), "type": "Finding", "label": item.get("summary") or item.get("title") or "finding", **item})
        for evidence_ref in stage.evidence_refs:
            records.append({"id": evidence_ref, "type": "Evidence", "label": evidence_ref, "payload_ref": evidence_ref})
        hints = stage.runtime_hints
        if hints.get("goal_satisfied") and hints.get("goal_id"):
            records.append({"id": f"goal-proof::{hints['goal_id']}", "type": "GoalProof", "label": f"goal proof: {hints['goal_id']}", "goal_id": hints["goal_id"], "evidence_refs": list(hints.get("goal_evidence_refs") or hints.get("evidence_refs") or []), "proof_token": hints.get("proof_token"), "redacted_summary": hints.get("goal_summary") or stage.summary})
        deltas = [self._entity_delta(record, stage) for record in records if self._string(record.get("id")) and self._string(record.get("type"))]
        for record in records:
            if str(record.get("type", "")).lower() != "service": continue
            host = self._string(record.get("address") or record.get("host"))
            if host:
                host_id, service_id = f"host::{host}", self._string(record.get("id"))
                if service_id:
                    deltas.append(self._relation_delta(host_id, service_id, {"type": "HOSTS", "id": f"hosts::{host_id}::{service_id}"}, stage))
        for relation in stage.discovered_relations:
            if isinstance(relation, dict):
                source, target = self._string(relation.get("source") or relation.get("from")), self._string(relation.get("target") or relation.get("to"))
                if source and target: deltas.append(self._relation_delta(source, target, relation, stage))
        return deltas

    def _structured_records(self, payload: dict[str, Any], stage: ExecutionResult) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for host in payload.get("hosts_up") or payload.get("hosts") or []:
            value = host.get("host") if isinstance(host, dict) else host
            if value: result.append({"id": f"host::{value}", "type": "Host", "label": str(value), "address": str(value), **(host if isinstance(host, dict) else {})})
        services = payload.get("service_discovery") or payload.get("services") or payload.get("analysis", {}).get("service_fingerprints", [])
        for service in services if isinstance(services, list) else []:
            if not isinstance(service, dict): continue
            host, port, protocol = self._string(service.get("host") or service.get("address")), self._int(service.get("port")), self._string(service.get("protocol")) or "tcp"
            if host and port:
                fp = service.get("improved_fingerprint") if isinstance(service.get("improved_fingerprint"), dict) else {}
                result.append({"id": f"service::{host}:{port}/{protocol}", "type": "Service", "label": f"{host}:{port}/{protocol}", "address": host, "port": port, "protocol": protocol, "service_name": service.get("service") or fp.get("application"), "version": service.get("version") or fp.get("application_version")})
        for index, item in enumerate(payload.get("negative_evidence") or []): result.append({"id": f"evidence::{stage.execution_id}::negative_evidence::{index}", "type": "Evidence", "label": str(item), "summary": str(item)})
        return result

    def _entity_delta(self, record: dict[str, Any], stage: ExecutionResult) -> dict[str, Any]:
        entity_id, entity_type = str(record["id"]), str(record["type"])
        attributes = {k: v for k, v in record.items() if k not in {"id", "type", "entity_type", "label", "summary", "confidence"} and v is not None}
        if stage.evidence_refs:
            attributes.setdefault("evidence_refs", list(stage.evidence_refs))
        return {"id": f"entity::{entity_id}", "payload": {"patch_kind": "entity"}, "patch": {"entity_kind": "node", "entity_id": entity_id, "entity_type": entity_type, "label": str(record.get("label") or record.get("summary") or entity_id), "attributes": attributes, "confidence": float(record.get("confidence") or stage.confidence)}}

    def _relation_delta(self, source: str, target: str, relation: dict[str, Any], stage: ExecutionResult) -> dict[str, Any]:
        relation_type = str(relation.get("relation_type") or relation.get("type") or "RELATED_TO")
        relation_id = str(relation.get("id") or f"{relation_type.lower()}::{source}::{target}")
        return {"id": f"relation::{relation_id}", "payload": {"patch_kind": "relation"}, "patch": {"entity_kind": "edge", "relation_id": relation_id, "relation_type": relation_type, "source": source, "target": target, "label": relation.get("label") or relation_type, "attributes": {k: v for k, v in relation.items() if k not in {"id", "source", "from", "target", "to", "type", "relation_type", "label"}}}}

    def _record_attack_step(self, stage: ExecutionResult, graph: AttackGraph, refs: list[str]) -> None:
        cycle = int(stage.runtime_hints.get("cycle_index") or 0)
        node = AttackStepNode(id=stable_node_id("attack-step", {"operation_id": stage.operation_id, "cycle_index": cycle, "stage": stage.execution_id}), node_type=AttackProcessNodeType.ATTACK_STEP, label=stage.summary, operation_id=stage.operation_id, cycle_index=cycle, agent_name=stage.agent_name, capability=stage.capability, status=stage.status, summary=stage.summary, evidence_refs=list(stage.evidence_refs), kg_node_refs=refs, properties={"execution_id": stage.execution_id})
        if all(existing.id != node.id for existing in graph.find_process_nodes()): graph.add_node(node)
        prior = [n for n in graph.find_process_nodes() if isinstance(n, AttackStepNode) and n.id != node.id]
        if prior:
            previous = max(prior, key=lambda n: n.cycle_index)
            edge = AttackProcessEdge(id=stable_node_id("attack-next", {"source": previous.id, "target": node.id}), source=previous.id, target=node.id, edge_type=AttackProcessEdgeType.NEXT, label="next")
            try: graph.add_edge(edge)
            except ValueError: pass

    def _audit(self, stage: ExecutionResult, state: RuntimeState) -> None:
        append_audit_log(state, {"event_type": "execution_result_applied", "source_task_id": stage.execution_id, "result_id": stage.result_id, "capability": stage.capability, "execution_status": stage.status, "summary": stage.summary})
        for trace in stage.tool_trace: append_audit_log(state, {"event_type": "execution_tool_trace", "source_task_id": stage.execution_id, "result_id": stage.result_id, "tool_name": trace.tool_name, "success": trace.success, "summary": trace.summary})

    @staticmethod
    def _ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]: return sorted(items, key=lambda x: 0 if x.get("payload", {}).get("patch_kind") == "entity" else 1)
    @staticmethod
    def _kg_refs(items: list[dict[str, Any]]) -> list[str]: return [str(x["patch"].get("entity_id")) for x in items if x.get("patch", {}).get("entity_id")]
    @staticmethod
    def _write_summary(count: int, value: dict[str, Any]) -> dict[str, Any]: return {"status": "partial_write" if value.get("failed_delta_ids") else "ok", "delta_count": count, "failed_delta_count": len(value.get("failed_delta_ids") or [])}
    @staticmethod
    def _push(state: RuntimeState, event: Any) -> RuntimeEventRef:
        ref = event_to_ref(event, cursor=state.event_cursor + 1); state.push_event(ref); return ref
    @staticmethod
    def _string(value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""; return text or None
    @staticmethod
    def _int(value: Any) -> int | None:
        try: return int(value)
        except (TypeError, ValueError): return None
    @staticmethod
    def _float(value: Any) -> float | None:
        try: return float(value)
        except (TypeError, ValueError): return None
    @staticmethod
    def _as_set(value: Any) -> set[Any] | None: return set(value) if isinstance(value, list) else ({value} if value is not None else None)
    @staticmethod
    def _hash(value: Any) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


__all__ = ["PhaseTwoApplyResult", "PhaseTwoResultApplier"]
