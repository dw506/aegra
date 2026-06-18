from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.agents.agent_protocol import GraphScope
from src.core.models.ag import AttackGraph, GraphRef
from src.core.models.attack_process import AttackProcessNodeType
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.planning.models import PlannerOutcome
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.models import RoundDirective, StageResult, StageType, ToolTrace


def test_result_applier_writes_planner_stage_tool_and_kg_facts_without_tg() -> None:
    state = RuntimeState(operation_id="op-apply", execution=OperationRuntime(operation_id="op-apply"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    outcome = PlannerOutcome(
        operation_id="op-apply",
        cycle_index=1,
        action="execute",
        directive=RoundDirective(
            operation_id="op-apply",
            cycle_index=1,
            capability="recon",
            objective="map exposed service",
            max_tools=2,
            risk_level="low",
        ),
        confidence=0.9,
    )
    planner_apply = applier.apply_planner_outcome(outcome, state, kg, ag)

    stage_result = StageResult(
        operation_id="op-apply",
        stage_task_id="stage-op-apply-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="host and service observed",
        discovered_entities=[
            {"id": "host-1", "type": "Host", "summary": "10.0.0.5", "address": "10.0.0.5", "confidence": 0.9},
            {"id": "svc-1", "type": "Service", "summary": "http service", "port": 80, "protocol": "http"},
            {"id": "evidence-1", "type": "Evidence", "summary": "probe output"},
        ],
        tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok")],
    )
    stage_apply = applier.apply_stage_result(stage_result, state, kg, ag)

    assert {delta.graph for delta in planner_apply.visual_graph_deltas + stage_apply.visual_graph_deltas} <= {
        "kg",
        "ag",
        "runtime",
    }
    # v3 result-tier AG: one ATTACK_STEP per round; legacy process nodes are gone.
    process_types = {node.node_type for node in ag.find_process_nodes()}
    assert process_types == {AttackProcessNodeType.ATTACK_STEP}
    assert kg.get_node("host-1") is not None
    assert kg.get_node("svc-1") is not None
    assert kg.get_node("evidence-1") is not None
    assert "task_graph" not in state.execution.metadata


def test_result_applier_mints_goal_proof_node_from_goal_satisfied_hint() -> None:
    """A stage asserting goal_satisfied with an explicit goal_id mints a typed
    GoalProof KG node (so success-contract oracle predicates can resolve it).
    A bare goal_satisfied without goal_id must NOT mint a proof node."""

    state = RuntimeState(operation_id="op-proof", execution=OperationRuntime(operation_id="op-proof"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    # No goal_id -> no GoalProof node.
    bare = StageResult(
        operation_id="op-proof",
        stage_task_id="stage-op-proof-1-goal_agent",
        stage_type=StageType.GOAL_STAGE,
        agent_name="goal_agent",
        status="succeeded",
        summary="reachability goal check passed",
        runtime_hints={"goal_satisfied": True, "goal_evidence_refs": ["evidence::reach"]},
    )
    applier.apply_stage_result(bare, state, kg, ag)
    assert "goal-proof::final_internal_goal" not in {node.id for node in kg.list_nodes()}

    # goal_satisfied + explicit goal_id -> typed GoalProof node carrying goal_id.
    proof = StageResult(
        operation_id="op-proof",
        stage_task_id="stage-op-proof-2-access_pivot_agent",
        stage_type=StageType.ACCESS_PIVOT_STAGE,
        agent_name="access_pivot_agent",
        status="succeeded",
        summary="controlled internal DB read proof recorded",
        runtime_hints={
            "goal_satisfied": True,
            "goal_id": "final_internal_goal",
            "goal_evidence_refs": ["evidence::db-proof"],
            "proof_token": "proof:final_internal_goal:abcdef0123456789",
        },
    )
    applier.apply_stage_result(proof, state, kg, ag)
    node = kg.get_node("goal-proof::final_internal_goal")
    assert node is not None
    payload = node.model_dump(mode="json")
    assert payload["type"] == "GoalProof"
    assert payload["goal_id"] == "final_internal_goal"
    assert "evidence::db-proof" in payload["evidence_refs"]


def test_result_applier_extracts_generic_stage_recon_shapes_to_kg() -> None:
    state = RuntimeState(operation_id="op-structured", execution=OperationRuntime(operation_id="op-structured"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    stage_result = StageResult(
        operation_id="op-structured",
        stage_task_id="stage-op-structured-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="structured recon output",
        observations=[
            {
                "category": "stage_structured_output",
                "summary": "structured recon output",
                "hosts_up": ["198.51.100.10"],
                "service_discovery": [
                    {"host": "198.51.100.10", "port": 8080, "protocol": "tcp", "service": "http"}
                ],
                "negative_evidence": ["no UDP services were confirmed"],
            }
        ],
        findings=[{"finding_id": "finding::service-discovery", "summary": "service discovery completed"}],
        evidence_refs=["runtime://tool-output/nmap"],
    )

    apply_result = applier.apply_stage_result(stage_result, state, kg, ag)

    assert apply_result.kg_apply_result is not None
    assert kg.get_node("host::198.51.100.10").address == "198.51.100.10"
    service = kg.get_node("service::198.51.100.10:8080/tcp")
    assert service.port == 8080
    assert service.protocol == "tcp"
    assert kg.get_edge("hosts::host::198.51.100.10::service::198.51.100.10:8080/tcp") is not None
    assert kg.get_node("finding::service-discovery") is not None
    assert kg.get_node("evidence::stage-op-structured-1-recon_agent::negative_evidence::0") is not None


def test_result_applier_extracts_service_fingerprints_to_kg() -> None:
    state = RuntimeState(operation_id="op-fingerprint", execution=OperationRuntime(operation_id="op-fingerprint"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    stage_result = StageResult(
        operation_id="op-fingerprint",
        stage_task_id="stage-op-fingerprint-2-vuln_analysis_agent",
        stage_type=StageType.VULN_ANALYSIS_STAGE,
        agent_name="vuln_analysis_agent",
        status="succeeded",
        summary="fingerprint analysis",
        observations=[
            {
                "category": "stage_structured_output",
                "summary": "fingerprint analysis",
                "analysis": {
                    "service_fingerprints": [
                        {
                            "host": "198.51.100.30",
                            "port": 8443,
                            "protocol": "https",
                            "improved_fingerprint": {
                                "application": "Example Console",
                                "application_version": "2.0",
                            },
                        }
                    ]
                },
            }
        ],
    )

    apply_result = applier.apply_stage_result(stage_result, state, kg, ag)

    assert apply_result.kg_apply_result is not None
    service = kg.get_node("service::198.51.100.30:8443/https")
    assert service.service_name == "Example Console"
    assert service.properties["version"] == "2.0"


def test_result_applier_writes_tool_result_evidence_when_no_structured_shape() -> None:
    """B4: a successful tool call with no structured parsed_output still lands in KG."""

    state = RuntimeState(operation_id="op-toolonly", execution=OperationRuntime(operation_id="op-toolonly"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    stage_result = StageResult(
        operation_id="op-toolonly",
        stage_task_id="stage-op-toolonly-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="unstructured tool output",
        tool_trace=[
            ToolTrace(
                tool_name="curl_probe",
                success=True,
                summary="fetched banner",
                raw_output_ref="runtime://tool-output/curl-1",
            )
        ],
    )

    apply_result = applier.apply_stage_result(stage_result, state, kg, ag)

    assert apply_result.kg_apply_result is not None
    assert apply_result.kg_write_diagnostics.get("status") in {"ok", "partial_write"}
    evidence_nodes = [node for node in kg.list_nodes() if node.type.value == "Evidence"]
    assert any(node.properties.get("tool_name") == "curl_probe" for node in evidence_nodes)


def test_result_applier_reports_diagnostics_when_no_deltas() -> None:
    """B5: an empty stage records a reason instead of writing silently."""

    state = RuntimeState(operation_id="op-empty", execution=OperationRuntime(operation_id="op-empty"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    stage_result = StageResult(
        operation_id="op-empty",
        stage_task_id="stage-op-empty-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="needs_replan",
        summary="nothing produced",
    )

    apply_result = applier.apply_stage_result(stage_result, state, kg, ag)

    assert apply_result.kg_write_diagnostics.get("status") == "no_deltas"
    assert apply_result.kg_write_diagnostics.get("reason")


def test_apply_patch_batch_tolerates_one_bad_delta() -> None:
    """A1: a single invalid delta is skipped, good deltas still apply."""

    kg = KnowledgeGraph()
    request = {
        "operation_id": "op-batch",
        "kg_ref": {"graph": "kg", "ref_id": "kg-root", "ref_type": "graph"},
        "base_kg_version": kg.version,
        "state_deltas": [
            {
                "id": "good",
                "payload": {"patch_kind": "entity"},
                "patch": {
                    "entity_kind": "node",
                    "entity_id": "host::10.0.0.9",
                    "entity_type": "Host",
                    "label": "10.0.0.9",
                    "attributes": {"address": "10.0.0.9"},
                },
            },
            {
                "id": "bad-edge",
                "payload": {"patch_kind": "relation"},
                "patch": {
                    "entity_kind": "edge",
                    "relation_id": "hosts::host::10.0.0.9::service::missing",
                    "relation_type": "HOSTS",
                    "source": "host::10.0.0.9",
                    "target": "service::does-not-exist",
                    "label": "hosts",
                    "attributes": {},
                },
            },
        ],
    }

    result = kg.apply_patch_batch(request)

    assert kg.get_node("host::10.0.0.9") is not None
    assert "bad-edge" in result["failed_delta_ids"]
    assert result["errors"]


def test_result_applier_preserves_rich_host_when_service_shares_host() -> None:
    """C6: the minimal service-loop host skeleton must not clobber rich host attributes."""

    state = RuntimeState(operation_id="op-merge", execution=OperationRuntime(operation_id="op-merge"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    stage_result = StageResult(
        operation_id="op-merge",
        stage_task_id="stage-op-merge-1-recon_agent",
        stage_type=StageType.RECON_STAGE,
        agent_name="recon_agent",
        status="succeeded",
        summary="host with hostname plus a service",
        observations=[
            {
                "category": "stage_structured_output",
                "summary": "recon",
                "hosts": [{"host": "203.0.113.7", "hostname": "web.example.test"}],
                "services": [{"host": "203.0.113.7", "port": 443, "protocol": "tcp", "service": "https"}],
            }
        ],
    )

    applier.apply_stage_result(stage_result, state, kg, ag)

    host = kg.get_node("host::203.0.113.7")
    # hostname 是 Host 的模型字段，富属性必须在 service 处理后仍被保留（C6）。
    assert host.hostname == "web.example.test"
    assert kg.get_node("service::203.0.113.7:443/tcp") is not None


def test_result_applier_maps_query_refs_to_ag_protocol_refs() -> None:
    ref = PhaseTwoResultApplier._to_protocol_ref(
        GraphRef(graph="query", ref_id="expected-output::svc", ref_type="ExpectedEvidence")
    )

    assert ref.graph == GraphScope.AG
    assert ref.ref_id == "expected-output::svc"
    assert ref.metadata["original_graph"] == "query"
