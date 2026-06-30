from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.models.attack_process import AttackProcessNodeType
from src.core.models.runtime import OperationRuntime, RuntimeState
from src.core.planning.models import PlannerOutcome
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.execution.models import RoundDirective, ExecutionResult, ToolTrace


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
    applier.apply_planner_outcome(outcome, state, kg, ag)

    execution_result = ExecutionResult(
        operation_id="op-apply",
        execution_id="stage-op-apply-1-recon_agent",
        capability="recon",
        agent_name="recon_agent",
        status="succeeded",
        summary="host and service observed",
        tool_trace=[ToolTrace(tool_name="safe_probe", success=True, summary="probe ok", raw_output_ref="runtime://tool-output/probe-1")],
    )
    applier.apply_execution_result(execution_result, state, kg, ag)

    # v3 result-tier AG: one ATTACK_STEP per round; legacy process nodes are gone.
    process_types = {node.node_type for node in ag.find_process_nodes()}
    assert process_types == {AttackProcessNodeType.ATTACK_STEP}
    # KG facts derive SOLELY from tool_trace now (channel ①): the successful probe
    # mints a tool-evidence node. The former discovered_entities self-report is gone.
    evidence_nodes = [node for node in kg.list_nodes() if node.type.value == "Evidence"]
    assert any(node.properties.get("tool_name") == "safe_probe" for node in evidence_nodes)
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
    bare = ExecutionResult(
        operation_id="op-proof",
        execution_id="stage-op-proof-1-goal_agent",
        capability="goal",
        agent_name="goal_agent",
        status="succeeded",
        summary="reachability goal check passed",
        runtime_hints={"goal_satisfied": True, "goal_evidence_refs": ["evidence::reach"]},
    )
    applier.apply_execution_result(bare, state, kg, ag)
    assert "goal-proof::final_internal_goal" not in {node.id for node in kg.list_nodes()}

    # goal_satisfied + explicit goal_id -> typed GoalProof node carrying goal_id.
    proof = ExecutionResult(
        operation_id="op-proof",
        execution_id="stage-op-proof-2-access_pivot_agent",
        capability="pivot",
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
    applier.apply_execution_result(proof, state, kg, ag)
    node = kg.get_node("goal-proof::final_internal_goal")
    assert node is not None
    payload = node.model_dump(mode="json")
    assert payload["type"] == "GoalProof"
    assert payload["goal_id"] == "final_internal_goal"
    assert "evidence::db-proof" in payload["evidence_refs"]


def test_result_applier_writes_tool_result_evidence_when_no_structured_shape() -> None:
    """B4: a successful tool call with no structured parsed_output still lands in KG."""

    state = RuntimeState(operation_id="op-toolonly", execution=OperationRuntime(operation_id="op-toolonly"))
    kg = KnowledgeGraph()
    ag = AttackGraph()
    applier = PhaseTwoResultApplier()

    execution_result = ExecutionResult(
        operation_id="op-toolonly",
        execution_id="stage-op-toolonly-1-recon_agent",
        capability="recon",
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

    apply_result = applier.apply_execution_result(execution_result, state, kg, ag)

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

    execution_result = ExecutionResult(
        operation_id="op-empty",
        execution_id="stage-op-empty-1-recon_agent",
        capability="recon",
        agent_name="recon_agent",
        status="needs_replan",
        summary="nothing produced",
    )

    apply_result = applier.apply_execution_result(execution_result, state, kg, ag)

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


