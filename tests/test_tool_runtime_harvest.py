"""Regression tests for the deterministic tool-output harvest + CIDR zone resolution.

These cover the two generality gaps that previously stalled the full-chain lab at
recon (never reaching the restricted/internal zone, never resolving the internal DB):

1. The MCP lab tools already report ``register_pivot_route`` / ``open_session`` as
   ``parsed_output.runtime_hints`` on each tool call, but those facts only became
   real runtime Sessions / PivotRoutes if the LLM finish payload re-emitted them.
   ``PhaseTwoResultApplier._harvest_tool_runtime_facts`` now lifts them
   deterministically from the tool traces.

2. Services discovered in the restricted zone carry no explicit ``zone_ref``; the
   predicate engine now resolves them to a zone by CIDR containment against the
   profile's zone CIDRs (no hardcoded network names/IPs).
"""

from __future__ import annotations

from src.core.evaluation.models import OperationProfile, ZoneBinding
from src.core.evaluation.predicate_engine import PredicateContext, PredicateEngine
from src.core.models.runtime import OperationRuntime, PivotRouteStatus, RuntimeState
from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.ag import AttackGraph
from src.core.runtime.result_applier import PhaseTwoResultApplier
from src.core.stage.models import StageResult, ToolTrace


def _state() -> RuntimeState:
    return RuntimeState(operation_id="op-harvest", execution=OperationRuntime(operation_id="op-harvest"))


def _restricted_profile() -> OperationProfile:
    # Mirrors full_chain_lab zone_bindings; CIDRs come from the profile, not code.
    return OperationProfile(
        profile_id="test-restricted-profile",
        zone_bindings={
            "entry": ZoneBinding(name="dmz", cidrs=["10.20.0.0/24"], directly_reachable=True),
            "restricted": ZoneBinding(name="internal", cidrs=["10.30.0.0/24"], directly_reachable=False),
        }
    )


def test_harvest_lifts_pivot_route_and_session_from_tool_trace() -> None:
    """A stage whose LLM finish payload omitted sessions/pivot_routes still
    materializes real runtime objects from the deterministic tool hints."""

    stage = StageResult(
        operation_id="op-harvest",
        stage_task_id="stage-op-harvest-1-access_pivot_agent",
        capability="pivot",
        agent_name="access_pivot_agent",
        status="succeeded",
        summary="authorized pivot established",
        # NB: pivot_routes / sessions intentionally left empty here.
        tool_trace=[
            ToolTrace(
                tool_name="session_open_lab",
                success=True,
                parsed_output={
                    "runtime_hints": {
                        "open_session": True,
                        "session_id": "sess-pivot-1",
                        "bound_target": "10.20.0.50",
                        "lease_seconds": 300,
                        "reuse_policy": "exclusive",
                    }
                },
            ),
            ToolTrace(
                tool_name="pivot_route_register",
                success=True,
                parsed_output={
                    "runtime_hints": {
                        "register_pivot_route": True,
                        "route_id": "route::pivot::10.30.0.12:8080",
                        "destination_host": "10.30.0.12",
                        "via_host": "10.20.0.50",
                        "session_id": "sess-pivot-1",
                        "allowed_ports": [8080],
                        "protocol": "tcp",
                        "reachable": True,
                    }
                },
            ),
            ToolTrace(  # blocked traces must be ignored
                tool_name="pivot_route_probe",
                success=False,
                parsed_output={"runtime_hints": {"blocked_by": "pivot_unavailable"}},
            ),
        ],
    )

    state = _state()
    PhaseTwoResultApplier().apply_stage_result(stage, state, KnowledgeGraph(), AttackGraph())

    assert "sess-pivot-1" in state.sessions
    assert "route::pivot::10.30.0.12:8080" in state.pivot_routes
    route = state.pivot_routes["route::pivot::10.30.0.12:8080"]
    assert route.destination_host == "10.30.0.12"
    assert route.status == PivotRouteStatus.ACTIVE  # auto-activated because reachable


def test_harvest_is_idempotent_with_llm_supplied_routes() -> None:
    """If the LLM already supplied the route, the harvest does not duplicate it."""

    stage = StageResult(
        operation_id="op-harvest",
        stage_task_id="stage-op-harvest-1-access_pivot_agent",
        capability="pivot",
        agent_name="access_pivot_agent",
        status="succeeded",
        summary="pivot",
        pivot_routes=[{"route_id": "route-x", "destination_host": "10.30.0.12", "active": True}],
        tool_trace=[
            ToolTrace(
                tool_name="pivot_route_register",
                success=True,
                parsed_output={
                    "runtime_hints": {
                        "register_pivot_route": True,
                        "route_id": "route-x",
                        "destination_host": "10.30.0.12",
                        "reachable": True,
                    }
                },
            )
        ],
    )

    state = _state()
    PhaseTwoResultApplier().apply_stage_result(stage, state, KnowledgeGraph(), AttackGraph())
    assert list(state.pivot_routes.keys()) == ["route-x"]


def test_tool_reported_vulnerability_candidate_becomes_typed_kg_node() -> None:
    """A tool-reported ``VulnerabilityCandidate`` entity is minted as a typed KG
    node (not flattened into a generic Finding), satisfying the strongly-typed
    ``vulnerability_candidate_recorded`` contract condition."""

    stage = StageResult(
        operation_id="op-harvest",
        stage_task_id="stage-op-harvest-1-vuln_analysis_agent",
        capability="analysis",
        agent_name="vuln_analysis_agent",
        status="succeeded",
        summary="vulnerability profile matched",
        tool_trace=[
            ToolTrace(
                tool_name="vuln_profile_match",
                success=True,
                parsed_output={
                    "entities": [
                        {
                            "type": "VulnerabilityCandidate",
                            "candidate_id": "vuln-candidate::cve-2017-5638::10.20.0.10",
                            "vulnerability_id": "cve-2017-5638",
                            "confidence": 0.75,
                        }
                    ]
                },
            )
        ],
    )

    state = _state()
    kg = KnowledgeGraph()
    PhaseTwoResultApplier().apply_stage_result(stage, state, kg, AttackGraph())

    types = {node.model_dump(mode="json").get("type") for node in kg.list_nodes()}
    assert "VulnerabilityCandidate" in types


def test_cidr_resolution_satisfies_restricted_service_via_route() -> None:
    """A Service at an internal IP (no zone_ref tag) plus an active pivot route to
    that IP satisfies ``service_discovered_via_route`` for the restricted zone via
    profile-CIDR resolution."""

    profile = _restricted_profile()
    kg_nodes = [
        # Discovered via the pivoted scan; carries NO explicit zone_ref.
        {"id": "service::10.30.0.12:8080/tcp", "type": "Service", "address": "10.30.0.12", "port": 8080},
    ]
    runtime_state = {
        "pivot_routes": {
            "route-1": {
                "route_id": "route-1",
                "destination_host": "10.30.0.12",
                "status": "active",
            }
        }
    }
    ctx = PredicateContext(profile=profile, kg_nodes=kg_nodes, runtime_state=runtime_state)
    result = PredicateEngine().evaluate(
        "service_discovered_via_route",
        {"zone_ref": "restricted", "route_type": "pivot_route"},
        ctx,
    )
    assert result.satisfied is True
    assert "service::10.30.0.12:8080/tcp" in result.matched_node_ids


def test_addressless_evidence_stays_permissive_for_zone_filter() -> None:
    """An address-less Evidence node (no zone_ref) must still satisfy a
    zone-filtered exists_node check — CIDR strictness must not regress
    free-text evidence (e.g. service_fingerprint_recorded)."""

    profile = _restricted_profile()
    kg_nodes = [{"id": "evidence-1", "type": "Evidence", "summary": "http fingerprint banner", "label": "fp"}]
    ctx = PredicateContext(profile=profile, kg_nodes=kg_nodes)
    result = PredicateEngine().evaluate(
        "exists_node",
        {"graph": "kg", "type": {"in": ["Evidence", "Observation", "Finding"]}, "filters": {"zone_ref": "entry"}},
        ctx,
    )
    assert result.satisfied is True


def test_cidr_resolution_excludes_service_outside_zone() -> None:
    """A DMZ-range service must NOT count as a restricted-zone service."""

    profile = _restricted_profile()
    kg_nodes = [{"id": "service::10.20.0.10:8080/tcp", "type": "Service", "address": "10.20.0.10", "port": 8080}]
    runtime_state = {
        "pivot_routes": {"route-1": {"route_id": "route-1", "destination_host": "10.30.0.12", "status": "active"}}
    }
    ctx = PredicateContext(profile=profile, kg_nodes=kg_nodes, runtime_state=runtime_state)
    result = PredicateEngine().evaluate(
        "service_discovered_via_route",
        {"zone_ref": "restricted", "route_type": "pivot_route"},
        ctx,
    )
    assert result.satisfied is False
