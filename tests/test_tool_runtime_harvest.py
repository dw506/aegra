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
from src.core.execution.models import ExecutionResult, ToolTrace


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

    stage = ExecutionResult(
        operation_id="op-harvest",
        execution_id="stage-op-harvest-1-access_pivot_agent",
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
    PhaseTwoResultApplier().apply_execution_result(stage, state, KnowledgeGraph(), AttackGraph())

    assert "sess-pivot-1" in state.sessions
    assert "route::pivot::10.30.0.12:8080" in state.pivot_routes
    route = state.pivot_routes["route::pivot::10.30.0.12:8080"]
    assert route.destination_host == "10.30.0.12"
    assert route.status == PivotRouteStatus.ACTIVE  # auto-activated because reachable


def test_harvest_dedupes_repeated_route_traces() -> None:
    """Two tool traces reporting the same route_id materialize a single route
    (the channel-② LLM self-report field is gone; harvest is the only source)."""

    def _register_trace() -> ToolTrace:
        return ToolTrace(
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

    stage = ExecutionResult(
        operation_id="op-harvest",
        execution_id="stage-op-harvest-1-access_pivot_agent",
        capability="pivot",
        agent_name="access_pivot_agent",
        status="succeeded",
        summary="pivot",
        tool_trace=[_register_trace(), _register_trace()],
    )

    state = _state()
    PhaseTwoResultApplier().apply_execution_result(stage, state, KnowledgeGraph(), AttackGraph())
    assert list(state.pivot_routes.keys()) == ["route-x"]


def test_harvest_lifts_credential_from_tool_trace() -> None:
    """A validated credential reported as parsed.runtime_hints (no channel-②
    ExecutionResult.credentials field anymore) registers a runtime credential."""

    stage = ExecutionResult(
        operation_id="op-harvest",
        execution_id="stage-op-harvest-1-recon_agent",
        capability="exploit",
        agent_name="recon_agent",
        status="succeeded",
        summary="credential validated",
        tool_trace=[
            ToolTrace(
                tool_name="credential_check",
                success=True,
                parsed_output={
                    "runtime_hints": {
                        "credential_id": "cred-1",
                        "credential_status": "valid",
                        "principal": "admin",
                        "bind_target": "http://10.20.0.10/",
                        "target_service_id": "http://10.20.0.10/",
                    }
                },
            )
        ],
    )

    state = _state()
    PhaseTwoResultApplier().apply_execution_result(stage, state, KnowledgeGraph(), AttackGraph())

    assert "cred-1" in state.credentials


def test_harvest_maps_zone_ref_hint_to_route_destination_zone() -> None:
    """A pivot_route_register hint tags the route with ``zone_ref`` (the tool's
    field name), but PivotRouteRuntime/the predicate only speak ``destination_zone``.
    The applier must fold ``zone_ref`` into ``destination_zone`` so a route whose
    bridge host sits in the entry CIDR (e.g. 10.20.0.50) still binds to the
    restricted zone — the exact case that stalled full-chain bookkeeping."""

    stage = ExecutionResult(
        operation_id="op-harvest",
        execution_id="stage-op-harvest-1-access_pivot_agent",
        capability="pivot",
        agent_name="access_pivot_agent",
        status="succeeded",
        summary="pivot route registered into restricted zone",
        tool_trace=[
            ToolTrace(
                tool_name="pivot_route_register",
                success=True,
                parsed_output={
                    "runtime_hints": {
                        "register_pivot_route": True,
                        "route_id": "route::10.20.0.11::10.20.0.50:22",
                        # Bridge host's entry-side address: NOT inside the restricted CIDR.
                        "destination_host": "10.20.0.50",
                        "zone_ref": "restricted",
                        "reachable": True,
                    }
                },
            )
        ],
    )

    state = _state()
    PhaseTwoResultApplier().apply_execution_result(stage, state, KnowledgeGraph(), AttackGraph())

    route = state.pivot_routes["route::10.20.0.11::10.20.0.50:22"]
    assert route.destination_zone == "restricted"


def test_tool_reported_vulnerability_candidate_becomes_typed_kg_node() -> None:
    """A tool-reported ``VulnerabilityCandidate`` entity is minted as a typed KG
    node (not flattened into a generic Finding), satisfying the strongly-typed
    ``vulnerability_candidate_recorded`` contract condition."""

    stage = ExecutionResult(
        operation_id="op-harvest",
        execution_id="stage-op-harvest-1-vuln_analysis_agent",
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
    PhaseTwoResultApplier().apply_execution_result(stage, state, kg, AttackGraph())

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
