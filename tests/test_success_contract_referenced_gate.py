"""Referenced-contract gate: the orchestrator producer must honour a profile's
``success_contract_ref`` by running the authoritative SuccessConditionTracker
against the live KG/AG/Runtime — not just inline ``success_conditions``.

These tests drive the *real* full_chain_lab contract (zones, predicates and the
goal oracle all come from the YAML files, nothing is hardcoded in product code)
and prove that ``eligible_for_stop`` only flips true once the whole chain —
including a restricted-zone service reached via an active pivot route and a
database proof — is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.models.runtime import (
    OperationRuntime,
    PivotRouteRuntime,
    PivotRouteStatus,
    RuntimeState,
    SessionRuntime,
    SessionStatus,
)

PROFILE_PATH = Path("lab/environments/full_chain_lab/profile.yml")


class _GraphStub:
    """Minimal stand-in exposing the ``to_dict`` surface the producer uses."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload


def _full_chain_kg_nodes() -> list[dict[str, Any]]:
    return [
        {"id": "goal-1", "type": "Goal", "category": "imported_target"},
        {"id": "host-1", "type": "Host", "zone_ref": "entry"},
        {"id": "svc-1", "type": "Service", "zone_ref": "entry"},
        {"id": "ev-1", "type": "Evidence", "zone_ref": "entry"},
        # Stage 2 (cg.md G.6): chain steps that collapse into Evidence are
        # discriminated by the ToolFact `kind`, so each condition reads a distinct
        # fact. Canned producers stamp the same kind the real tools do.
        {"id": "vc-1", "type": "Evidence", "kind": "vuln_candidate"},
        {"id": "ea-1", "type": "Evidence", "kind": "exploit_attempt"},
        {"id": "cap-1", "type": "Session"},
        {"id": "pao-1", "type": "PostAccessObservation", "kind": "post_access"},
        {"id": "cred-1", "type": "Credential", "kind": "credential"},
        {"id": "isvc-1", "type": "Service", "zone_ref": "restricted"},
        {
            "id": "db-proof-1",
            "type": "ControlledDataReadProof",
            "service": "postgres",
            "zone_ref": "restricted",
            "proof_sha256": "redacted-proof-hash",
        },
    ]


def _state_with_route_and_session() -> RuntimeState:
    state = RuntimeState(operation_id="op-ref", execution=OperationRuntime(operation_id="op-ref"))
    state.pivot_routes["route-1"] = PivotRouteRuntime(
        route_id="route-1",
        destination_host="internal-db",
        destination_zone="internal",
        destination_cidr="10.30.0.0/24",
        status=PivotRouteStatus.ACTIVE,
    )
    state.sessions["sess-1"] = SessionRuntime(session_id="sess-1", status=SessionStatus.ACTIVE)
    return state


def _orch() -> AppOrchestrator:
    settings = AppSettings(
        runtime_store_backend="memory",
        lab_profile_path=PROFILE_PATH,
    )
    return AppOrchestrator(settings=settings)


def test_referenced_contract_gate_opens_only_when_full_chain_present() -> None:
    if not PROFILE_PATH.exists():
        import pytest

        pytest.skip("full_chain_lab profile not present")

    orch = _orch()
    state = _state_with_route_and_session()
    kg = _GraphStub({"nodes": _full_chain_kg_nodes(), "edges": []})
    ag = _GraphStub({"nodes": [], "edges": []})

    progress = orch._update_success_condition_progress(state=state, kg=kg, ag=ag)

    assert progress["contract_id"] == "full_chain_pentest_contract_v1"
    assert progress["missing"] == []
    assert progress["all_required_satisfied"] is True
    assert progress["eligible_for_stop"] is True
    assert progress["achieved_level"] == "full"
    assert progress["target_level"] == "full"
    assert progress["level_results"]["full"]["satisfied"] is True
    assert progress["recommended_planner_action"] == "stop_success"
    # Backward-compatible conditions map is populated for downstream consumers.
    assert progress["conditions"]["database_proof_recorded"]["satisfied"] is True


def test_referenced_contract_gate_stays_closed_without_database_proof() -> None:
    if not PROFILE_PATH.exists():
        import pytest

        pytest.skip("full_chain_lab profile not present")

    orch = _orch()
    state = _state_with_route_and_session()
    nodes = [n for n in _full_chain_kg_nodes() if n["type"] != "ControlledDataReadProof"]
    kg = _GraphStub({"nodes": nodes, "edges": []})
    ag = _GraphStub({"nodes": [], "edges": []})

    progress = orch._update_success_condition_progress(state=state, kg=kg, ag=ag)

    assert progress["eligible_for_stop"] is False
    assert progress["achieved_level"] == "standard"
    assert progress["level_results"]["standard"]["satisfied"] is True
    assert progress["level_results"]["full"]["missing"] == ["database_proof_recorded"]
    assert "database_proof_recorded" in progress["missing"]
    assert progress["recommended_planner_action"] == "continue"


def test_kind_discriminator_blocks_false_condition_satisfaction() -> None:
    """Stage 2 (cg.md G.6): a plain fingerprint Evidence must not leak into the
    exploit/credential chain steps now that they are discriminated by ToolFact kind."""
    if not PROFILE_PATH.exists():
        import pytest

        pytest.skip("full_chain_lab profile not present")

    orch = _orch()
    state = _state_with_route_and_session()
    # Recon-only graph: an entry fingerprint Evidence exists, but no kind-tagged
    # vuln_candidate / exploit_attempt / post_access / credential facts.
    nodes = [
        {"id": "goal-1", "type": "Goal", "category": "imported_target"},
        {"id": "host-1", "type": "Host", "zone_ref": "entry"},
        {"id": "svc-1", "type": "Service", "zone_ref": "entry"},
        {"id": "ev-1", "type": "Evidence", "zone_ref": "entry", "kind": "web_fingerprint"},
    ]
    kg = _GraphStub({"nodes": nodes, "edges": []})
    ag = _GraphStub({"nodes": [], "edges": []})

    progress = orch._update_success_condition_progress(state=state, kg=kg, ag=ag)
    missing = set(progress["missing"])

    # The fingerprint Evidence satisfies the kind-agnostic fingerprint step ...
    assert "service_fingerprint_recorded" not in missing
    # ... but must NOT falsely satisfy the kind-discriminated exploit/credential steps.
    assert "vulnerability_candidate_recorded" in missing
    assert "exploit_attempt_recorded" in missing
    assert "post_access_observation_recorded" in missing
    assert "credential_or_pivot_hint_discovered" in missing


def test_referenced_contract_gate_stays_closed_without_pivot_route() -> None:
    """Restricted-zone discovery must go through an active pivot route."""
    if not PROFILE_PATH.exists():
        import pytest

        pytest.skip("full_chain_lab profile not present")

    orch = _orch()
    # No pivot route / session at all.
    state = RuntimeState(operation_id="op-ref", execution=OperationRuntime(operation_id="op-ref"))
    kg = _GraphStub({"nodes": _full_chain_kg_nodes(), "edges": []})
    ag = _GraphStub({"nodes": [], "edges": []})

    progress = orch._update_success_condition_progress(state=state, kg=kg, ag=ag)

    assert progress["eligible_for_stop"] is False
    assert progress["achieved_level"] == "minimal"
    assert progress["level_results"]["standard"]["satisfied"] is False
    assert "restricted_zone_service_discovered" in progress["missing"]
    assert "pivot_route_recorded" in progress["missing"]
