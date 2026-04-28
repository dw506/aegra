from __future__ import annotations

import pytest

from src.core.models.runtime import OperationRuntime, PivotRouteStatus, RuntimeState, SessionRuntime, SessionStatus
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager


def build_state() -> RuntimeState:
    return RuntimeState(operation_id="op-1", execution=OperationRuntime(operation_id="op-1"))


def test_register_candidate_and_activate_route() -> None:
    state = build_state()
    manager = RuntimePivotRouteManager()
    route = manager.register_candidate(
        state,
        "route-1",
        "host-2",
        source_host="host-0",
        via_host="host-1",
        session_id="sess-1",
        protocol="ssh",
    )

    activated = manager.activate_route(state, "route-1")

    assert route.status == PivotRouteStatus.ACTIVE
    assert activated.last_verified_at is not None


def test_refresh_from_reachability_marks_route_failed_when_unreachable() -> None:
    state = build_state()
    manager = RuntimePivotRouteManager()

    route = manager.refresh_from_reachability(
        state,
        destination_host="host-2",
        reachable=False,
        source_host="host-0",
        via_host="host-1",
        session_id="sess-1",
        protocol="ssh",
    )

    assert route.status == PivotRouteStatus.FAILED
    assert route.metadata["failure_reason"] == "reachability_lost"


def test_select_best_route_prefers_active_route_with_session() -> None:
    state = build_state()
    manager = RuntimePivotRouteManager()
    state.sessions["sess-1"] = SessionRuntime(session_id="sess-1", status=SessionStatus.ACTIVE)
    manager.refresh_from_reachability(
        state,
        route_id="route-1",
        destination_host="host-2",
        reachable=True,
        source_host="host-0",
        protocol="tcp",
    )
    manager.refresh_from_reachability(
        state,
        route_id="route-2",
        destination_host="host-2",
        reachable=True,
        source_host="host-0",
        via_host="host-1",
        session_id="sess-1",
        protocol="ssh",
    )

    best = manager.select_best_route(state, "host-2", source_host="host-0")

    assert best is not None
    assert best.route_id == "route-2"


def test_select_best_route_ignores_route_with_missing_session() -> None:
    state = build_state()
    manager = RuntimePivotRouteManager()
    manager.refresh_from_reachability(
        state,
        route_id="route-1",
        destination_host="host-2",
        reachable=True,
        source_host="host-0",
        via_host="host-1",
        session_id="missing-session",
        protocol="ssh",
    )

    best = manager.select_best_route(state, "host-2", source_host="host-0")

    assert best is None


def test_get_missing_route_raises() -> None:
    state = build_state()
    manager = RuntimePivotRouteManager()

    with pytest.raises(ValueError):
        manager.get_route(state, "missing")
