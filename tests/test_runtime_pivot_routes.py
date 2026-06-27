from __future__ import annotations

import pytest

from src.core.models.runtime import OperationRuntime, PivotRouteStatus, RuntimeState
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


def test_get_missing_route_raises() -> None:
    state = build_state()
    manager = RuntimePivotRouteManager()

    with pytest.raises(ValueError):
        manager.get_route(state, "missing")
