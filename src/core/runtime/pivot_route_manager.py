"""Runtime pivot route manager.

This module manages transient route candidates and active pivot paths used to
reach downstream hosts during one operation.
"""

from __future__ import annotations

from src.core.models.runtime import PivotRouteRuntime, PivotRouteStatus, RuntimeState, SessionStatus, utc_now


class RuntimePivotRouteManager:
    """Manage candidate, active and failed runtime pivot routes."""

    def register_candidate(
        self,
        state: RuntimeState,
        route_id: str,
        destination_host: str,
        *,
        source_host: str | None = None,
        via_host: str | None = None,
        session_id: str | None = None,
        protocol: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PivotRouteRuntime:
        """Create or refresh one candidate route."""

        now = utc_now()
        route = state.pivot_routes.get(route_id)
        if route is None:
            route = PivotRouteRuntime(
                route_id=route_id,
                destination_host=destination_host,
                source_host=source_host,
                via_host=via_host,
                session_id=session_id,
                status=PivotRouteStatus.CANDIDATE,
                protocol=protocol,
                metadata=dict(metadata or {}),
            )
            state.pivot_routes[route_id] = route
        else:
            route.destination_host = destination_host
            route.source_host = source_host
            route.via_host = via_host
            route.session_id = session_id
            route.protocol = protocol
            route.status = PivotRouteStatus.CANDIDATE
            if metadata:
                route.metadata.update(dict(metadata))
        state.last_updated = now
        return route

    def activate_route(self, state: RuntimeState, route_id: str) -> PivotRouteRuntime:
        """Mark one route as active and refresh its verification timestamp."""

        route = self.get_route(state, route_id)
        now = utc_now()
        route.status = PivotRouteStatus.ACTIVE
        route.last_verified_at = now
        state.last_updated = now
        return route

    def fail_route(
        self,
        state: RuntimeState,
        route_id: str,
        *,
        reason: str | None = None,
    ) -> PivotRouteRuntime:
        """Mark one route as failed and record the optional reason."""

        route = self.get_route(state, route_id)
        now = utc_now()
        route.status = PivotRouteStatus.FAILED
        route.last_verified_at = now
        if reason is not None:
            route.metadata["failure_reason"] = reason
        state.last_updated = now
        return route

    def close_route(
        self,
        state: RuntimeState,
        route_id: str,
        *,
        reason: str | None = None,
    ) -> PivotRouteRuntime:
        """Close one route and record the optional reason."""

        route = self.get_route(state, route_id)
        now = utc_now()
        route.status = PivotRouteStatus.CLOSED
        route.last_verified_at = now
        if reason is not None:
            route.metadata["close_reason"] = reason
        state.last_updated = now
        return route

    def refresh_from_reachability(
        self,
        state: RuntimeState,
        *,
        destination_host: str,
        reachable: bool,
        source_host: str | None = None,
        via_host: str | None = None,
        session_id: str | None = None,
        protocol: str | None = None,
        route_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PivotRouteRuntime:
        """Refresh one route using the latest reachability signal."""

        resolved_route_id = route_id or self._derive_route_id(
            destination_host=destination_host,
            source_host=source_host,
            via_host=via_host,
            session_id=session_id,
            protocol=protocol,
        )
        route = self.register_candidate(
            state,
            resolved_route_id,
            destination_host,
            source_host=source_host,
            via_host=via_host,
            session_id=session_id,
            protocol=protocol,
            metadata=metadata,
        )
        if reachable:
            return self.activate_route(state, route.route_id)
        return self.fail_route(state, route.route_id, reason="reachability_lost")

    def select_best_route(
        self,
        state: RuntimeState,
        destination_host: str,
        *,
        source_host: str | None = None,
    ) -> PivotRouteRuntime | None:
        """Return the best currently usable route for one destination host."""

        candidates = [
            route
            for route in state.pivot_routes.values()
            if route.destination_host == destination_host
            and route.status == PivotRouteStatus.ACTIVE
            and self._route_session_usable(state, route)
            and (source_host is None or route.source_host == source_host)
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (
                0 if item.session_id is not None else 1,
                0 if item.via_host is not None else 1,
                item.route_id,
            ),
        )[0]

    def get_route(self, state: RuntimeState, route_id: str) -> PivotRouteRuntime:
        """Return one tracked pivot route."""

        try:
            return state.pivot_routes[route_id]
        except KeyError as exc:
            raise ValueError(f"pivot route '{route_id}' does not exist") from exc

    def list_routes_for_session(self, state: RuntimeState, session_id: str) -> list[PivotRouteRuntime]:
        """Return tracked pivot routes attached to the given session."""

        return sorted(
            [route for route in state.pivot_routes.values() if route.session_id == session_id],
            key=lambda item: item.route_id,
        )

    def fail_routes_for_session(
        self,
        state: RuntimeState,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> int:
        """Mark routes attached to the given session as failed."""

        failed = 0
        for route in self.list_routes_for_session(state, session_id):
            self.fail_route(state, route.route_id, reason=reason or "source_session_failed")
            failed += 1
        return failed

    def close_routes_for_session(
        self,
        state: RuntimeState,
        session_id: str,
        *,
        reason: str | None = None,
    ) -> int:
        """Close routes attached to the given session."""

        closed = 0
        for route in self.list_routes_for_session(state, session_id):
            self.close_route(state, route.route_id, reason=reason or "source_session_closed")
            closed += 1
        return closed

    @staticmethod
    def _route_session_usable(state: RuntimeState, route: PivotRouteRuntime) -> bool:
        if route.session_id is None:
            return True
        session = state.sessions.get(route.session_id)
        if session is None:
            return False
        return session.status == SessionStatus.ACTIVE and session.is_session_usable()

    @staticmethod
    def _derive_route_id(
        *,
        destination_host: str,
        source_host: str | None,
        via_host: str | None,
        session_id: str | None,
        protocol: str | None,
    ) -> str:
        source_part = source_host or "unknown-source"
        via_part = via_host or "direct"
        session_part = session_id or "no-session"
        protocol_part = protocol or "unknown-protocol"
        return f"route::{source_part}::{via_part}::{destination_host}::{session_part}::{protocol_part}"


__all__ = ["RuntimePivotRouteManager"]
