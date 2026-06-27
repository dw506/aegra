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
        destination_zone: str | None = None,
        destination_cidr: str | None = None,
        allowed_ports: set[int] | list[int] | None = None,
        protocols: set[str] | list[str] | None = None,
        hop_count: int | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PivotRouteRuntime:
        """Create or refresh one candidate route."""

        now = utc_now()
        normalized_ports = self._normalize_ports(allowed_ports)
        normalized_protocols = self._normalize_protocols(protocols)
        if protocol is not None:
            normalized_protocols.add(protocol)
        route = state.pivot_routes.get(route_id)
        if route is None:
            route = PivotRouteRuntime(
                route_id=route_id,
                destination_host=destination_host,
                destination_zone=destination_zone,
                destination_cidr=destination_cidr,
                source_host=source_host,
                via_host=via_host,
                session_id=session_id,
                status=PivotRouteStatus.CANDIDATE,
                protocol=protocol,
                allowed_ports=normalized_ports,
                protocols=normalized_protocols,
                hop_count=hop_count or 1,
                confidence=confidence or 0.0,
                metadata=dict(metadata or {}),
            )
            state.pivot_routes[route_id] = route
        else:
            route.destination_host = destination_host
            route.destination_zone = destination_zone
            route.destination_cidr = destination_cidr
            route.source_host = source_host
            route.via_host = via_host
            route.session_id = session_id
            route.protocol = protocol
            if normalized_ports:
                route.allowed_ports = normalized_ports
            if normalized_protocols:
                route.protocols = normalized_protocols
            if hop_count is not None:
                route.hop_count = hop_count
            if confidence is not None:
                route.confidence = confidence
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
        destination_zone: str | None = None,
        destination_cidr: str | None = None,
        allowed_ports: set[int] | list[int] | None = None,
        protocols: set[str] | list[str] | None = None,
        hop_count: int | None = None,
        confidence: float | None = None,
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
            destination_zone=destination_zone,
            destination_cidr=destination_cidr,
            allowed_ports=allowed_ports,
            protocols=protocols,
            hop_count=hop_count,
            confidence=confidence,
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
        destination_zone: str | None = None,
        destination_cidr: str | None = None,
        port: int | None = None,
        protocol: str | None = None,
    ) -> PivotRouteRuntime | None:
        """Return the best currently usable route for one destination host."""

        candidates = [
            route
            for route in state.pivot_routes.values()
            if route.status == PivotRouteStatus.ACTIVE
            and self._route_session_usable(state, route)
            and self._route_matches_destination(
                route,
                destination_host=destination_host,
                destination_zone=destination_zone,
                destination_cidr=destination_cidr,
            )
            and (source_host is None or route.source_host == source_host)
            and self._route_matches_service(route, port=port, protocol=protocol)
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (
                0 if item.session_id is not None else 1,
                item.hop_count,
                -item.confidence,
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

    @staticmethod
    def _route_session_usable(state: RuntimeState, route: PivotRouteRuntime) -> bool:
        if route.session_id is None:
            return True
        session = state.sessions.get(route.session_id)
        if session is None:
            return False
        return session.status == SessionStatus.ACTIVE and session.is_session_usable()

    @staticmethod
    def _route_matches_destination(
        route: PivotRouteRuntime,
        *,
        destination_host: str,
        destination_zone: str | None,
        destination_cidr: str | None,
    ) -> bool:
        if route.destination_host == destination_host:
            return True
        if destination_zone is not None and route.destination_zone == destination_zone:
            return True
        if destination_cidr is not None and route.destination_cidr == destination_cidr:
            return True
        return False

    @staticmethod
    def _route_matches_service(route: PivotRouteRuntime, *, port: int | None, protocol: str | None) -> bool:
        if port is not None and route.allowed_ports and port not in route.allowed_ports:
            return False
        if protocol is None:
            return True
        protocol_key = protocol.strip().lower()
        route_protocols = {item.lower() for item in route.protocols}
        if route.protocol is not None:
            route_protocols.add(route.protocol.lower())
        return not route_protocols or protocol_key in route_protocols

    @staticmethod
    def _normalize_ports(values: set[int] | list[int] | None) -> set[int]:
        if values is None:
            return set()
        return {int(value) for value in values if 1 <= int(value) <= 65535}

    @staticmethod
    def _normalize_protocols(values: set[str] | list[str] | None) -> set[str]:
        if values is None:
            return set()
        return {str(value).strip().lower() for value in values if str(value).strip()}

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
