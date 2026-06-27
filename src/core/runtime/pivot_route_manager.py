"""Runtime pivot route manager.

This module manages transient route candidates and active pivot paths used to
reach downstream hosts during one operation.
"""

from __future__ import annotations

from src.core.models.runtime import PivotRouteRuntime, PivotRouteStatus, RuntimeState, utc_now


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

    def get_route(self, state: RuntimeState, route_id: str) -> PivotRouteRuntime:
        """Return one tracked pivot route."""

        try:
            return state.pivot_routes[route_id]
        except KeyError as exc:
            raise ValueError(f"pivot route '{route_id}' does not exist") from exc

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


__all__ = ["RuntimePivotRouteManager"]
