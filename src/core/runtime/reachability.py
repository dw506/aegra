"""Runtime reachability propagation helpers."""

from __future__ import annotations

from typing import Any

from src.core.execution.models import ExecutionResult
from src.core.models.runtime import PivotRouteRuntime, RuntimeState
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager


class ReachabilityPropagator:
    """Translate worker reachability output into runtime route state."""

    def __init__(self, pivot_route_manager: RuntimePivotRouteManager | None = None) -> None:
        self._pivot_route_manager = pivot_route_manager or RuntimePivotRouteManager()

    def sync_from_execution_result(self, *, state: RuntimeState, result: ExecutionResult) -> PivotRouteRuntime | None:
        """Refresh runtime pivot routes from one stage result."""

        reachability = self._dict(result.runtime_hints.get("reachability"))
        route_view = self._dict(result.runtime_hints.get("selected_route"))
        route_id = self._string(route_view.get("route_id")) or self._string(reachability.get("route_id"))
        destination_host = (
            self._string(route_view.get("destination_host"))
            or self._string(reachability.get("target_id"))
            or self._string(result.runtime_hints.get("bound_target"))
            or self._string(result.runtime_hints.get("target_id"))
        )
        if destination_host is None:
            return None
        if self._string(reachability.get("via")) not in {"pivot", "session"} and route_id is None and not route_view:
            return None
        route = self._pivot_route_manager.refresh_from_reachability(
            state,
            route_id=route_id,
            destination_host=destination_host,
            reachable=bool(reachability.get("reachable", result.status in {"success", "succeeded", "partial"})),
            source_host=self._string(route_view.get("source_host")) or self._string(reachability.get("source_id")),
            via_host=self._string(route_view.get("via_host")) or self._string(reachability.get("via_host")),
            session_id=self._string(route_view.get("session_id")) or self._result_session_id(result=result),
            protocol=self._string(route_view.get("protocol")) or self._string(reachability.get("protocol")),
            destination_zone=self._string(route_view.get("destination_zone")) or self._string(reachability.get("destination_zone")),
            destination_cidr=self._string(route_view.get("destination_cidr")) or self._string(reachability.get("destination_cidr")),
            allowed_ports=self._ports(route_view.get("allowed_ports") or reachability.get("allowed_ports") or reachability.get("port")),
            protocols=self._protocols(route_view.get("protocols") or reachability.get("protocols")),
            hop_count=self._int(route_view.get("hop_count") or reachability.get("hop_count")),
            confidence=self._float(route_view.get("confidence") or reachability.get("confidence")),
            metadata={
                "source_task_id": result.execution_id,
                "result_status": result.status,
                "reachability": reachability,
                "selected_route": route_view,
            },
        )
        return route

    @staticmethod
    def _dict(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _ports(cls, value: Any) -> set[int]:
        if value is None:
            return set()
        values = value if isinstance(value, (list, set, tuple)) else [value]
        ports: set[int] = set()
        for item in values:
            port = cls._int(item)
            if port is not None and 1 <= port <= 65535:
                ports.add(port)
        return ports

    @staticmethod
    def _protocols(value: Any) -> set[str]:
        if value is None:
            return set()
        values = value if isinstance(value, (list, set, tuple)) else [value]
        return {str(item).strip().lower() for item in values if str(item).strip()}

    @classmethod
    def _result_session_id(cls, *, result: ExecutionResult) -> str | None:
        return cls._string(result.runtime_hints.get("session_id"))


__all__ = ["ReachabilityPropagator"]
