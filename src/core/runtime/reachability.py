"""Runtime reachability propagation helpers."""

from __future__ import annotations

from typing import Any

from src.core.models.events import AgentTaskResult
from src.core.models.runtime import PivotRouteRuntime, RuntimeState, TaskRuntime
from src.core.runtime.pivot_route_manager import RuntimePivotRouteManager


class ReachabilityPropagator:
    """Translate worker reachability output into runtime route state."""

    def __init__(self, pivot_route_manager: RuntimePivotRouteManager | None = None) -> None:
        self._pivot_route_manager = pivot_route_manager or RuntimePivotRouteManager()

    def sync_from_task_result(self, *, state: RuntimeState, result: AgentTaskResult) -> PivotRouteRuntime | None:
        """Refresh runtime pivot routes from one worker result payload."""

        reachability = self._dict(result.outcome_payload.get("reachability"))
        route_view = self._dict(result.outcome_payload.get("selected_route"))
        route_id = self._string(route_view.get("route_id")) or self._string(reachability.get("route_id"))
        destination_host = (
            self._string(route_view.get("destination_host"))
            or self._string(reachability.get("target_id"))
            or self._string(result.outcome_payload.get("bound_target"))
            or self._string(result.outcome_payload.get("target_id"))
        )
        task = state.execution.tasks.get(result.task_id)
        if destination_host is None:
            destination_host = self._string((task.metadata if task is not None else {}).get("bound_target"))
        if destination_host is None:
            return None
        if self._string(reachability.get("via")) not in {"pivot", "session"} and route_id is None and not route_view:
            return None
        route = self._pivot_route_manager.refresh_from_reachability(
            state,
            route_id=route_id,
            destination_host=destination_host,
            reachable=bool(reachability.get("reachable", result.status.value == "succeeded")),
            source_host=self._string(route_view.get("source_host")) or self._string(reachability.get("source_id")),
            via_host=self._string(route_view.get("via_host")) or self._string(reachability.get("via_host")),
            session_id=self._string(route_view.get("session_id")) or self._result_session_id(result=result, task=task),
            protocol=self._string(route_view.get("protocol")) or self._string(reachability.get("protocol")),
            destination_zone=self._string(route_view.get("destination_zone")) or self._string(reachability.get("destination_zone")),
            destination_cidr=self._string(route_view.get("destination_cidr")) or self._string(reachability.get("destination_cidr")),
            allowed_ports=self._ports(route_view.get("allowed_ports") or reachability.get("allowed_ports") or reachability.get("port")),
            protocols=self._protocols(route_view.get("protocols") or reachability.get("protocols")),
            hop_count=self._int(route_view.get("hop_count") or reachability.get("hop_count")),
            confidence=self._float(route_view.get("confidence") or reachability.get("confidence")),
            metadata={
                "source_task_id": result.task_id,
                "result_status": result.status.value,
                "reachability": reachability,
                "selected_route": route_view,
            },
        )
        if task is not None:
            task.metadata["selected_route_id"] = route.route_id
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
    def _result_session_id(cls, *, result: AgentTaskResult, task: TaskRuntime | None) -> str | None:
        session_id = cls._string(result.outcome_payload.get("session_id"))
        if session_id is not None:
            return session_id
        if task is not None:
            return cls._string(task.metadata.get("session_id"))
        return None


__all__ = ["ReachabilityPropagator"]
