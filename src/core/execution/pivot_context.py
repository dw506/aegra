"""Resolve pivot routes into adapter-facing execution context."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.execution.tool_plan import ToolPlan
from src.core.models.runtime import PivotRouteRuntime, RuntimeState, SessionRuntime


class PivotExecutionContext(BaseModel):
    """Normalized execution context derived from a selected pivot route."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    route_id: str | None = None
    session_id: str | None = None
    adapter: str = Field(default="local_shell", min_length=1)
    transport_kind: str | None = None
    proxy_url: str | None = None
    tunnel_endpoint: str | None = None
    network_namespace: str | None = None
    target_agent_ref: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    wrapper: list[str] = Field(default_factory=list)
    route: dict[str, Any] = Field(default_factory=dict)
    session: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_pivot(self) -> bool:
        return self.route_id is not None


class PivotExecutionContextResolver:
    """Turn ToolPlan route/session hints into a concrete execution context."""

    def resolve(self, plan: ToolPlan, runtime_state: RuntimeState | None = None) -> PivotExecutionContext:
        route = self._resolve_route(plan, runtime_state)
        session = self._resolve_session(plan, runtime_state, route)
        route_payload = route.model_dump(mode="json") if route is not None else self._mapping(plan.args.get("selected_route"))
        session_payload = session.model_dump(mode="json") if session is not None else {}

        route_transport = self._mapping(route.metadata.get("transport")) if route is not None else {}
        session_endpoint = self._mapping(session.metadata.get("execution_endpoint")) if session is not None else {}
        plan_transport = self._mapping(plan.metadata.get("transport")) | self._mapping(plan.args.get("transport"))

        merged = route_transport | session_endpoint | plan_transport
        proxy_url = self._string(
            plan.args.get("proxy_url")
            or plan.metadata.get("proxy_url")
            or merged.get("proxy_url")
        )
        namespace = self._string(
            plan.args.get("network_namespace")
            or plan.metadata.get("network_namespace")
            or merged.get("namespace")
            or merged.get("network_namespace")
        )
        tunnel_endpoint = self._string(
            plan.args.get("tunnel_endpoint")
            or plan.metadata.get("tunnel_endpoint")
            or merged.get("tunnel_endpoint")
            or merged.get("endpoint")
        )
        target_agent_ref = self._string(
            plan.target_agent_ref
            or plan.args.get("agent_id")
            or plan.metadata.get("agent_id")
            or merged.get("agent_id")
        )
        adapter = self._string(plan.adapter or merged.get("adapter")) or "local_shell"
        env = self._proxy_env(proxy_url) | self._string_mapping(merged.get("env"))

        return PivotExecutionContext(
            route_id=route.route_id if route is not None else self._string(plan.args.get("route_id") or plan.metadata.get("route_id")),
            session_id=(
                route.session_id
                if route is not None and route.session_id is not None
                else self._string(plan.args.get("session_id") or plan.metadata.get("session_id"))
            ),
            adapter=adapter,
            transport_kind=self._string(merged.get("kind") or merged.get("transport_kind")),
            proxy_url=proxy_url,
            tunnel_endpoint=tunnel_endpoint,
            network_namespace=namespace,
            target_agent_ref=target_agent_ref,
            env=env,
            wrapper=self._wrapper(namespace),
            route=route_payload,
            session=session_payload,
        )

    def _resolve_route(self, plan: ToolPlan, runtime_state: RuntimeState | None) -> PivotRouteRuntime | None:
        if runtime_state is None:
            return None
        route_id = self._string(
            plan.args.get("route_id")
            or plan.args.get("selected_route_id")
            or plan.metadata.get("route_id")
            or plan.metadata.get("selected_route_id")
        )
        if route_id is None:
            return None
        route = runtime_state.pivot_routes.get(route_id)
        if route is None or not route.is_usable():
            return None
        if route.session_id is not None:
            session = runtime_state.sessions.get(route.session_id)
            if session is None or not session.is_session_usable():
                return None
        return route

    def _resolve_session(
        self,
        plan: ToolPlan,
        runtime_state: RuntimeState | None,
        route: PivotRouteRuntime | None,
    ) -> SessionRuntime | None:
        if runtime_state is None:
            return None
        session_id = route.session_id if route is not None else self._string(plan.args.get("session_id") or plan.metadata.get("session_id"))
        if session_id is None:
            return None
        session = runtime_state.sessions.get(session_id)
        if session is None or not session.is_session_usable():
            return None
        return session

    @staticmethod
    def _proxy_env(proxy_url: str | None) -> dict[str, str]:
        if proxy_url is None:
            return {}
        return {
            "ALL_PROXY": proxy_url,
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "all_proxy": proxy_url,
            "http_proxy": proxy_url,
            "https_proxy": proxy_url,
        }

    @staticmethod
    def _wrapper(namespace: str | None) -> list[str]:
        if namespace is None:
            return []
        return ["ip", "netns", "exec", namespace]

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _string_mapping(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items() if item is not None}

    @staticmethod
    def _string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["PivotExecutionContext", "PivotExecutionContextResolver"]
