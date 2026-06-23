"""Network topology query helpers built from KG reachability facts."""
#KG 查询层 / 网络拓扑读模型层，基于 KG 中已经沉淀的边和节点，提供“哪些主机可达、哪些服务需要 pivot、有哪些 route 候选”的查询能力
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.kg import BaseEdge, BaseNode, CanReachEdge, NetworkZone, PivotsToEdge, Service
from src.core.models.kg_enums import EdgeType, NodeType


ReachabilityVia = Literal["direct", "session", "pivot", "inferred"]


class ReachabilityPath(BaseModel):
    """One known path from a source host to a target host or service."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_host: str
    destination_host: str
    service_id: str | None = None
    via: ReachabilityVia = "direct"
    route_id: str | None = None
    session_id: str | None = None
    protocol: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    hops: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteCandidate(BaseModel):
    """A compact KG-derived pivot route candidate."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    route_id: str | None = None
    source_host: str
    destination_host: str
    via_host: str | None = None
    session_id: str | None = None
    protocol: str | None = None
    destination_zone: str | None = None
    destination_cidr: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

#只读拓扑查询器
class NetworkTopology:
    """Read-only reachability and pivot queries over the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def reachable_paths(
        self,
        *,
        source_host: str | None = None,
        destination_host: str | None = None,
        service_id: str | None = None,
    ) -> list[ReachabilityPath]:
        """Return known reachability paths matching the optional filters."""

        paths = [
            path
            for edge in self.graph.list_edges(type=EdgeType.CAN_REACH)
            if (path := self._path_from_can_reach(edge)) is not None
        ]
        if source_host is not None:
            paths = [path for path in paths if path.source_host == source_host]
        if destination_host is not None:
            paths = [path for path in paths if path.destination_host == destination_host]
        if service_id is not None:
            paths = [path for path in paths if path.service_id == service_id]
        return sorted(paths, key=lambda item: (item.source_host, item.destination_host, item.service_id or "", item.route_id or ""))

    def direct_reachability(self, *, source_host: str, destination_host: str) -> list[ReachabilityPath]:
        """Return direct paths between two hosts."""

        return [
            path
            for path in self.reachable_paths(source_host=source_host, destination_host=destination_host)
            if path.via == "direct"
        ]

    def pivot_reachability(self, *, destination_host: str, source_host: str | None = None) -> list[ReachabilityPath]:
        """Return paths that require a pivot or runtime session."""

        return [
            path
            for path in self.reachable_paths(source_host=source_host, destination_host=destination_host)
            if path.via in {"pivot", "session"}
        ]

    def routes_to(self, destination_host: str) -> list[RouteCandidate]:
        """Return KG route candidates for a destination host."""

        candidates: list[RouteCandidate] = []
        for edge in self.graph.list_edges(type=EdgeType.PIVOTS_TO):
            candidate = self._candidate_from_pivot(edge)
            if candidate is not None and candidate.destination_host == destination_host:
                candidates.append(candidate)
        for path in self.pivot_reachability(destination_host=destination_host):
            candidates.append(
                RouteCandidate(
                    route_id=path.route_id,
                    source_host=path.source_host,
                    destination_host=path.destination_host,
                    via_host=path.hops[-2] if len(path.hops) >= 2 else None,
                    session_id=path.session_id,
                    protocol=path.protocol,
                    confidence=path.confidence,
                    metadata=path.metadata,
                )
            )
        return sorted(candidates, key=lambda item: (item.source_host, item.destination_host, item.route_id or ""))

    def zones_for_host(self, host_id: str) -> list[NetworkZone]:
        """Return network zones the host belongs to."""

        zones: list[NetworkZone] = []
        for edge in self.graph.list_edges(type=EdgeType.BELONGS_TO_ZONE, source=host_id):
            node = self._get_node(edge.target)
            if isinstance(node, NetworkZone):
                zones.append(node)
        return sorted(zones, key=lambda item: item.id)

    def reachable_services_from(self, source_host: str) -> list[ReachabilityPath]:
        """Return reachable service paths from one source host."""

        return [path for path in self.reachable_paths(source_host=source_host) if path.service_id is not None]

    def services_requiring_pivot(self, source_host: str) -> list[ReachabilityPath]:
        """Return reachable service paths that require a pivot or session."""

        return [
            path
            for path in self.reachable_services_from(source_host)
            if path.via in {"pivot", "session"}
        ]

    def _path_from_can_reach(self, edge: BaseEdge) -> ReachabilityPath | None:
        source_host = self._edge_string(edge, "source_host") or edge.source
        target_node = self._get_node(edge.target)
        service = target_node if isinstance(target_node, Service) else None
        destination_host = (
            self._edge_string(edge, "target_host")
            or self._edge_string(edge, "destination_host")
            or self._service_host(service)
            or edge.target
        )
        via = self._edge_string(edge, "via") or self._property_string(edge, "reachable_via") or "direct"
        if via not in {"direct", "session", "pivot", "inferred"}:
            via = "inferred"
        route_id = self._edge_string(edge, "route_id") or self._property_string(edge, "route_id")
        session_id = self._edge_string(edge, "session_id") or self._property_string(edge, "session_id")
        protocol = self._edge_string(edge, "protocol") or self._property_string(edge, "protocol")
        port = self._edge_int(edge, "port") or self._property_int(edge, "port")
        if service is not None:
            protocol = protocol or service.protocol
            port = port or service.port
        hops = [source_host]
        via_host = self._property_string(edge, "via_host")
        if via_host is not None and via_host not in hops:
            hops.append(via_host)
        if destination_host not in hops:
            hops.append(destination_host)
        return ReachabilityPath(
            source_host=source_host,
            destination_host=destination_host,
            service_id=self._edge_string(edge, "target_service") or (service.id if service is not None else None),
            via=via,  # type: ignore[arg-type]
            route_id=route_id,
            session_id=session_id,
            protocol=protocol,
            port=port,
            hops=hops,
            confidence=edge.confidence,
            metadata=dict(edge.properties),
        )

    def _candidate_from_pivot(self, edge: BaseEdge) -> RouteCandidate | None:
        source_host = self._edge_string(edge, "source_host") or edge.source
        destination_host = self._edge_string(edge, "destination_host") or edge.target
        return RouteCandidate(
            route_id=self._edge_string(edge, "route_id") or self._property_string(edge, "route_id"),
            source_host=source_host,
            destination_host=destination_host,
            via_host=self._edge_string(edge, "via_host") or self._property_string(edge, "via_host"),
            session_id=self._edge_string(edge, "session_id") or self._property_string(edge, "session_id"),
            protocol=self._edge_string(edge, "protocol") or self._property_string(edge, "protocol"),
            destination_zone=self._edge_string(edge, "destination_zone") or self._property_string(edge, "destination_zone"),
            destination_cidr=self._edge_string(edge, "destination_cidr") or self._property_string(edge, "destination_cidr"),
            confidence=edge.confidence,
            metadata=dict(edge.properties),
        )

    def _service_host(self, service: Service | None) -> str | None:
        if service is None:
            return None
        for edge in self.graph.list_edges(type=EdgeType.HOSTS, target=service.id):
            return edge.source
        host = service.properties.get("host") or service.properties.get("host_id")
        return str(host) if host else None

    def _get_node(self, node_id: str) -> BaseNode | None:
        try:
            return self.graph.get_node(node_id)
        except Exception:
            return None

    @staticmethod
    def _edge_string(edge: BaseEdge, field_name: str) -> str | None:
        value = getattr(edge, field_name, None)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _edge_int(edge: BaseEdge, field_name: str) -> int | None:
        value = getattr(edge, field_name, None)
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _property_string(edge: BaseEdge, key: str) -> str | None:
        value = edge.properties.get(key)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _property_int(edge: BaseEdge, key: str) -> int | None:
        try:
            return int(edge.properties.get(key))
        except (TypeError, ValueError):
            return None


__all__ = ["NetworkTopology", "ReachabilityPath", "RouteCandidate", "ReachabilityVia"]
