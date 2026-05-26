from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.graph.topology import NetworkTopology
from src.core.models.kg import CanReachEdge, Host, HostsEdge, NetworkZone, PivotsToEdge, Service


def build_topology_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_node(Host(id="operator", label="operator"))
    graph.add_node(Host(id="web-1", label="web-1"))
    graph.add_node(Host(id="db-1", label="db-1"))
    graph.add_node(NetworkZone(id="zone-db", label="db subnet", cidr="10.20.0.0/24", zone_kind="internal"))
    graph.add_node(Service(id="svc-db", label="postgres", port=5432, protocol="tcp"))
    graph.add_edge(HostsEdge(id="hosts::db-1::svc-db", label="hosts", source="db-1", target="svc-db"))
    graph.add_edge(
        CanReachEdge(
            id="reach::operator::web-1",
            label="operator reaches web",
            source="operator",
            target="web-1",
            source_host="operator",
            target_host="web-1",
            via="direct",
        )
    )
    graph.add_edge(
        CanReachEdge(
            id="reach::web-1::svc-db",
            label="web reaches postgres",
            source="web-1",
            target="svc-db",
            source_host="web-1",
            target_host="db-1",
            target_service="svc-db",
            via="pivot",
            route_id="route-db",
            session_id="sess-web",
            protocol="tcp",
            port=5432,
            properties={"via_host": "web-1"},
            confidence=0.8,
        )
    )
    graph.add_edge(
        PivotsToEdge(
            id="pivot::web-1::db-1",
            label="web pivots to db",
            source="web-1",
            target="db-1",
            source_host="web-1",
            destination_host="db-1",
            route_id="route-db",
            session_id="sess-web",
            protocol="tcp",
            destination_zone="zone-db",
            destination_cidr="10.20.0.0/24",
        )
    )
    return graph


def test_network_topology_extracts_reachability_paths_and_routes() -> None:
    topology = NetworkTopology(build_topology_graph())

    service_paths = topology.reachable_services_from("web-1")
    routes = topology.routes_to("db-1")

    assert len(service_paths) == 1
    assert service_paths[0].destination_host == "db-1"
    assert service_paths[0].service_id == "svc-db"
    assert service_paths[0].via == "pivot"
    assert service_paths[0].route_id == "route-db"
    assert routes[0].route_id == "route-db"
    assert routes[0].destination_cidr == "10.20.0.0/24"


def test_network_topology_finds_services_requiring_pivot() -> None:
    topology = NetworkTopology(build_topology_graph())

    paths = topology.services_requiring_pivot("web-1")

    assert [path.service_id for path in paths] == ["svc-db"]
