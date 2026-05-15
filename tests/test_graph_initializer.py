from __future__ import annotations

from src.core.graph.graph_initializer import initialize_graph_memory, normalize_initial_target
from src.core.graph.graph_memory_store import GraphMemoryStore
from src.core.models.ag import ActionNodeType, StateNodeType
from src.core.models.kg import Goal, Host, NetworkZone
from src.core.models.tg import BaseTaskNode, TaskType


def test_normalize_initial_target_accepts_ip_and_url() -> None:
    ip_target = normalize_initial_target("192.0.2.10")
    url_target = normalize_initial_target("https://example.com:8443/app")

    assert ip_target.kind == "ip"
    assert ip_target.address == "192.0.2.10"
    assert ip_target.hostname is None

    assert url_target.kind == "url"
    assert url_target.host_value == "example.com"
    assert url_target.hostname == "example.com"
    assert url_target.scheme == "https"
    assert url_target.port == 8443


def test_initialize_graph_memory_builds_and_persists_initial_graphs(tmp_path) -> None:
    store = GraphMemoryStore(tmp_path / "runtime-store")

    result = initialize_graph_memory(
        operation_id="op-init-1",
        target="192.0.2.10",
        store=store,
    )

    assert result.host_id
    assert result.goal_id
    assert result.scope_id
    assert result.initial_action_ids
    assert result.initial_task_ids

    host = result.kg.get_node(result.host_id)
    goal = result.kg.get_node(result.goal_id)
    scope = result.kg.get_node(result.scope_id)
    assert isinstance(host, Host)
    assert isinstance(goal, Goal)
    assert isinstance(scope, NetworkZone)
    assert host.address == "192.0.2.10"

    assert result.ag.find_states(StateNodeType.HOST_KNOWN)
    initial_actions = result.ag.find_actions(ActionNodeType.ENUMERATE_HOST)
    assert [action.id for action in initial_actions] == result.initial_action_ids

    initial_tasks = [
        node
        for node in result.tg.list_nodes()
        if isinstance(node, BaseTaskNode) and node.source_action_id in result.initial_action_ids
    ]
    assert [task.id for task in initial_tasks] == result.initial_task_ids
    assert initial_tasks[0].task_type == TaskType.ASSET_CONFIRMATION

    restored_kg = store.load_kg("op-init-1")
    restored_ag = store.load_ag("op-init-1")
    restored_tg = store.load_tg("op-init-1")
    assert restored_kg.get_node(result.host_id).label == "192.0.2.10"
    assert restored_ag.find_actions(ActionNodeType.ENUMERATE_HOST)
    assert [
        node.id
        for node in restored_tg.list_nodes()
        if isinstance(node, BaseTaskNode)
    ] == result.initial_task_ids


def test_initialize_graph_memory_preserves_url_target_metadata(tmp_path) -> None:
    result = initialize_graph_memory(
        operation_id="op-url-1",
        target="https://example.com:8443/app",
        root_dir=tmp_path,
        goal_label="Validate example.com",
    )

    host = result.kg.get_node(result.host_id)
    assert isinstance(host, Host)
    assert host.hostname == "example.com"
    assert host.properties["url"] == "https://example.com:8443/app"
    assert host.properties["scheme"] == "https"
    assert host.properties["port"] == 8443
    assert result.kg.get_node(result.goal_id).label == "Validate example.com"
