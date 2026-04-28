from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from kg.enums import EdgeType, EntityStatus, NodeType
from kg.models import GraphChange, GraphEntityRef, Host, HostsEdge


def test_host_model_defaults() -> None:
    host = Host(id="host-1", label="Gateway", hostname="gateway.local", tags={"prod"})

    assert host.type == NodeType.HOST
    assert host.status == EntityStatus.OBSERVED
    assert host.confidence == 1.0
    assert "prod" in host.tags


def test_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        Host(id="host-1", label="bad", confidence=1.2)


def test_last_seen_before_first_seen_raises() -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ValidationError):
        Host(
            id="host-1",
            label="bad-window",
            first_seen=now,
            last_seen=now - timedelta(seconds=1),
        )


def test_graph_entity_ref_key_is_stable() -> None:
    ref = GraphEntityRef(entity_id="obs-1", entity_kind="node", entity_type="Observation")

    assert ref.key() == "node:obs-1"


def test_edge_model_uses_literal_type() -> None:
    edge = HostsEdge(
        id="edge-1",
        label="hosts",
        source="host-1",
        target="svc-1",
    )

    assert edge.type == EdgeType.HOSTS


def test_graph_change_accepts_serializable_payloads() -> None:
    change = GraphChange(
        operation="create",
        entity_ref=GraphEntityRef(entity_id="host-1", entity_kind="node"),
        after={"id": "host-1", "label": "Gateway"},
    )

    assert change.after is not None
