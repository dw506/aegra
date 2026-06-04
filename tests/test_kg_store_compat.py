from __future__ import annotations

from src.core.graph.kg_store import KnowledgeGraph
from src.core.models.kg_enums import EdgeType, NodeType


def test_kg_store_normalizes_legacy_node_type_aliases() -> None:
    assert KnowledgeGraph._normalize_node_type("Fingerprint") == NodeType.OBSERVATION
    assert KnowledgeGraph._normalize_node_type("WebEndpoint") == NodeType.SERVICE


def test_kg_store_normalizes_legacy_edge_type_aliases() -> None:
    assert KnowledgeGraph._normalize_edge_type("HOSTS_SERVICE") == EdgeType.HOSTS
    assert KnowledgeGraph._normalize_edge_type("HAS_FINGERPRINT") == EdgeType.RELATED_TO
