from __future__ import annotations

from src.core.agents.state_writer import StateWriterAgent
from src.core.models.kg_enums import EdgeType


def test_state_writer_normalizes_legacy_host_service_edge_alias() -> None:
    assert StateWriterAgent._normalize_enum_value("HOSTS_SERVICE", EdgeType) == EdgeType.HOSTS.value
    assert StateWriterAgent._normalize_enum_value("HAS_FINGERPRINT", EdgeType) == EdgeType.RELATED_TO.value
