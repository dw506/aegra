from __future__ import annotations

from src.app.settings import AppSettings
from src.core.visualization.graph_serializer import build_visual_snapshot


def test_legacy_tg_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("AEGRA_LEGACY_TG", raising=False)

    assert AppSettings.from_env().legacy_tg is False


def test_visual_snapshot_excludes_legacy_tg_by_default() -> None:
    snapshot = build_visual_snapshot(
        operation_id="op-no-legacy-tg",
        kg_payload={"nodes": [], "edges": []},
        ag_payload={"nodes": [], "edges": []},
        tg_payload={"nodes": [{"id": "legacy-task"}], "edges": []},
        runtime_state=None,
    )

    assert "tg" not in snapshot.graphs
