from __future__ import annotations

from pathlib import Path

import pytest


LEGACY_IMPORT_MARKERS = (
    "src.core.models.tg",
    "src.core.graph.tg_builder",
    "src.core.graph.tg_merge",
    "src.core.scheduling.",
    "src.core.agents.scheduler_agent",
    "src.core.workers.llm_worker",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    cache: dict[Path, bool] = {}
    for item in items:
        path = Path(str(item.fspath))
        if path not in cache:
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                source = ""
            cache[path] = any(marker in source for marker in LEGACY_IMPORT_MARKERS)
        if cache[path]:
            item.add_marker(pytest.mark.legacy)
