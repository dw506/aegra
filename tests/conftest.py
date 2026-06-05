from __future__ import annotations

from pathlib import Path

import pytest


LEGACY_IMPORT_MARKERS = (
    "src.core.models.tg",
    "src.core.graph.tg_builder",
    "src.core.graph.tg_merge",
    "src.core.scheduling.",
    "src.core.agents.scheduler_agent",
    "src.core.agents.critic",
    "src.core.workers.llm_worker",
    "src.core.planning.stage_task_builder",
    "pipeline=pipeline",
    "StageTaskGraphBuilder",
    "StageTask",
    "PlannerResult",
    "MissionPlannerResult",
)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    if config.getoption("--run-legacy-tg", default=False):
        return False
    try:
        source = Path(collection_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return any(marker in source for marker in LEGACY_IMPORT_MARKERS)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-legacy-tg",
        action="store_true",
        default=False,
        help="collect legacy TaskGraph/Scheduler/LLMWorker compatibility tests",
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
