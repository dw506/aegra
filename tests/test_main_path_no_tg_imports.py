from __future__ import annotations

from pathlib import Path


def test_main_path_does_not_import_legacy_tg_components() -> None:
    files = [
        "src/app/orchestrator.py",
        "src/core/execution/execution_agent.py",
        "src/core/stage/registry.py",
        "src/core/runtime/result_applier.py",
    ]
    forbidden = [
        "TaskGraph",
        "tg_builder",
        "tg_merge",
        "SchedulerAgent",
        "LLMWorkerAgent",
        "TaskBuilderAgent",
        "StageTaskGraphBuilder",
    ]
    for file in files:
        text = Path(file).read_text(encoding="utf-8")
        for item in forbidden:
            assert item not in text, f"{item} should not appear in {file}"
