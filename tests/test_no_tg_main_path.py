from __future__ import annotations

import inspect

from src.app.orchestrator import AppOrchestrator


def test_run_operation_cycle_main_path_does_not_use_tg_scheduler_or_task_graph_metadata() -> None:
    source = inspect.getsource(AppOrchestrator.run_operation_cycle)

    assert "schedule_ready_tasks" not in source
    assert "CandidateTaskService" not in source
    assert "task_graph" not in source
    assert "save_tg" not in source
    assert 'graph="tg"' not in source
