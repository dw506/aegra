from __future__ import annotations

import ast
from pathlib import Path

from src.core.workers.access_validation_worker import AccessValidationWorker
from src.core.workers.base import BaseWorkerAgent
from src.core.workers.goal_validation_worker import GoalValidationWorker
from src.core.workers.privilege_validation_worker import PrivilegeValidationWorker
from src.core.workers.registry import WorkerRegistry


def test_core_planner_does_not_import_agent_wrappers() -> None:
    for path, module in _imports_under(Path("src/core/planner")):
        assert not module.startswith("src.core.agents"), f"{path} imports {module}"


def test_core_perception_keeps_incalmo_external_except_compat_shim() -> None:
    for path, module in _imports_under(Path("src/core/perception")):
        if path.name == "c2_parser.py":
            continue
        assert not module.startswith("src.integrations.incalmo"), f"{path} imports {module}"


def test_worker_services_do_not_depend_on_incalmo_client() -> None:
    for path, module in _imports_under(Path("src/core/workers/services")):
        assert not module.startswith("src.integrations.incalmo"), f"{path} imports {module}"
        assert not module.endswith(".incalmo.client"), f"{path} imports {module}"


def test_execution_adapters_do_not_import_graph_or_result_applier_owners() -> None:
    banned_prefixes = (
        "src.core.runtime.result_applier",
        "src.core.graph",
        "src.core.models.kg",
        "src.core.models.ag",
        "src.core.models.tg",
    )
    for path, module in _imports_under(Path("src/core/execution/adapters")):
        assert not module.startswith(banned_prefixes), f"{path} imports {module}"


def test_worker_registry_default_excludes_legacy_workers() -> None:
    workers = WorkerRegistry.default().list_all()

    assert {type(worker) for worker in workers} == {
        AccessValidationWorker,
        GoalValidationWorker,
        PrivilegeValidationWorker,
    }
    assert all(isinstance(worker, BaseWorkerAgent) for worker in workers)


def _imports_under(root: Path) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append((path, node.module))
    return imports
