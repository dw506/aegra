from __future__ import annotations

import ast
from pathlib import Path

from src.core.workers.base import BaseWorkerAgent
from src.core.workers.llm_worker import LLMWorkerAgent
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


def test_visualization_is_not_imported_by_parser_worker_or_execution_layers() -> None:
    roots = [
        Path("src/core/perception"),
        Path("src/core/workers"),
        Path("src/core/execution"),
    ]
    for root in roots:
        for path, module in _imports_under(root):
            assert not module.startswith("src.core.visualization"), f"{path} imports {module}"


def test_primary_workers_route_external_execution_through_execution_layer() -> None:
    primary_worker_paths = [
        Path("src/core/workers/recon_worker.py"),
        Path("src/core/workers/web_discovery_worker.py"),
    ]
    for path in primary_worker_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {module for _, module in _imports_under_file(path)}
        assert "urllib.request" not in imports, f"{path} imports urllib.request directly"
        assert "src.core.workers.tool_runner" not in imports, f"{path} imports legacy ToolRunner directly"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "urlopen", f"{path} calls urlopen directly"


def test_worker_registry_default_excludes_legacy_workers() -> None:
    workers = WorkerRegistry.default().list_all()

    assert {type(worker) for worker in workers} == {LLMWorkerAgent}
    assert all(isinstance(worker, BaseWorkerAgent) for worker in workers)


def _imports_under(root: Path) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*.py")):
        imports.extend(_imports_under_file(path))
    return imports


def _imports_under_file(path: Path) -> list[tuple[Path, str]]:
    imports: list[tuple[Path, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((path, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((path, node.module))
    return imports
