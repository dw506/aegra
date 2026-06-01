from __future__ import annotations

import ast
from pathlib import Path

from src.app.orchestrator import AppOrchestrator
from src.app.settings import AppSettings
from src.core.agents.agent_protocol import AgentKind, AgentContext, AgentInput
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.packy_llm import PackyLLMResponse
from src.core.scheduling.llm_scheduler_advisor import LLMSchedulerAdvisor
from src.core.workers.base import BaseWorkerAgent
from src.core.workers.llm_worker import LLMWorkerAgent
from src.core.workers.registry import WorkerRegistry


def test_core_planner_does_not_import_agent_wrappers() -> None:
    for path, module in _imports_under(Path("src/core/planner")):
        assert not module.startswith("src.core.agents"), f"{path} imports {module}"


def test_core_perception_has_no_external_c2_dependency() -> None:
    for path, module in _imports_under(Path("src/core/perception")):
        assert ".c2" not in module.lower(), f"{path} imports {module}"


def test_worker_services_do_not_depend_on_external_c2_client() -> None:
    for path, module in _imports_under(Path("src/core/workers/services")):
        assert ".c2" not in module.lower(), f"{path} imports {module}"


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


def test_default_pipeline_contains_llm_agents() -> None:
    pipeline = AppOrchestrator._build_default_pipeline(AppSettings())

    assert [agent.name for agent in pipeline.registry.list_by_kind(AgentKind.PLANNER)] == ["planner_agent"]
    assert [agent.name for agent in pipeline.registry.list_by_kind(AgentKind.SCHEDULER)] == ["scheduler_agent"]
    assert pipeline.registry.list_by_kind(AgentKind.CRITIC) == []
    assert pipeline.registry.list_by_kind(AgentKind.SUPERVISOR) == []
    assert [agent.name for agent in pipeline.registry.list_by_kind(AgentKind.WORKER)] == ["llm_worker_agent"]


def test_orchestrator_main_chain_is_two_graph_stage_dispatch_and_disables_feedback() -> None:
    source = Path("src/app/orchestrator.py").read_text(encoding="utf-8")
    run_cycle_source = source.split("    def run_legacy_operation_cycle", maxsplit=1)[0]

    assert "StageDispatcher(" in run_cycle_source
    assert "AttackLogExtractor(" in run_cycle_source
    assert "schedule_ready_tasks(" not in run_cycle_source
    assert "CandidateTaskService" not in run_cycle_source
    assert 'state.execution.metadata["task_graph"]' not in run_cycle_source
    assert "_run_feedback_phase(" not in run_cycle_source
    assert 'cycle_name="feedback_disabled"' in run_cycle_source


def test_orchestrator_does_not_import_legacy_tg_scheduler_or_worker_agents() -> None:
    imports = {module for _, module in _imports_under_file(Path("src/app/orchestrator.py"))}

    assert "src.core.models.tg" not in imports
    assert "src.core.agents.scheduler_agent" not in imports
    assert "src.core.workers.llm_worker" not in imports


def test_result_applier_main_path_does_not_import_task_graph_builder() -> None:
    source = Path("src/core/runtime/result_applier.py").read_text(encoding="utf-8")
    main_path_source = source.split("    def _apply_schedule_decision", maxsplit=1)[0]

    assert "TaskGraphBuilder" not in main_path_source
    assert "StageTaskGraphBuilder" not in main_path_source


def test_scheduler_agent_does_not_hardcode_final_dispatch() -> None:
    source = Path("src/core/agents/scheduler_agent.py").read_text(encoding="utf-8")

    assert "schedule_ready_stage_tasks" not in source
    assert "RuntimeScheduler" not in source
    assert "choose_next_task(" in source


def test_scheduler_llm_unavailable_has_no_deterministic_fallback() -> None:
    scheduler = SchedulerAgent()
    result = scheduler.run(
        AgentInput(
            context=AgentContext(operation_id="op-1"),
            raw_payload={
                "graph_context": {"operation_id": "op-1"},
                "candidate_tasks": [
                    {
                        "task_id": "stage-recon-1",
                        "stage_type": "RECON_STAGE",
                        "objective": "Recon target",
                    }
                ],
                "runtime_summary": {},
                "policy_context": {},
                "tool_catalog": {},
                "recent_outcomes": [],
            },
        )
    )

    decision = result.output.decisions[0]["schedule_decision"]
    assert decision["decision"] == "blocked"
    assert decision["task_id"] is None
    assert decision["metadata"]["accepted"] is False
    assert "scheduler_llm_unavailable" in decision["metadata"]["reason"]


def test_agent_naming_boundary_for_scheduler() -> None:
    source = Path("src/core/agents/scheduler_agent.py").read_text(encoding="utf-8")

    assert "advisor: LLMSchedulerAdvisor" in source
    assert "self._advisor.choose_next_task" in source
    assert "select_schedulable_tasks" not in source


class FakeSchedulerLLMClient:
    config = type("Config", (), {"model": "test-model"})()

    def complete_chat(self, **kwargs):
        return PackyLLMResponse(
            model="test-model",
            text=(
                '{"decision":"dispatch","task_id":"task-1","worker_id":"llm_worker_agent",'
                '"rationale":"dispatch","confidence":0.8,'
                '"scheduled_task":{"task_id":"task-1","stage_type":"RECON_STAGE",'
                '"objective":"recon","known_facts":["fact one"],"target_refs":[],'
                '"constraints":[],"allowed_tools":[],"success_criteria":[],'
                '"policy_context":{},"runtime_context":{}},'
                '"runtime_update_intents":[],"metadata":{"accepted":true}}'
            ),
        )


def test_scheduler_advisor_normalizes_llm_string_known_facts() -> None:
    decision = LLMSchedulerAdvisor(client=FakeSchedulerLLMClient()).choose_next_task(
        operation_id="op-1",
        graph_context={"operation_id": "op-1"},
        candidate_tasks=[{"task_id": "task-1"}],
        runtime_summary={},
        policy_context={},
        tool_catalog={},
        recent_outcomes=[],
    )

    assert decision.decision == "dispatch"
    assert decision.scheduled_task is not None
    assert decision.scheduled_task.known_facts == [{"summary": "fact one"}]


class FakeWaitSchedulerLLMClient:
    config = type("Config", (), {"model": "test-model"})()

    def complete_chat(self, **kwargs):
        return PackyLLMResponse(
            model="test-model",
            text=(
                '{"decision":"wait","task_id":null,"worker_id":null,'
                '"rationale":"no candidates","confidence":0.8,'
                '"scheduled_task":{"task_id":"","stage_type":"","objective":""},'
                '"runtime_update_intents":[],"metadata":{"accepted":true}}'
            ),
        )


def test_scheduler_advisor_normalizes_wait_without_scheduled_task() -> None:
    decision = LLMSchedulerAdvisor(client=FakeWaitSchedulerLLMClient()).choose_next_task(
        operation_id="op-1",
        graph_context={"operation_id": "op-1"},
        candidate_tasks=[],
        runtime_summary={},
        policy_context={},
        tool_catalog={},
        recent_outcomes=[],
    )

    assert decision.decision == "wait"
    assert decision.scheduled_task is None


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
