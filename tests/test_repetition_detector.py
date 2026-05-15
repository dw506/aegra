from __future__ import annotations

from src.core.agents.agent_protocol import AgentContext, AgentInput, GraphRef, GraphScope
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.planner import PlannerAgent
from src.core.graph.ag_projector import AttackGraphProjector
from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import GraphRef as AGGraphRef
from src.core.models.kg import Goal, Host, Service, TargetsEdge
from src.core.models.kg_enums import EntityStatus
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime, TaskRuntimeStatus, WorkerRuntime, WorkerStatus
from src.core.models.tg import TaskGraph, TaskStatus, TaskType
from src.core.graph.kg_store import KnowledgeGraph
from src.core.runtime.repetition_detector import RepetitionAction, RepetitionDetector


def _task(port: int = 80, *, tool: str = "nmap") -> TaskCandidate:
    return TaskCandidate(
        source_action_id=f"scan-{port}",
        task_type=TaskType.SERVICE_VALIDATION,
        input_bindings={
            "tool_hint": tool,
            "port": port,
            "flags": ["-sV"],
            "expected_evidence": ["service banner"],
        },
        target_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
        expected_output_refs=[AGGraphRef(graph="query", ref_id="service-evidence", ref_type="ExpectedEvidence")],
    )


def test_repetition_detector_matches_same_target_same_task() -> None:
    detector = RepetitionDetector()
    detector.record_task(_task(80), task_id="task-old", status="failed")

    decision = detector.decide(_task(80))

    assert decision.action == RepetitionAction.REJECT
    assert decision.matched_task_ids == ["task-old"]
    assert decision.similarity == 1.0


def test_repetition_detector_does_not_match_different_port() -> None:
    detector = RepetitionDetector()
    detector.record_task(_task(80), task_id="task-old", status="failed")

    decision = detector.decide(_task(443))

    assert decision.action == RepetitionAction.ALLOW
    assert decision.matched_task_ids == []


def test_repetition_detector_rejects_repeated_failed_task() -> None:
    detector = RepetitionDetector()
    failed = detector.record_task(_task(8080), task_id="task-failed", status="blocked")

    decision = detector.decide(
        {
            "task_id": "task-new",
            "task_type": TaskType.SERVICE_VALIDATION.value,
            "input_bindings": {
                "tool": "nmap",
                "port": 8080,
                "flags": ["-sV"],
                "expected_evidence": ["service banner"],
            },
            "target_refs": [{"graph": "kg", "ref_id": "host-1", "ref_type": "Host"}],
            "expected_output_refs": [{"graph": "query", "ref_id": "service-evidence", "ref_type": "ExpectedEvidence"}],
        }
    )

    assert failed.signature_hash == decision.signature_hash
    assert decision.action == RepetitionAction.REJECT
    assert decision.reason == "exact repeat of failed or blocked task"


def test_repetition_detector_skips_repeated_completed_task() -> None:
    detector = RepetitionDetector()
    detector.record_task(_task(80), task_id="task-complete", status="succeeded")

    decision = detector.decide(_task(80))

    assert decision.action == RepetitionAction.SKIP
    assert decision.allow is False
    assert decision.skip is True
    assert decision.matched_task_ids == ["task-complete"]


def _planner_input(repetition_history: list[dict] | None = None) -> AgentInput:
    kg = KnowledgeGraph()
    kg.add_node(Host(id="host-1", label="Gateway", status=EntityStatus.VALIDATED, confidence=0.95))
    kg.add_node(Service(id="svc-1", label="HTTP", confidence=0.75))
    kg.add_node(Goal(id="goal-1", label="Validate target", category="service", confidence=0.9))
    kg.add_edge(TargetsEdge(id="e-goal-host", label="targets", source="goal-1", target="host-1"))
    ag = AttackGraphProjector().project(kg)
    goal_node = ag.get_goal_nodes()[0]
    return AgentInput(
        graph_refs=[
            GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
            GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode"),
        ],
        context=AgentContext(operation_id="op-repetition-planner-test"),
        raw_payload={
            "ag_graph": ag.to_dict(),
            "goal_refs": [GraphRef(graph=GraphScope.AG, ref_id=goal_node.id, ref_type="GoalNode").model_dump(mode="json")],
            "planning_context": {
                "top_k": 3,
                "max_depth": 2,
                "repetition_history": repetition_history or [],
            },
        },
    )


def test_planner_repetition_history_rejects_failed_candidate() -> None:
    first = PlannerAgent().run(_planner_input())
    candidate = first.output.decisions[0]["payload"]["planning_candidate"]
    task = candidate["task_candidates"][0]
    history = [{"task_id": "old-failed-task", "status": "failed", "task": task}]

    result = PlannerAgent().run(_planner_input(history))

    assert result.success is True
    assert len(result.output.decisions) < len(first.output.decisions)
    assert any("repetition detector rejected" in log for log in result.output.logs)


def test_planner_repetition_history_skips_completed_candidate() -> None:
    first = PlannerAgent().run(_planner_input())
    candidate = first.output.decisions[0]["payload"]["planning_candidate"]
    task = candidate["task_candidates"][0]
    history = [{"task_id": "old-completed-task", "status": "succeeded", "task": task}]

    result = PlannerAgent().run(_planner_input(history))

    repeated = [
        decision["payload"]["planning_candidate"]
        for decision in result.output.decisions
        if decision["payload"]["planning_candidate"]["task_candidates"][0]["source_action_id"] == task["source_action_id"]
    ]
    assert repeated == []
    assert any("repetition detector skipped satisfied" in log for log in result.output.logs)


def test_scheduler_repetition_detector_rejects_repeat_failed_task_before_assignment() -> None:
    task = TaskBuilderAgentShim.create_task(_task(80))
    task.status = TaskStatus.READY
    graph = TaskGraph()
    graph.add_node(task)
    state = RuntimeState(operation_id="op-scheduler-repetition", execution=OperationRuntime(operation_id="op-scheduler-repetition"))
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    state.register_task(
        TaskRuntime(
            task_id=task.id,
            tg_node_id=task.id,
            status=TaskRuntimeStatus.FAILED,
        )
    )

    result = SchedulerAgent().run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=AgentContext(operation_id="op-scheduler-repetition"),
            raw_payload={"tg_graph": graph.to_dict(), "runtime_state": state.model_dump(mode="json")},
        )
    )

    assert result.success is True
    assert not any(decision["accepted"] for decision in result.output.decisions)
    assert result.output.decisions[0]["action"] == "reject"
    assert any("repetition detector rejected 1 task" in log for log in result.output.logs)


def test_scheduler_runtime_probe_does_not_create_repetition_history() -> None:
    task = TaskBuilderAgentShim.create_task(_task(80))
    task.status = TaskStatus.READY
    graph = TaskGraph()
    graph.add_node(task)
    state = RuntimeState(operation_id="op-scheduler-runtime-probe", execution=OperationRuntime(operation_id="op-scheduler-runtime-probe"))
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    scheduler = SchedulerAgent()

    def mutating_probe(*, task_graph: TaskGraph, runtime_state: RuntimeState) -> list[str]:
        runtime_state.register_task(
            TaskRuntime(
                task_id=task.id,
                tg_node_id=task.id,
                status=TaskRuntimeStatus.BLOCKED,
            )
        )
        return []

    scheduler._runtime_scheduler.select_schedulable_tasks = mutating_probe  # noqa: SLF001

    result = scheduler.run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=AgentContext(operation_id="op-scheduler-runtime-probe"),
            raw_payload={"tg_graph": graph.to_dict(), "runtime_state": state.model_dump(mode="json")},
        )
    )

    assert result.success is True
    assert result.output.decisions[0]["action"] == "assign"
    assert result.output.decisions[0]["accepted"] is True
    assert not any("repetition detector rejected 1 task" in log for log in result.output.logs)


def test_scheduler_repetition_detector_skips_repeat_succeeded_task_before_assignment() -> None:
    task = TaskBuilderAgentShim.create_task(_task(80))
    task.status = TaskStatus.READY
    graph = TaskGraph()
    graph.add_node(task)
    state = RuntimeState(operation_id="op-scheduler-repetition", execution=OperationRuntime(operation_id="op-scheduler-repetition"))
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    state.register_task(
        TaskRuntime(
            task_id=task.id,
            tg_node_id=task.id,
            status=TaskRuntimeStatus.SUCCEEDED,
        )
    )

    result = SchedulerAgent().run(
        AgentInput(
            graph_refs=[GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph")],
            context=AgentContext(operation_id="op-scheduler-repetition"),
            raw_payload={"tg_graph": graph.to_dict(), "runtime_state": state.model_dump(mode="json")},
        )
    )

    assert result.success is True
    assert not any(decision["accepted"] for decision in result.output.decisions)
    assert result.output.decisions[0]["action"] == "skip"
    assert any("repetition detector skipped 1 satisfied task" in log for log in result.output.logs)
    patches = [delta["patch"] for delta in result.output.state_deltas]
    assert any(patch.get("status") == TaskStatus.SKIPPED.value for patch in patches)
    assert any(patch.get("status") == TaskRuntimeStatus.SKIPPED.value for patch in patches)


class TaskBuilderAgentShim:
    @staticmethod
    def create_task(candidate: TaskCandidate):
        from src.core.graph.tg_builder import TaskGraphBuilder

        return TaskGraphBuilder().create_task_node(candidate)
