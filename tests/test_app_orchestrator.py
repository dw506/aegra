from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.api import FASTAPI_UNAVAILABLE_MESSAGE, app, create_app
from src.app.orchestrator import AppOrchestrator, TargetHost
from src.app.settings import AppSettings
from src.core.agents.agent_models import DecisionRecord
from src.core.agents.packy_llm import DEFAULT_PACKY_BASE_URL, DEFAULT_PACKY_MODEL, PackyLLMConfig
from src.core.agents.agent_pipeline import AgentPipeline
from src.core.agents.agent_protocol import (
    AgentContext,
    AgentInput,
    AgentKind,
    AgentOutput,
    BaseAgent,
    GraphRef,
    GraphScope,
    WritePermission,
)
from src.core.agents.critic import CriticAgent
from src.core.agents.pipeline_builders import AgentPipelineAssemblyOptions
from src.core.agents.scheduler_agent import SchedulerAgent
from src.core.agents.task_builder import TaskBuilderAgent
from src.core.graph.tg_builder import TaskCandidate
from src.core.models.ag import GraphRef as AGGraphRef
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime, WorkerRuntime, WorkerStatus
from src.core.models.runtime import LockStatus, ResourceLock, RuntimeEventRef, SessionLeaseRuntime, SessionRuntime, SessionStatus
from src.core.models.tg import TaskType
from src.core.workers.base import BaseWorkerAgent, WorkerTaskSpec


class FakePlannerAgent(BaseAgent):
    def __init__(self, *, emit_once: bool = False) -> None:
        super().__init__(
            name="fake_planner",
            kind=AgentKind.PLANNER,
            write_permission=WritePermission(scopes=[], allow_structural_write=False, allow_state_write=False, allow_event_emit=True),
        )
        self._emit_once = emit_once
        self._calls = 0

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        self._calls += 1
        if self._emit_once and self._calls > 1:
            return AgentOutput(logs=["planner emitted no more decisions"])
        candidate = TaskCandidate(
            source_action_id="action-service-1",
            task_type=TaskType.SERVICE_VALIDATION,
            input_bindings={"host_id": "host-1"},
            target_refs=[AGGraphRef(graph="kg", ref_id="host-1", ref_type="Host")],
            estimated_cost=0.1,
            estimated_risk=0.1,
            estimated_noise=0.1,
            goal_relevance=0.8,
            resource_keys={"host:host-1"},
            parallelizable=True,
        )
        decision = DecisionRecord(
            source_agent=self.name,
            summary="selected fake plan",
            confidence=0.9,
            refs=[],
            payload={
                "planning_candidate": {
                    "action_ids": ["action-service-1"],
                    "task_candidates": [candidate.model_dump(mode="json")],
                }
            },
            decision_type="plan_selection",
            score=0.9,
            target_refs=[GraphRef(graph=GraphScope.AG, ref_id="action-service-1", ref_type="ActionNode")],
            rationale="fake plan rationale",
        )
        return AgentOutput(decisions=[decision.to_agent_output_fragment()], logs=["planner emitted one decision"])


class FakeWorkerAgent(BaseWorkerAgent):
    def __init__(self) -> None:
        super().__init__(name="fake_worker")

    def supports_task(self, task_spec: WorkerTaskSpec) -> bool:
        return task_spec.task_type == TaskType.SERVICE_VALIDATION.value

    def execute_task(self, task_spec: WorkerTaskSpec, agent_input: AgentInput) -> AgentOutput:
        del agent_input
        raw_result = self.build_raw_result(
            task_id=task_spec.task_id,
            result_type="probe_result",
            summary="fake probe completed",
            payload_ref=f"runtime://worker-results/{task_spec.task_id}",
            refs=task_spec.target_refs,
            extra={"tool": "fake_probe"},
        )
        outcome = self.build_outcome(
            task_id=task_spec.task_id,
            outcome_type="execution_result",
            success=True,
            summary="fake probe completed",
            raw_result_ref=raw_result["payload_ref"],
            confidence=0.8,
            refs=task_spec.target_refs,
            payload={"status": "ok"},
        )
        return AgentOutput(
            outcomes=[outcome.to_agent_output_fragment()],
            evidence=[raw_result],
            logs=["worker executed fake probe"],
        )


def build_runtime_state() -> RuntimeState:
    runtime = RuntimeState(operation_id="op-loop", execution=OperationRuntime(operation_id="op-loop"))
    runtime.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.IDLE)
    return runtime


def build_graph_refs() -> list[GraphRef]:
    return [
        GraphRef(graph=GraphScope.KG, ref_id="kg-root", ref_type="graph"),
        GraphRef(graph=GraphScope.AG, ref_id="ag-root", ref_type="graph"),
        GraphRef(graph=GraphScope.TG, ref_id="tg-root", ref_type="graph"),
    ]


def test_settings_from_env(monkeypatch, tmp_path) -> None:
    policy_path = tmp_path / "runtime-policy.json"
    policy_path.write_text('{"cidr_whitelist": ["10.0.0.0/24"]}', encoding="utf-8")
    monkeypatch.setenv("AEGRA_RUNTIME_STORE_BACKEND", "file")
    monkeypatch.setenv("AEGRA_RUNTIME_STORE_DIR", str(tmp_path / "runtime-store"))
    monkeypatch.setenv("AEGRA_MAX_CONCURRENT_WORKERS", "8")
    monkeypatch.setenv("AEGRA_AUDIT_ENABLED", "false")
    monkeypatch.setenv("AEGRA_AUDIT_PERSIST_ENABLED", "true")
    monkeypatch.setenv("AEGRA_RECOVERY_ENABLED", "true")
    monkeypatch.setenv("AEGRA_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("AEGRA_RUNTIME_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("AEGRA_RUNTIME_POLICY_JSON", '{"blocked_hosts": ["host-9"]}')
    monkeypatch.setenv("AEGRA_LLM_API_KEY", "planner-key")
    monkeypatch.setenv("AEGRA_LLM_BASE_URL", "https://planner.example/v1")
    monkeypatch.setenv("AEGRA_LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("AEGRA_LLM_TIMEOUT_SEC", "45")
    monkeypatch.setenv("AEGRA_ENABLE_PLANNER_LLM_ADVISOR", "true")
    monkeypatch.setenv("AEGRA_ENABLE_CRITIC_LLM_ADVISOR", "true")
    monkeypatch.setenv("AEGRA_ENABLE_SUPERVISOR_LLM_ADVISOR", "true")

    settings = AppSettings.from_env()

    assert settings.runtime_store_backend == "file"
    assert settings.runtime_store_dir == (tmp_path / "runtime-store").resolve()
    assert settings.max_concurrent_workers == 8
    assert settings.audit_enabled is False
    assert settings.audit_persist_enabled is True
    assert settings.recovery_enabled is True
    assert settings.audit_dir == (tmp_path / "audit").resolve()
    assert settings.llm_api_key == "planner-key"
    assert settings.llm_base_url == "https://planner.example/v1"
    assert settings.llm_model == "gpt-5.4"
    assert settings.llm_timeout_sec == 45.0
    assert settings.enable_planner_llm_advisor is True
    assert settings.enable_critic_llm_advisor is True
    assert settings.enable_supervisor_llm_advisor is True
    policy = settings.load_runtime_policy()

    assert policy.model_dump(mode="json") == {
        "blocked_hosts": ["host-9"],
        "cidr_whitelist": ["10.0.0.0/24"],
        "authorized_hosts": [],
        "default_task_timeout_sec": 900,
        "deny_egress": False,
        "loaded_at": policy.model_dump(mode="json")["loaded_at"],
        "loaded_from": str(policy_path.resolve()),
        "max_concurrent_per_host": {},
        "policy_version": "v1",
        "rate_limit_per_subnet_per_min": {},
        "retry_backoff_base_sec": 0,
        "safety_stop": False,
        "sensitive_tags": [],
        "sensitive_task_types": [],
        "session_policies": {},
    }


def test_settings_to_packy_llm_config_uses_settings_values() -> None:
    settings = AppSettings(
        llm_api_key="planner-key",
        llm_base_url="https://planner.example/v1",
        llm_model="gpt-5.4",
        llm_timeout_sec=45.0,
    )

    config = settings.to_packy_llm_config()

    assert config == PackyLLMConfig(
        api_key="planner-key",
        base_url="https://planner.example/v1",
        model="gpt-5.4",
        timeout_sec=45.0,
    )


def test_settings_to_packy_llm_config_returns_none_without_key() -> None:
    settings = AppSettings()

    assert settings.to_packy_llm_config() is None


def test_settings_to_packy_llm_config_uses_packy_defaults_when_optional_fields_missing() -> None:
    settings = AppSettings(llm_api_key="planner-key")

    config = settings.to_packy_llm_config()

    assert config == PackyLLMConfig(
        api_key="planner-key",
        base_url=DEFAULT_PACKY_BASE_URL,
        model=DEFAULT_PACKY_MODEL,
        timeout_sec=30.0,
    )


def test_orchestrator_create_import_and_start_operation(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        runtime_policy={"authorized_hosts": ["10.0.0.10"]},
    )
    orchestrator = AppOrchestrator(settings=settings)

    created = orchestrator.create_operation("op-1", metadata={"engagement": "lab"})
    imported = orchestrator.import_targets(
        "op-1",
        [
            TargetHost(address="10.0.0.20", hostname="dc-1"),
            TargetHost(address="10.0.0.10", hostname="jump-1"),
            TargetHost(address="10.0.0.20", hostname="dc-1-duplicate"),
        ],
    )
    started = orchestrator.start_operation("op-1")
    summary = orchestrator.get_operation_summary("op-1")

    assert created.execution.metadata["engagement"] == "lab"
    assert created.execution.metadata["runtime_policy"]["authorized_hosts"] == ["10.0.0.10"]
    assert created.execution.metadata["runtime_policy"]["policy_version"] == "v1"
    assert created.execution.metadata["runtime_policy"]["loaded_from"] == "settings"
    assert created.execution.metadata["control_plane"]["llm_advisors"] == {
        "planner_enabled": False,
        "critic_enabled": False,
        "supervisor_enabled": False,
        "configured": False,
        "model": None,
        "base_url": None,
    }
    assert imported.execution.metadata["target_count"] == 2
    assert [item["address"] for item in imported.execution.metadata["target_inventory"]] == [
        "10.0.0.10",
        "10.0.0.20",
    ]
    assert started.operation_status.value == "ready"
    assert summary.target_count == 2
    assert summary.metadata["last_control_cycle"]["cycle_type"] == "bootstrap"
    assert summary.last_cycle_phase is None
    assert summary.unclean_shutdown is False
    assert summary.audit_event_count == 0
    assert summary.pending_event_count == 0
    assert created.execution.metadata["operation_log"][0]["event_type"] == "operation_created"
    assert started.execution.metadata["operation_log"][-1]["event_type"] == "operation_started"
    assert len(orchestrator.list_operations()) == 1


def test_orchestrator_builds_default_pipeline_without_planner_llm_when_no_key(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_optional_agent_pipeline(
        *,
        options=None,
        planner_llm_advisor=None,
        critic_llm_advisor=None,
        supervisor_llm_advisor=None,
        llm_client_config=None,
        event_sink=None,
        state_delta_sink=None,
    ):
        captured["options"] = options
        captured["planner_llm_advisor"] = planner_llm_advisor
        captured["llm_client_config"] = llm_client_config
        captured["event_sink"] = event_sink
        captured["state_delta_sink"] = state_delta_sink
        return AgentPipeline()

    monkeypatch.delenv("AEGRA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.app.orchestrator.build_optional_agent_pipeline", fake_build_optional_agent_pipeline)

    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)

    options = captured["options"]
    assert isinstance(options, AgentPipelineAssemblyOptions)
    assert options.enable_packy_planner_advisor is False
    assert options.enable_packy_critic_advisor is False
    assert isinstance(orchestrator.pipeline, AgentPipeline)


def test_orchestrator_builds_default_pipeline_with_planner_llm_when_key_exists(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_optional_agent_pipeline(
        *,
        options=None,
        planner_llm_advisor=None,
        critic_llm_advisor=None,
        supervisor_llm_advisor=None,
        llm_client_config=None,
        event_sink=None,
        state_delta_sink=None,
    ):
        captured["options"] = options
        captured["planner_llm_advisor"] = planner_llm_advisor
        captured["llm_client_config"] = llm_client_config
        captured["event_sink"] = event_sink
        captured["state_delta_sink"] = state_delta_sink
        return AgentPipeline()

    monkeypatch.setattr("src.app.orchestrator.build_optional_agent_pipeline", fake_build_optional_agent_pipeline)

    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        llm_api_key="test-llm-key",
        llm_model="gpt-5.4",
        llm_timeout_sec=45.0,
        enable_planner_llm_advisor=True,
        enable_critic_llm_advisor=True,
        enable_supervisor_llm_advisor=True,
    )
    orchestrator = AppOrchestrator(settings=settings)

    options = captured["options"]
    assert isinstance(options, AgentPipelineAssemblyOptions)
    assert options.enable_packy_planner_advisor is True
    assert options.enable_packy_critic_advisor is True
    assert options.enable_packy_supervisor_advisor is True
    assert captured["planner_llm_advisor"] is None
    assert captured["event_sink"] is None
    assert captured["state_delta_sink"] is None
    assert captured["llm_client_config"] == PackyLLMConfig(
        api_key="test-llm-key",
        base_url=DEFAULT_PACKY_BASE_URL,
        model="gpt-5.4",
        timeout_sec=45.0,
    )
    assert isinstance(orchestrator.pipeline, AgentPipeline)


def test_orchestrator_requires_llm_key_when_planner_llm_is_enabled(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        enable_planner_llm_advisor=True,
    )

    try:
        AppOrchestrator(settings=settings)
    except ValueError as exc:
        assert "enable_planner_llm_advisor require llm_api_key" in str(exc)
    else:
        raise AssertionError("AppOrchestrator should reject planner LLM enablement without a key")


def test_orchestrator_requires_llm_key_when_critic_llm_is_enabled(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        enable_critic_llm_advisor=True,
    )

    try:
        AppOrchestrator(settings=settings)
    except ValueError as exc:
        assert "enable_critic_llm_advisor require llm_api_key" in str(exc)
    else:
        raise AssertionError("AppOrchestrator should reject critic LLM enablement without a key")


def test_orchestrator_requires_llm_key_when_supervisor_llm_is_enabled(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        enable_supervisor_llm_advisor=True,
    )

    try:
        AppOrchestrator(settings=settings)
    except ValueError as exc:
        assert "enable_supervisor_llm_advisor require llm_api_key" in str(exc)
    else:
        raise AssertionError("AppOrchestrator should reject supervisor LLM enablement without a key")


def test_settings_load_runtime_policy_raises_clear_error_for_invalid_policy_file(tmp_path) -> None:
    policy_path = tmp_path / "runtime-policy.json"
    policy_path.write_text('{"max_concurrent_per_host": {"default": 0}}', encoding="utf-8")
    settings = AppSettings(runtime_policy_path=policy_path)

    try:
        settings.load_runtime_policy()
    except ValueError as exc:
        assert "invalid runtime policy" in str(exc)
        assert str(policy_path.resolve()) in str(exc)
    else:
        raise AssertionError("invalid runtime policy should raise a clear error")


def test_create_app_raises_when_fastapi_is_unavailable() -> None:
    try:
        create_app()
    except RuntimeError as exc:
        assert str(exc) == FASTAPI_UNAVAILABLE_MESSAGE
    else:  # pragma: no cover - only hit when FastAPI is installed locally
        assert True


def test_global_app_is_none_when_fastapi_is_unavailable() -> None:
    assert app is None


def test_orchestrator_run_operation_cycle_executes_plan_schedule_apply_feedback(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    pipeline = AgentPipeline(
        agents=[FakePlannerAgent(), TaskBuilderAgent(), SchedulerAgent(), FakeWorkerAgent(), CriticAgent()]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-loop")
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"sensitive_task_types": []}
    orchestrator.runtime_store.save_state(runtime_state)

    result = orchestrator.run_operation_cycle(
        "op-loop",
        graph_refs=build_graph_refs(),
        planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
    )

    assert result.planning is not None and result.planning.success is True
    assert result.execution is not None and result.execution.success is True
    assert result.feedback is not None and result.feedback.success is True
    assert result.selected_task_ids
    assert result.applied_task_ids == result.selected_task_ids
    assert len(result.apply_results) == 1
    state = orchestrator.get_operation_state("op-loop")
    assert state.execution.metadata["last_control_cycle"]["cycle_index"] == 1
    assert state.execution.metadata["control_cycle_history"][0]["applied_results"][0]["task_id"] == result.applied_task_ids[0]
    assert any(entry["event_type"] == "tool_invocation" for entry in state.execution.metadata["audit_log"])
    assert [item["phase"] for item in state.execution.metadata["phase_checkpoints"]] == [
        "cycle_started",
        "planning_completed",
        "execution_completed",
        "apply_completed",
        "feedback_completed",
        "cycle_completed",
    ]
    assert state.execution.metadata["last_phase_checkpoint"]["phase"] == "cycle_completed"
    assert state.execution.metadata["recovery"]["last_phase"] == "cycle_completed"
    assert state.execution.metadata["recovery"]["last_phase_status"] == "completed"


def test_orchestrator_run_until_quiescent_stops_when_no_more_tasks(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    pipeline = AgentPipeline(
        agents=[FakePlannerAgent(emit_once=True), TaskBuilderAgent(), SchedulerAgent(), FakeWorkerAgent(), CriticAgent()]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-loop")
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"sensitive_task_types": []}
    orchestrator.runtime_store.save_state(runtime_state)

    results = orchestrator.run_until_quiescent(
        "op-loop",
        graph_refs=build_graph_refs(),
        planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
        max_cycles=3,
    )

    assert len(results) == 2
    assert results[-1].stopped is True
    assert results[-1].stop_reason == "no schedulable work and no replan request"


def test_orchestrator_resume_operation_requeues_inflight_work(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.create_operation("op-recover")
    state = orchestrator.get_operation_state("op-recover")
    state.operation_status = state.execution.status = "running"
    state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status="running",
            assigned_worker="worker-1",
        )
    )
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.BUSY, current_task_id="task-1")
    state.execution.metadata["recovery"] = {"unclean_shutdown": True, "last_replayed_cursor": 2}
    orchestrator.runtime_store.save_state(state)

    resumed = orchestrator.resume_operation("op-recover", reason="test_resume")

    assert resumed.operation_status.value == "ready"
    assert resumed.execution.tasks["task-1"].status.value == "pending"
    assert resumed.execution.tasks["task-1"].assigned_worker is None
    assert resumed.workers["worker-1"].status == WorkerStatus.IDLE
    assert resumed.execution.metadata["operation_log"][-1]["event_type"] == "operation_resumed"


def test_orchestrator_resume_operation_cleans_runtime_recovery_artifacts(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.create_operation("op-recover")
    state = orchestrator.get_operation_state("op-recover")
    state.operation_status = state.execution.status = "running"
    state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status="claimed",
            assigned_worker="worker-1",
        )
    )
    state.workers["worker-1"] = WorkerRuntime(
        worker_id="worker-1",
        status=WorkerStatus.BUSY,
        current_task_id="task-1",
        current_load=1,
    )
    state.sessions["session-1"] = SessionRuntime(
        session_id="session-1",
        status=SessionStatus.ACTIVE,
        metadata={"bound_task_ids": ["task-1"], "lease_ids": ["lease-1"]},
    )
    state.session_leases["lease-1"] = SessionLeaseRuntime(
        lease_id="lease-1",
        session_id="session-1",
        owner_task_id="task-1",
        owner_worker_id="worker-1",
    )
    state.locks["host:host-1"] = ResourceLock(
        lock_key="host:host-1",
        owner_type="task",
        owner_id="task-1",
        status=LockStatus.ACTIVE,
    )
    state.pending_events.append(
        RuntimeEventRef(
            event_id="evt-1",
            event_type="lock.acquired",
            cursor=3,
        )
    )
    state.execution.metadata["recovery"] = {"unclean_shutdown": True, "last_replayed_cursor": 2}
    orchestrator.runtime_store.save_state(state)

    resumed = orchestrator.resume_operation("op-recover", reason="test_resume")
    recovery_snapshot = orchestrator.runtime_store.export_recovery_snapshot("op-recover")

    assert resumed.execution.tasks["task-1"].status.value == "pending"
    assert resumed.workers["worker-1"].status == WorkerStatus.IDLE
    assert resumed.sessions["session-1"].status == SessionStatus.EXPIRED
    assert resumed.session_leases["lease-1"].metadata["release_reason"] == "test_resume"
    assert resumed.locks["host:host-1"].status == LockStatus.RELEASED
    assert resumed.pending_events[0].metadata["recovery"]["requires_replay"] is True
    assert resumed.pending_events[0].metadata["replay"]["replay_status"] == "planned"
    assert resumed.execution.metadata["recovery"]["last_resume_reason"] == "test_resume"
    assert resumed.execution.metadata["recovery"]["last_replayed_cursor"] == 2
    assert resumed.execution.metadata["recovery"]["replay_status"] == "planned"
    assert resumed.execution.metadata["recovery"]["replay_candidate_event_ids"] == ["evt-1"]
    assert resumed.execution.metadata["recovery"]["released_lock_ids"] == ["host:host-1"]
    assert resumed.execution.metadata["recovery"]["expired_session_ids"] == ["session-1"]
    assert recovery_snapshot["recovery_metadata"]["recovered_event_count"] == 1
    assert recovery_snapshot["replay_plan"]["start_cursor"] == 3
    assert recovery_snapshot["replay_plan"]["last_replayed_cursor"] == 2
    assert recovery_snapshot["replay_plan"]["replay_candidate_event_ids"] == ["evt-1"]


def test_orchestrator_recover_operation_and_export_audit(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    orchestrator = AppOrchestrator(settings=settings)
    created = orchestrator.create_operation("op-audit")
    state = orchestrator.get_operation_state("op-audit")
    state.operation_status = state.execution.status = "running"
    state.execution.metadata["audit_log"] = [{"audit_id": "audit-1", "event_type": "fact_write"}]
    state.execution.metadata["recovery"] = {"unclean_shutdown": True}
    state.register_task(
        TaskRuntime(
            task_id="task-1",
            tg_node_id="task-1",
            status="running",
            assigned_worker="worker-1",
        )
    )
    state.workers["worker-1"] = WorkerRuntime(worker_id="worker-1", status=WorkerStatus.BUSY, current_task_id="task-1")
    orchestrator.runtime_store.save_state(state)

    recovered = orchestrator.recover_operation("op-audit", reason="ops_recover")
    summary = orchestrator.get_operation_summary("op-audit")
    audit_report = orchestrator.export_audit_report("op-audit")

    assert created.operation_id == "op-audit"
    assert recovered.execution.tasks["task-1"].status.value == "pending"
    assert recovered.operation_status.value == "running"
    assert summary.unclean_shutdown is False
    assert summary.audit_event_count == 1
    assert summary.last_cycle_phase is None
    assert audit_report["audit_log"][0]["event_type"] == "fact_write"
    assert audit_report["operation_log"][-1]["event_type"] == "operation_recovered"


def test_orchestrator_health_and_readiness_status(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        llm_api_key="planner-key",
        llm_model="gpt-5.4",
        enable_planner_llm_advisor=True,
    )
    orchestrator = AppOrchestrator(settings=settings)
    orchestrator.create_operation("op-health")

    health = orchestrator.get_health_status()
    readiness = orchestrator.get_readiness_status()

    assert health == {
        "status": "ok",
        "runtime_store_backend": "file",
        "operation_count": 1,
        "llm_advisors": {
            "planner_enabled": True,
            "critic_enabled": False,
            "supervisor_enabled": False,
            "configured": True,
            "model": "gpt-5.4",
            "base_url": DEFAULT_PACKY_BASE_URL,
        },
    }
    assert readiness == {
        "status": "ready",
        "runtime_store_backend": "file",
        "recovery_enabled": True,
        "llm_advisors": {
            "planner_enabled": True,
            "critic_enabled": False,
            "supervisor_enabled": False,
            "configured": True,
            "model": "gpt-5.4",
            "base_url": DEFAULT_PACKY_BASE_URL,
        },
    }


def test_orchestrator_persists_phase_checkpoints_to_file_store(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    pipeline = AgentPipeline(
        agents=[FakePlannerAgent(), TaskBuilderAgent(), SchedulerAgent(), FakeWorkerAgent(), CriticAgent()]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-phase")
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"sensitive_task_types": []}
    orchestrator.runtime_store.save_state(runtime_state)

    orchestrator.run_operation_cycle(
        "op-phase",
        graph_refs=build_graph_refs(),
        planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
    )

    recovery_snapshot = orchestrator.runtime_store.export_recovery_snapshot("op-phase")
    state_snapshot = orchestrator.runtime_store.export_state_snapshot("op-phase")

    assert recovery_snapshot["last_phase_checkpoint"]["phase"] == "cycle_completed"
    assert recovery_snapshot["last_phase_checkpoint"]["status"] == "completed"
    assert recovery_snapshot["recovery_metadata"]["phase_checkpoint_count"] == 6
    assert state_snapshot["execution"]["metadata"]["phase_checkpoints"][3]["phase"] == "apply_completed"
    assert "runtime_event_count" in state_snapshot["execution"]["metadata"]["phase_checkpoints"][3]


def test_orchestrator_cycle_summary_records_llm_advisor_status(tmp_path) -> None:
    settings = AppSettings(
        runtime_store_backend="file",
        runtime_store_dir=tmp_path / "runtime-store",
        llm_api_key="planner-key",
        llm_model="gpt-5.4",
        enable_planner_llm_advisor=True,
        enable_critic_llm_advisor=True,
    )
    pipeline = AgentPipeline(
        agents=[FakePlannerAgent(), TaskBuilderAgent(), SchedulerAgent(), FakeWorkerAgent(), CriticAgent()]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-llm-summary")
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"sensitive_task_types": []}
    orchestrator.runtime_store.save_state(runtime_state)

    result = orchestrator.run_operation_cycle(
        "op-llm-summary",
        graph_refs=build_graph_refs(),
        planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
    )

    assert result.runtime_state.execution.metadata["last_control_cycle"]["llm_advisors"] == {
        "planner_enabled": True,
        "critic_enabled": True,
        "supervisor_enabled": False,
        "configured": True,
        "model": "gpt-5.4",
        "base_url": DEFAULT_PACKY_BASE_URL,
    }


def test_orchestrator_persists_apply_checkpoint_before_feedback_crash(tmp_path) -> None:
    settings = AppSettings(runtime_store_backend="file", runtime_store_dir=tmp_path / "runtime-store")
    pipeline = AgentPipeline(
        agents=[FakePlannerAgent(), TaskBuilderAgent(), SchedulerAgent(), FakeWorkerAgent(), CriticAgent()]
    )
    orchestrator = AppOrchestrator(settings=settings, pipeline=pipeline)
    orchestrator.create_operation("op-crash")
    runtime_state = build_runtime_state()
    runtime_state.execution.metadata["runtime_policy"] = {"sensitive_task_types": []}
    orchestrator.runtime_store.save_state(runtime_state)

    def fail_feedback(**_: object) -> object:
        raise RuntimeError("feedback crashed")

    orchestrator._run_feedback_phase = fail_feedback  # type: ignore[method-assign]

    try:
        orchestrator.run_operation_cycle(
            "op-crash",
            graph_refs=build_graph_refs(),
            planner_payload={"goal_refs": [], "planning_context": {"top_k": 1, "max_depth": 1}},
        )
    except RuntimeError as exc:
        assert str(exc) == "feedback crashed"
    else:
        raise AssertionError("run_operation_cycle should propagate feedback failure")

    persisted = orchestrator.runtime_store.export_state_snapshot("op-crash")
    recovery_snapshot = orchestrator.runtime_store.export_recovery_snapshot("op-crash")
    phases = [item["phase"] for item in persisted["execution"]["metadata"]["phase_checkpoints"]]

    assert phases == [
        "cycle_started",
        "planning_completed",
        "execution_completed",
        "apply_completed",
    ]
    assert persisted["execution"]["metadata"]["last_phase_checkpoint"]["phase"] == "apply_completed"
    assert recovery_snapshot["last_phase_checkpoint"]["phase"] == "apply_completed"
    assert recovery_snapshot["recovery_metadata"]["last_phase"] == "apply_completed"
    assert recovery_snapshot["recovery_metadata"]["unclean_shutdown"] is True
