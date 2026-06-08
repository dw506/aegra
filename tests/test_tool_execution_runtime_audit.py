from __future__ import annotations

from src.core.models.events import AgentResultStatus, AgentRole, AgentTaskResult, EvidenceArtifact
from src.core.models.runtime import OperationRuntime, RuntimeState, TaskRuntime
from src.core.runtime.result_applier import PhaseTwoResultApplier


def _state() -> RuntimeState:
    state = RuntimeState(operation_id="op-tool-audit", execution=OperationRuntime(operation_id="op-tool-audit"))
    state.register_task(TaskRuntime(task_id="task-1", execution_node_id="task-1"))
    return state


def _result(**kwargs) -> AgentTaskResult:
    payload = {
        "request_id": "request-1",
        "agent_role": AgentRole.ACCESS_WORKER,
        "operation_id": "op-tool-audit",
        "task_id": "task-1",
        "execution_node_id": "task-1",
        "status": AgentResultStatus.SUCCEEDED,
        "summary": "access validation completed",
    }
    payload.update(kwargs)
    return AgentTaskResult(**payload)


def _audit_events(state: RuntimeState, event_type: str) -> list[dict]:
    return [entry for entry in state.execution.metadata.get("audit_log", []) if entry["event_type"] == event_type]


def test_tool_execution_success_enters_audit_log() -> None:
    state = _state()
    result = _result(
        outcome_payload={
            "tool_execution": {
                "adapter": "local_shell",
                "tool": "session_probe",
                "success": True,
                "exit_code": 0,
                "stdout": "ok",
                "stderr": "",
                "payload_ref": "runtime://tool/session-probe",
            }
        }
    )

    PhaseTwoResultApplier().apply(result, state)

    entries = _audit_events(state, "tool_execution_recorded")
    assert len(entries) == 1
    assert entries[0]["adapter"] == "local_shell"
    assert entries[0]["tool"] == "session_probe"
    assert entries[0]["success"] is True
    assert entries[0]["exit_code"] == 0
    assert entries[0]["stdout_excerpt"] == "ok"
    assert entries[0]["payload_ref"] == "runtime://tool/session-probe"


def test_tool_execution_failure_preserves_stderr_excerpt_and_exit_code() -> None:
    state = _state()
    result = _result(
        status=AgentResultStatus.FAILED,
        summary="session probe failed",
        metadata={
            "tool_execution": {
                "adapter": "local_shell",
                "tool": "session_probe",
                "success": False,
                "exit_code": 126,
                "stderr": "permission denied" * 80,
            }
        },
    )

    PhaseTwoResultApplier().apply(result, state)

    entries = _audit_events(state, "tool_execution_failed")
    assert len(entries) == 1
    assert entries[0]["success"] is False
    assert entries[0]["exit_code"] == 126
    assert entries[0]["stderr_excerpt"].startswith("permission denied")
    assert "(truncated" in entries[0]["stderr_excerpt"]


def test_missing_tool_execution_does_not_add_tool_execution_audit() -> None:
    state = _state()

    PhaseTwoResultApplier().apply(_result(), state)

    event_types = [entry["event_type"] for entry in state.execution.metadata.get("audit_log", [])]
    assert "tool_execution_recorded" not in event_types
    assert "tool_execution_failed" not in event_types


def test_tool_execution_audit_is_adapter_name_agnostic_and_reads_evidence_metadata() -> None:
    state = _state()
    result = _result(
        evidence=[
            EvidenceArtifact(
                kind="tool_execution_evidence",
                summary="custom adapter output",
                payload_ref="adapter://commands/cmd-42",
                metadata={
                    "tool_execution": {
                        "adapter": "external_c2_alpha",
                        "tool": "operator_task",
                        "success": True,
                        "command_id": "cmd-42",
                    }
                },
            )
        ]
    )

    PhaseTwoResultApplier().apply(result, state)

    entries = _audit_events(state, "tool_execution_recorded")
    assert len(entries) == 1
    assert entries[0]["adapter"] == "external_c2_alpha"
    assert entries[0]["tool"] == "operator_task"
    assert entries[0]["command_id"] == "cmd-42"

