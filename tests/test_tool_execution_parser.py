from __future__ import annotations

from src.core.agents.agent_models import OutcomeRecord
from src.core.perception.parser_registry import ParserRegistry
from src.core.perception.tool_execution_parser import ToolExecutionParser


def _outcome(payload: dict | None = None) -> OutcomeRecord:
    return OutcomeRecord(
        source_agent="access_validation_worker",
        task_id="task-1",
        outcome_type="identity_context_confirmation",
        success=True,
        summary="access validation completed",
        payload=payload or {},
    )


def test_tool_execution_raw_result_is_parsed() -> None:
    parser = ToolExecutionParser()
    raw_result = {
        "tool_execution": {
            "adapter": "local_shell",
            "tool": "session_probe",
            "success": True,
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        }
    }

    assert parser.supports(raw_result, _outcome())

    parsed = parser.parse(raw_result, _outcome())

    assert parsed.metadata["parser"] == "tool_execution_parser"
    assert parsed.observations[0]["payload"]["category"] == "tool_execution"
    assert parsed.observations[0]["payload"]["adapter"] == "local_shell"
    assert parsed.observations[0]["payload"]["stdout_excerpt"] == "ok"
    assert parsed.evidence[0]["payload"]["kind"] == "tool_execution_evidence"


def test_tool_execution_outcome_payload_is_parsed() -> None:
    outcome = _outcome(
        {
            "tool_execution": {
                "adapter": "incalmo_c2",
                "tool": "session_probe",
                "success": True,
                "command_id": "cmd-1",
                "payload_ref": "incalmo://commands/cmd-1",
            }
        }
    )

    parsed = ToolExecutionParser().parse({}, outcome)

    assert parsed.observations[0]["payload"]["adapter"] == "incalmo_c2"
    assert parsed.observations[0]["payload"]["command_id"] == "cmd-1"
    assert parsed.evidence[0]["payload_ref"] == "incalmo://commands/cmd-1"


def test_tool_execution_failure_preserves_stderr_and_exit_code() -> None:
    parsed = ToolExecutionParser().parse(
        {
            "extra": {
                "tool_execution": {
                    "adapter": "local_shell",
                    "tool": "session_probe",
                    "success": False,
                    "exit_code": "policy_denied",
                    "stderr": "not allowed",
                }
            }
        },
        _outcome(),
    )

    payload = parsed.observations[0]["payload"]
    assert payload["success"] is False
    assert payload["exit_code"] == "policy_denied"
    assert payload["stderr_excerpt"] == "not allowed"


def test_tool_execution_parser_preserves_structured_mcp_parsed_payload() -> None:
    parsed = ToolExecutionParser().parse(
        {
            "tool_execution": {
                "adapter": "mcp_direct",
                "tool": "nmap_scan",
                "success": True,
                "exit_code": 0,
                "stdout": "80/tcp open http",
            },
            "parsed": {
                "entities": [{"type": "service", "port": 80, "service": "http"}],
                "relations": [{"type": "HOSTS", "source": "10.0.0.5", "target": "10.0.0.5:80"}],
                "findings": [{"kind": "open_service", "port": 80}],
                "runtime_hints": {"services": 1},
                "writeback_hints": {"observation_category": "service_discovery"},
            },
        },
        _outcome(),
    )

    payload = parsed.observations[0]["payload"]
    assert payload["parsed_entities"][0]["service"] == "http"
    assert payload["parsed_relations"][0]["type"] == "HOSTS"
    assert payload["runtime_hints"]["services"] == 1
    assert payload["writeback_hints"]["observation_category"] == "service_discovery"
    assert parsed.findings[0]["kind"] == "open_service"
    assert parsed.metadata["parsed"]["entities"][0]["port"] == 80


def test_tool_execution_parser_truncates_stdout_excerpt() -> None:
    parsed = ToolExecutionParser().parse(
        {
            "tool_execution": {
                "adapter": "local_shell",
                "tool": "session_probe",
                "success": True,
                "stdout": "x" * 600,
            }
        },
        _outcome(),
    )

    assert len(parsed.observations[0]["payload"]["stdout_excerpt"]) == 500


def test_parser_registry_default_prioritizes_tool_execution_before_generic() -> None:
    registry = ParserRegistry.default()

    assert [parser.name for parser in registry.parsers][-1] == "generic_parser"
    assert registry.parsers[0].name == "tool_execution_parser"

    parsed = registry.parse(
        {
            "tool_execution": {
                "adapter": "local_shell",
                "tool": "session_probe",
                "success": True,
            }
        },
        _outcome(),
    )

    assert parsed.metadata["parser"] == "tool_execution_parser"
