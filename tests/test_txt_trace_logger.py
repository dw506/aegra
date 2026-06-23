from __future__ import annotations

from src.core.models.ag import GraphRef
from src.core.runtime.txt_trace_logger import TxtTraceLogger


def test_txt_trace_logger_serializes_pydantic_values(tmp_path) -> None:
    logger = TxtTraceLogger("op-trace", log_dir=tmp_path)

    logger.write_block(
        "PLANNER",
        "decision",
        {"target_refs": [GraphRef(graph="kg", ref_id="host-1", ref_type="Host")]},
    )

    text = tmp_path.joinpath("op-trace.run.txt").read_text(encoding="utf-8")

    assert '"ref_id": "host-1"' in text


def test_txt_trace_logger_redacts_sensitive_key_fragments(tmp_path) -> None:
    logger = TxtTraceLogger("op-redact", log_dir=tmp_path)

    logger.write_block(
        "TOOL_CALL",
        "redaction",
        {
            "access_token": "tok-1",
            "Authorization-Token": "bearer-1",
            "headers": {"cookie_value": "cookie-1", "private-key": "key-1"},
            "safe": "visible",
        },
    )

    text = tmp_path.joinpath("op-redact.run.txt").read_text(encoding="utf-8")

    assert "visible" in text
    assert "tok-1" not in text
    assert "bearer-1" not in text
    assert "cookie-1" not in text
    assert "key-1" not in text


def test_txt_trace_logger_operation_trace_path(tmp_path) -> None:
    logger = TxtTraceLogger.operation_trace("op-1", runtime_root=tmp_path)

    logger.write_header("Operation: op-1", {"cycle_index": 1})
    logger.write_block("LLM_DECISION", "execution agent decision", {"action": "finish"})

    text = tmp_path.joinpath("op-1", "operation-trace.txt").read_text(encoding="utf-8")

    assert "Operation: op-1" in text
    assert "cycle_index" in text
    assert "LLM_DECISION" in text
    assert "finish" in text
