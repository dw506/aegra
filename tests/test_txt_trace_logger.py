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
