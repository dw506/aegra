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
