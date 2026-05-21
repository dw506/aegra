from __future__ import annotations

from pathlib import Path

from src.core.agents.agent_models import OutcomeRecord
from src.core.perception.parser_registry import ParserRegistry
from src.integrations.incalmo.perception import IncalmoC2Parser


def test_core_perception_does_not_import_incalmo() -> None:
    paths = [
        path
        for path in Path("src/core/perception").glob("*.py")
        if path.name != "c2_parser.py"
    ]
    paths.append(Path("src/core/agents/perception.py"))

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "integrations.incalmo" not in text
        assert "src.integrations.incalmo" not in text


def test_parser_registry_default_uses_generic_parser() -> None:
    registry = ParserRegistry.default()

    assert any(parser.name == "generic_parser" for parser in registry.parsers)
    assert any(parser.name == "tool_execution_parser" for parser in registry.parsers)
    assert registry.parsers[-1].name == "generic_parser"
    assert not any("incalmo" in parser.name for parser in registry.parsers)


def test_incalmo_parser_can_be_registered_externally() -> None:
    registry = ParserRegistry.default()
    registry.register(IncalmoC2Parser())

    assert any(parser.name == "incalmo_c2_parser" for parser in registry.parsers)

    parsed = registry.parse(
        {"adapter": "incalmo_c2", "summary": "command completed"},
        OutcomeRecord(
            source_agent="worker",
            task_id="task-1",
            outcome_type="execution_result",
            success=True,
            summary="worker completed",
        ),
    )

    assert parsed.metadata["parser"] == "incalmo_c2_parser"
