from __future__ import annotations

from pathlib import Path

from src.core.perception.parser_registry import ParserRegistry


def test_core_perception_does_not_import_external_c2() -> None:
    paths = [
        path
        for path in Path("src/core/perception").glob("*.py")
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "src.integrations." not in text


def test_parser_registry_default_uses_generic_parser() -> None:
    registry = ParserRegistry.default()

    assert any(parser.name == "generic_parser" for parser in registry.parsers)
    assert any(parser.name == "tool_execution_parser" for parser in registry.parsers)
    assert registry.parsers[-1].name == "generic_parser"
    assert not any("c2" in parser.name.lower() for parser in registry.parsers)
