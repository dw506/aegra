"""Registry for controlled tool recipes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.core.tools.recipe import ToolRecipe, ToolRecipeAdapter, ToolTask


class ToolRecipeRegistry:
    """Small registry exposing the first supported tool templates."""

    supported_templates = frozenset({"http_probe", "nmap_service_scan"})

    def __init__(self, *, adapter: ToolRecipeAdapter | None = None) -> None:
        self._adapter = adapter or ToolRecipeAdapter()

    def build(self, task: ToolTask | Mapping[str, Any]) -> ToolRecipe:
        """Build a recipe by delegating through the controlled adapter."""

        return self._adapter.build(dict(task) if isinstance(task, Mapping) else task)

    def is_supported(self, tool_hint: str) -> bool:
        """Return whether a tool hint is registered."""

        return tool_hint in self.supported_templates


__all__ = ["ToolRecipeRegistry"]

