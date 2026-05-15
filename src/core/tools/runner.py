"""Runner for controlled tool recipes."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.tools.recipe import ToolRecipe, ToolRecipeError
from src.core.workers.tool_runner import ToolExecutionResult, ToolRunner


class RecipeRunResult(BaseModel):
    """Normalized result from a recipe run."""

    model_config = ConfigDict(extra="forbid")

    recipe_name: str
    target: str
    success: bool
    category: str
    output: dict[str, Any] = Field(default_factory=dict)
    tool: dict[str, Any] = Field(default_factory=dict)
    duration_sec: float = Field(default=0.0, ge=0.0)
    error_message: str | None = None


class ToolRecipeRunner:
    """Execute controlled recipes without accepting free-form commands."""

    def __init__(self, *, tool_runner: ToolRunner | None = None) -> None:
        self._tool_runner = tool_runner or ToolRunner()

    def run(self, recipe: ToolRecipe) -> RecipeRunResult:
        """Run one recipe and return a normalized payload."""

        started = perf_counter()
        if recipe.execution_kind == "python_function":
            if recipe.function is None:
                raise ToolRecipeError("python_function recipe requires a function")
            try:
                output = recipe.function()
                return RecipeRunResult(
                    recipe_name=recipe.recipe_name,
                    target=recipe.target,
                    success=bool(output.get("success")),
                    category="success" if output.get("success") else "failed",
                    output=output,
                    duration_sec=perf_counter() - started,
                    error_message=output.get("failure_reason"),
                )
            except Exception as exc:  # pragma: no cover - defensive boundary.
                return RecipeRunResult(
                    recipe_name=recipe.recipe_name,
                    target=recipe.target,
                    success=False,
                    category="process_error",
                    duration_sec=perf_counter() - started,
                    error_message=str(exc),
                )

        tool_result = self._tool_runner.run(recipe.to_execution_spec())
        return RecipeRunResult(
            recipe_name=recipe.recipe_name,
            target=recipe.target,
            success=tool_result.success,
            category=tool_result.category,
            output={
                "summary": f"{recipe.recipe_name} {tool_result.category} for {recipe.target}",
                "raw_output": tool_result.stdout,
                "stderr": tool_result.stderr,
                **recipe.metadata,
            },
            tool=tool_result.to_payload(),
            duration_sec=tool_result.duration_sec,
            error_message=tool_result.error_message,
        )


__all__ = ["RecipeRunResult", "ToolRecipeRunner"]

