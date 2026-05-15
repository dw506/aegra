from __future__ import annotations

from src.core.feedback.evidence_extractor import EvidenceExtractor
from src.core.feedback.result_verifier import ResultVerifier
from src.core.tools.recipe import ToolRecipeAdapter, ToolRecipeError
from src.core.tools.runner import RecipeRunResult


def test_tool_recipe_adapter_builds_http_function_without_shell_command() -> None:
    recipe = ToolRecipeAdapter().build(
        {
            "task_type": "SERVICE_VALIDATION",
            "tool_hint": "http_probe",
            "target": "http://127.0.0.1:8080/",
        }
    )

    assert recipe.execution_kind == "python_function"
    assert recipe.command is None
    assert recipe.function is not None
    assert recipe.metadata["host"] == "127.0.0.1"
    assert recipe.metadata["port"] == 8080


def test_tool_recipe_adapter_builds_safe_nmap_service_scan_command() -> None:
    recipe = ToolRecipeAdapter(nmap_path="nmap").build(
        {
            "task_type": "SERVICE_VALIDATION",
            "tool_hint": "nmap_service_scan",
            "target": "127.0.0.1:8080",
        }
    )

    assert recipe.execution_kind == "command"
    assert recipe.command == ["nmap", "-n", "-Pn", "-sV", "-p", "8080", "127.0.0.1"]
    assert recipe.command_allowlist == {"nmap"}


def test_tool_recipe_adapter_rejects_unsafe_target_text() -> None:
    try:
        ToolRecipeAdapter().build(
            {
                "task_type": "SERVICE_VALIDATION",
                "tool_hint": "nmap_service_scan",
                "target": "127.0.0.1;whoami:8080",
            }
        )
    except ToolRecipeError as exc:
        assert "unsafe" in str(exc)
    else:  # pragma: no cover - documents the failure mode.
        raise AssertionError("unsafe target was accepted")


def test_result_verifier_and_evidence_extractor_normalize_probe_output() -> None:
    result = RecipeRunResult(
        recipe_name="http_probe",
        target="http://127.0.0.1:8080/",
        success=True,
        category="success",
        output={
            "target": "http://127.0.0.1:8080/",
            "host": "127.0.0.1",
            "port": 8080,
            "reachable": True,
            "success": True,
            "http_status": 200,
            "banner": "Werkzeug",
            "title": "Vuln App",
            "entities": [{"id": "127.0.0.1:8080/tcp", "type": "Service", "port": 8080, "service_name": "http"}],
        },
    )

    verified = ResultVerifier().verify(result, expected_target="http://127.0.0.1:8080/")
    evidence = EvidenceExtractor().extract(result)

    assert verified == {
        "valid": True,
        "task_status": "succeeded",
        "new_information_found": True,
        "target_matched": True,
        "retry_needed": False,
        "reason": None,
    }
    assert {
        "type": "service_detected",
        "host": "127.0.0.1",
        "port": 8080,
        "service_name": "http",
        "confidence": 0.9,
    } in evidence
    assert any(item["type"] == "http_status" and item["status"] == 200 for item in evidence)
