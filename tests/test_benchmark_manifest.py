from __future__ import annotations

import json
from pathlib import Path

from benchmarks.evaluate import validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_manifest_schema_validates_seed_manifests() -> None:
    for path in (
        "benchmarks/vulnhub_local.json",
        "benchmarks/custom_vm.json",
        "benchmarks/autopentester_htb10.json",
    ):
        assert validate_manifest(load_manifest(path)) == []


def test_manifest_requires_explicit_scope_tools_and_risk() -> None:
    manifest = load_manifest("benchmarks/vulnhub_local.json")
    target = manifest["targets"][0]
    del target["allowed_tools"]
    target["risk_level"] = "unknown"
    target["scope"].pop("authorized_hosts")

    errors = validate_manifest(manifest)

    assert any("allowed_tools" in error for error in errors)
    assert any("risk_level" in error for error in errors)
    assert any("authorized_hosts" in error for error in errors)
