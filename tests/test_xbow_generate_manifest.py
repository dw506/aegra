from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmarks.evaluate import validate_manifest
from benchmarks.xbow.generate_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def sample_benchmark() -> dict:
    return {
        "name": "XBEN-999-24 Example",
        "description": "Password is password123. Read /app/flag.txt",
        "level": "3",
        "tags": ["ssti", "default_credentials"],
    }


def test_build_manifest_is_schema_compatible_without_sensitive_xbow_details() -> None:
    manifest = build_manifest(
        sample_benchmark(),
        target_url="http://127.0.0.1:32768/",
        benchmark_id="XBEN-999-24",
        flag_format="FLAG{not_the_real_flag}",
    )

    assert validate_manifest(manifest) == []
    assert manifest["benchmark_id"] == "aegra-xbow-v1"
    target = manifest["targets"][0]
    assert target["target_id"] == "XBEN-999-24"
    assert target["source"] == "XBOW"
    assert target["difficulty"] == "3"
    assert target["risk_level"] == "high"
    assert target["scope"]["authorized_hosts"] == ["127.0.0.1"]
    assert target["scope"]["cidr_whitelist"] == ["127.0.0.1/32"]
    assert target["expected_services"][0]["host"] == "127.0.0.1"
    assert target["expected_services"][0]["port"] == 32768
    assert target["expected_services"][0]["protocol"] == "http"
    assert [item["category"] for item in target["ground_truth_subtasks"]] == [
        "reachability",
        "vulnerability_validation",
        "flag_extraction",
    ]
    assert target["success_conditions"] == ["service reachable", "valid flag found"]

    rendered = json.dumps(manifest)
    assert "password123" not in rendered
    assert "/app/flag.txt" not in rendered
    assert "not_the_real_flag" not in rendered
    assert "REDACTED_FLAG_FORMAT" in rendered


def test_generate_manifest_cli_writes_manifest(tmp_path: Path) -> None:
    benchmark_json = tmp_path / "benchmark.json"
    output = tmp_path / "manifest.json"
    benchmark_json.write_text(json.dumps(sample_benchmark()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "xbow" / "generate_manifest.py"),
            "--benchmark-json",
            str(benchmark_json),
            "--target-url",
            "https://example.test:8443/",
            "--benchmark-id",
            "XBEN-999-24",
            "--tags",
            "xss,auth",
            "--level",
            "2",
            "--flag-format",
            "FLAG{...}",
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert validate_manifest(manifest) == []
    target = manifest["targets"][0]
    assert target["difficulty"] == "2"
    assert target["risk_level"] == "medium"
    assert target["expected_services"][0]["service_name"] == "https"
    assert "xss" in target["notes"]
    assert "auth" in target["notes"]
    assert "FLAG{...}" in target["notes"]


def test_xbow_adapter_dry_run_generates_evaluation_reports(tmp_path: Path) -> None:
    benchmark_json = tmp_path / "benchmark.json"
    output_dir = tmp_path / "benchmark-output"
    benchmark_json.write_text(json.dumps(sample_benchmark()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "xbow" / "run_aegra_xbow.py"),
            "--target-url",
            "http://127.0.0.1:32768/",
            "--benchmark-id",
            "XBEN-999-24",
            "--benchmark-json",
            str(benchmark_json),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        cwd=ROOT,
    )

    for name in (
        "aegra-audit.json",
        "aegra-findings.json",
        "aegra-graph.json",
        "aegra-state.json",
        "aegra-manifest.json",
        "aegra-report.json",
        "aegra-report.md",
        "aegra-report-no-graph.json",
        "aegra-report-no-graph.md",
    ):
        assert (output_dir / name).exists()

    report = json.loads((output_dir / "aegra-report.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert "steps" in metrics
    assert "loops" in metrics
    assert "incomplete_commands" in metrics
    assert "human_interaction" in metrics
    assert report["target"]["target_id"] == "XBEN-999-24"


def test_xbow_no_graph_report_handles_missing_graph_file(tmp_path: Path) -> None:
    from benchmarks.xbow.run_aegra_xbow import generate_reports, run_evaluate

    benchmark_json = tmp_path / "benchmark.json"
    output_dir = tmp_path / "benchmark-output"
    benchmark_json.write_text(json.dumps(sample_benchmark()), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "benchmark_json": benchmark_json,
            "benchmark_id": "XBEN-999-24",
            "tags": [],
            "level": None,
            "flag_format": "FLAG{...}",
            "timeout": 900,
        },
    )()

    generate_reports(
        args,
        output_dir,
        "http://127.0.0.1:32768/",
        {
            "audit": {"audit_log": []},
            "findings": {"findings": [], "evidence": []},
            "state": {"execution": {"metadata": {}}},
        },
    )
    (output_dir / "aegra-graph.json").unlink()

    assert run_evaluate(output_dir, no_graph=True) == 0
    report = json.loads((output_dir / "aegra-report-no-graph.json").read_text(encoding="utf-8"))
    assert report["metrics"]["kg_node_recall_percent"] is None
    assert report["metrics"]["kg_edge_recall_percent"] is None
