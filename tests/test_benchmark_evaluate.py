from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmarks.evaluate import evaluate, report_to_markdown


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "benchmark"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_report(ablation: str | None = None) -> dict:
    return evaluate(
        load_json(ROOT / "benchmarks" / "vulnhub_local.json"),
        target_id="vulhub-struts2-s2045-local",
        audit_payload=load_json(FIXTURES / "audit.json"),
        findings_payload=load_json(FIXTURES / "findings.json"),
        graph_payload=load_json(FIXTURES / "graph.json"),
        state_payload=load_json(FIXTURES / "state.json"),
        ablation=ablation,
    )


def test_evaluate_fixture_scores_full_coverage() -> None:
    report = fixture_report()
    metrics = report["metrics"]

    assert metrics["service_coverage_percent"] == 100.0
    assert metrics["vulnerability_coverage_percent"] == 100.0
    assert metrics["evidence_chain_completeness_percent"] == 100.0
    assert metrics["false_positive_rate"] == 0
    assert metrics["steps"] == 2
    assert metrics["loops"] == 0
    assert metrics["human_interaction"] == 0
    assert metrics["incomplete_commands"] == 0


def test_no_graph_ablation_outputs_null_graph_metrics() -> None:
    metrics = fixture_report(ablation="no-graph")["metrics"]

    assert metrics["kg_node_recall_percent"] is None
    assert metrics["kg_edge_recall_percent"] is None
    assert metrics["evidence_chain_completeness_percent"] is None


def test_markdown_contains_pentestgpt_alignment_table() -> None:
    markdown = report_to_markdown(fixture_report())

    assert "PentestGPT / AutoPentester Alignment Metrics" in markdown
    assert "| Metric | Alignment | Value |" in markdown
    assert "Service Coverage %" in markdown
    assert "Vulnerability Coverage %" in markdown
    assert "Evidence Chain Completeness" in markdown


def test_evaluate_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "evaluate.py"),
            "--manifest",
            str(ROOT / "benchmarks" / "vulnhub_local.json"),
            "--target-id",
            "vulhub-struts2-s2045-local",
            "--audit",
            str(FIXTURES / "audit.json"),
            "--findings",
            str(FIXTURES / "findings.json"),
            "--graph",
            str(FIXTURES / "graph.json"),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        cwd=ROOT,
    )

    report = load_json(output_json)
    markdown = output_md.read_text(encoding="utf-8")
    assert report["target"]["target_id"] == "vulhub-struts2-s2045-local"
    assert "Service Coverage" in markdown
    assert "Vulnerability Coverage" in markdown
