#!/usr/bin/env python3
"""Compare PentestGPT and Aegra XBOW benchmark summaries.

The script aligns two standalone runner summary.json files by benchmark_id and
emits CSV plus Markdown tables. Aegra-specific execution metrics are loaded
from per-benchmark aegra-report.json files when available.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "benchmark_id",
    "classification",
    "pentestgpt_success",
    "aegra_success",
    "pentestgpt_status",
    "aegra_status",
    "pentestgpt_duration_seconds",
    "aegra_duration_seconds",
    "pentestgpt_cost_usd",
    "aegra_cost_usd",
    "pentestgpt_found_flags_count",
    "aegra_found_flags_count",
    "pentestgpt_timeout_occurred",
    "aegra_timeout_occurred",
    "aegra_steps",
    "aegra_loops",
    "aegra_incomplete_commands",
    "aegra_evidence_chain_completeness",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} does not contain a list at results")

    aligned: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        benchmark_id = item.get("benchmark_id")
        if benchmark_id:
            aligned[str(benchmark_id)] = item
    return aligned


def found_flags_count(result: dict[str, Any] | None) -> int | str:
    if not result:
        return ""
    flags = result.get("found_flags")
    if isinstance(flags, list):
        return len(flags)
    return 0


def scalar(result: dict[str, Any] | None, key: str) -> Any:
    if not result:
        return ""
    value = result.get(key)
    return "" if value is None else value


def classify(pentestgpt: dict[str, Any] | None, aegra: dict[str, Any] | None) -> str:
    p_success = bool(pentestgpt and pentestgpt.get("success"))
    a_success = bool(aegra and aegra.get("success"))
    if p_success and a_success:
        return "both success"
    if a_success and not p_success:
        return "Aegra-only success"
    if p_success and not a_success:
        return "PentestGPT-only success"
    return "both failed"


def candidate_report_paths(summary_path: Path, benchmark_id: str, reports_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    if reports_dir:
        paths.append(reports_dir / benchmark_id / "aegra-report.json")
        paths.append(reports_dir / f"{benchmark_id}-aegra-report.json")

    summary_dir = summary_path.resolve().parent
    paths.extend(
        [
            summary_dir / "benchmarks" / benchmark_id / "aegra-report.json",
            summary_dir / benchmark_id / "aegra-report.json",
        ]
    )
    return paths


def load_aegra_metrics(summary_path: Path, aegra_results: dict[str, dict[str, Any]], reports_dir: Path | None) -> dict[str, dict[str, Any]]:
    metrics_by_id: dict[str, dict[str, Any]] = {}
    for benchmark_id in aegra_results:
        for path in candidate_report_paths(summary_path, benchmark_id, reports_dir):
            if not path.exists():
                continue
            report = load_json(path)
            metrics = report.get("metrics") if isinstance(report, dict) else None
            if isinstance(metrics, dict):
                metrics_by_id[benchmark_id] = metrics
                break
    return metrics_by_id


def build_rows(
    pentestgpt_summary: Path,
    aegra_summary: Path,
    reports_dir: Path | None,
) -> list[dict[str, Any]]:
    pentestgpt = load_summary(pentestgpt_summary)
    aegra = load_summary(aegra_summary)
    aegra_metrics = load_aegra_metrics(aegra_summary, aegra, reports_dir)

    rows: list[dict[str, Any]] = []
    for benchmark_id in sorted(set(pentestgpt) | set(aegra)):
        p = pentestgpt.get(benchmark_id)
        a = aegra.get(benchmark_id)
        metrics = aegra_metrics.get(benchmark_id, {})
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "classification": classify(p, a),
                "pentestgpt_success": scalar(p, "success"),
                "aegra_success": scalar(a, "success"),
                "pentestgpt_status": scalar(p, "status"),
                "aegra_status": scalar(a, "status"),
                "pentestgpt_duration_seconds": scalar(p, "duration_seconds"),
                "aegra_duration_seconds": scalar(a, "duration_seconds"),
                "pentestgpt_cost_usd": scalar(p, "cost_usd"),
                "aegra_cost_usd": scalar(a, "cost_usd"),
                "pentestgpt_found_flags_count": found_flags_count(p),
                "aegra_found_flags_count": found_flags_count(a),
                "pentestgpt_timeout_occurred": scalar(p, "timeout_occurred"),
                "aegra_timeout_occurred": scalar(a, "timeout_occurred"),
                "aegra_steps": metrics.get("steps", ""),
                "aegra_loops": metrics.get("loops", ""),
                "aegra_incomplete_commands": metrics.get("incomplete_commands", ""),
                "aegra_evidence_chain_completeness": metrics.get("evidence_chain_completeness_percent", ""),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    counts = {label: 0 for label in ["Aegra-only success", "PentestGPT-only success", "both failed", "both success"]}
    for row in rows:
        counts[str(row["classification"])] = counts.get(str(row["classification"]), 0) + 1

    lines = [
        "# XBOW Comparison Results",
        "",
        "## Classification Summary",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for label in ["Aegra-only success", "PentestGPT-only success", "both success", "both failed"]:
        lines.append(f"| {label} | {counts.get(label, 0)} |")

    lines.extend(["", "## Aligned Results", "", "| " + " | ".join(FIELDS) + " |", "| " + " | ".join(["---"] * len(FIELDS)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(row.get(field, "")) for field in FIELDS) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PentestGPT and Aegra XBOW summary.json files.")
    parser.add_argument("--pentestgpt-summary", required=True, type=Path, help="Path to PentestGPT summary.json.")
    parser.add_argument("--aegra-summary", required=True, type=Path, help="Path to Aegra summary.json.")
    parser.add_argument("--aegra-reports-dir", type=Path, help="Optional directory containing per-benchmark Aegra reports.")
    parser.add_argument("--output-csv", type=Path, default=Path("compare.csv"), help="CSV output path.")
    parser.add_argument("--output-md", type=Path, default=Path("compare.md"), help="Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.pentestgpt_summary, args.aegra_summary, args.aegra_reports_dir)
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
