from __future__ import annotations

import json
from pathlib import Path

from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest


MIN_RANKING_REQUESTS = 8


def compare_run_summaries(summary_paths: list[str | Path], output_dir: str | Path) -> dict[str, object]:
    if len(summary_paths) < 2:
        raise ValueError("at least two summary paths are required")

    rows = []
    baseline_throughput: float | None = None
    for summary_path in summary_paths:
        path = Path(summary_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        metadata = summary["metadata"]
        metrics = summary["metrics"]
        throughput = float(metrics["throughput"]["output_tokens_per_second"])
        if baseline_throughput is None:
            baseline_throughput = throughput
        rows.append(
            {
                "summary_path": str(path),
                "model": metadata["model"],
                "backend": metadata["backend"],
                "dtype": metadata.get("dtype", "unknown"),
                "quantization": metadata.get("quantization", "unknown"),
                "hardware_label": metadata.get("hardware_label", "unknown"),
                "workload_mode": metadata.get("workload_mode", "synthetic"),
                "workload_fingerprint": metadata.get("workload_fingerprint"),
                "concurrency": metadata["concurrency"],
                "input_tokens": metadata["input_tokens"],
                "output_tokens": metadata["output_tokens"],
                "request_count": metrics["request_count"],
                "failed_count": metrics["failed_count"],
                "latency_p95_ms": metrics["latency_ms"]["p95"],
                "output_tokens_per_second": throughput,
                "relative_to_first": throughput / baseline_throughput if baseline_throughput else 0.0,
            }
        )

    warnings = _comparison_warnings(rows)
    comparable = not any("not comparable" in warning for warning in warnings)
    ranking_allowed = comparable and not warnings
    report = {
        "summary_count": len(rows),
        "runs": rows,
        "comparable": comparable,
        "ranking_allowed": ranking_allowed,
        "warnings": warnings,
        "notes": [
            "Compare only runs with compatible model, backend, dtype, hardware, and workload metadata.",
            "Relative throughput uses the first summary as baseline.",
            "Relative throughput is not a ranking when warnings are present.",
        ],
    }
    out_dir = Path(output_dir)
    write_json(out_dir / "comparison.json", report)
    _write_markdown(out_dir / "comparison.md", report)
    write_run_manifest(
        out_dir,
        run_type="run_comparison",
        artifacts=["manifest.json", "comparison.json", "comparison.md"],
    )
    return report


def _comparison_warnings(rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    compatibility_keys = [
        "model",
        "backend",
        "dtype",
        "quantization",
        "hardware_label",
        "workload_mode",
        "workload_fingerprint",
        "input_tokens",
        "output_tokens",
    ]
    for key in compatibility_keys:
        values = {row.get(key) for row in rows}
        if len(values) > 1:
            warnings.append(f"Runs are not comparable: {key} differs across summaries.")
    for row in rows:
        if int(row.get("request_count", 0)) < MIN_RANKING_REQUESTS:
            warnings.append(
                f"Run {row['summary_path']} has only {row.get('request_count', 0)} measured requests; ranking is not justified."
            )
        if int(row.get("failed_count", 0)) > 0:
            warnings.append(f"Run {row['summary_path']} has failed requests; inspect raw metrics before comparing.")
    return warnings


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    warnings = report.get("warnings", [])
    rows = [
        f"| `{run['summary_path']}` | `{run['backend']}` | {run['concurrency']} | "
        f"{run['request_count']} | {run['latency_p95_ms']:.3f} | "
        f"{run['output_tokens_per_second']:.3f} | {run['relative_to_first']:.3f} | {run['failed_count']} |"
        for run in report["runs"]  # type: ignore[index]
    ]
    text = "\n".join(
        [
            "# Run Comparison",
            "",
            f"- Comparable: `{report['comparable']}`",
            f"- Ranking allowed: `{report['ranking_allowed']}`",
            "",
            "## Warnings",
            "",
            *([f"- {warning}" for warning in warnings] if warnings else ["- None"]),
            "",
            "| Summary | Backend | Concurrency | Requests | p95 latency ms | Output tokens/sec | Relative throughput | Failed |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "Relative throughput is an inspection aid, not a ranking, when warnings are present.",
            "Only compare runs with compatible model, backend, dtype, hardware, and workload metadata.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
