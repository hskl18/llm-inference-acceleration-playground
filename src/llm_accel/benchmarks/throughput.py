from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest


def run_throughput_benchmark(**benchmark_kwargs: Any) -> dict[str, object]:
    output_dir = Path(benchmark_kwargs["output_dir"])
    summary = run_latency_benchmark(**benchmark_kwargs)
    payload = _build_throughput_summary(summary)
    write_json(output_dir / "throughput_summary.json", payload)
    _write_throughput_markdown(output_dir / "throughput_summary.md", payload)
    write_run_manifest(
        output_dir,
        run_type="throughput_benchmark",
        artifacts=[
            "manifest.json",
            "resolved_config.json",
            "raw_requests.jsonl",
            "raw_requests.csv",
            "run_metadata.json",
            "summary.json",
            "summary.md",
            "plots/latency.svg",
            "throughput_summary.json",
            "throughput_summary.md",
        ],
    )
    return summary


def _build_throughput_summary(summary: dict[str, object]) -> dict[str, object]:
    metrics = summary["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("summary.metrics must be a mapping")
    return {
        "schema_version": summary.get("schema_version"),
        "metadata": summary.get("metadata", {}),
        "throughput": metrics.get("throughput", {}),
        "request_count": metrics.get("request_count"),
        "completed_count": metrics.get("completed_count"),
        "failed_count": metrics.get("failed_count"),
        "timeout_count": metrics.get("timeout_count"),
        "warnings": summary.get("warnings", []),
    }


def _write_throughput_markdown(path: Path, payload: dict[str, object]) -> None:
    metadata = payload.get("metadata", {})
    throughput = payload.get("throughput", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(throughput, dict):
        throughput = {}
    lines = [
        "# Throughput Summary",
        "",
        "## Run",
        "",
        f"- Model: `{metadata.get('model', 'unknown')}`",
        f"- Backend: `{metadata.get('backend', 'unknown')}`",
        f"- Hardware label: `{metadata.get('hardware_label', 'unknown')}`",
        f"- Concurrency: `{metadata.get('concurrency', 'unknown')}`",
        f"- Input tokens: `{metadata.get('input_tokens', 'unknown')}`",
        f"- Output tokens: `{metadata.get('output_tokens', 'unknown')}`",
        "",
        "## Throughput",
        "",
        f"- Output tokens/sec: `{throughput.get('output_tokens_per_second', 0.0)}`",
        f"- Requests/sec: `{throughput.get('requests_per_second', 0.0)}`",
        f"- Measured elapsed seconds: `{throughput.get('measured_elapsed_seconds', 0.0)}`",
        f"- Completed requests: `{payload.get('completed_count', 0)}`",
        f"- Failed requests: `{payload.get('failed_count', 0)}`",
        f"- Timeout count: `{payload.get('timeout_count', 0)}`",
        "",
        "Raw per-request metrics remain in `raw_requests.jsonl` and `raw_requests.csv`.",
    ]
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
