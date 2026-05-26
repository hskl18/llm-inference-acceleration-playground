from __future__ import annotations

from pathlib import Path
from typing import Any


def render_summary_markdown(summary: dict[str, Any]) -> str:
    metadata = summary.get("metadata", {})
    metrics = summary.get("metrics", {})
    latency = metrics.get("latency_ms", {})
    ttft = metrics.get("ttft_ms", {})
    tpot = metrics.get("tpot_ms", {})
    throughput = metrics.get("throughput", {})
    memory = summary.get("memory", {})
    warnings = summary.get("warnings", [])

    lines = [
        "# Benchmark Summary",
        "",
        "## Run Metadata",
        "",
        f"- Model: `{metadata.get('model', 'unknown')}`",
        f"- Backend: `{metadata.get('backend', 'unknown')}`",
        f"- Backend version: `{metadata.get('backend_version') or 'unavailable'}`",
        f"- API kind: `{metadata.get('api_kind', 'unknown')}`",
        f"- Concurrency: `{metadata.get('concurrency', 'unknown')}`",
        f"- Input tokens: `{metadata.get('input_tokens', 'unknown')}`",
        f"- Output tokens: `{metadata.get('output_tokens', 'unknown')}`",
        f"- Measured requests: `{metadata.get('request_count', 'unknown')}`",
        f"- Warmup requests: `{metadata.get('warmup_count', 'unknown')}`",
        f"- Hardware label: `{metadata.get('hardware_label', 'unknown')}`",
        f"- GPU name: `{metadata.get('gpu_name') or 'unavailable'}`",
        f"- Python: `{metadata.get('python_version', 'unknown')}`",
        f"- OS: `{metadata.get('operating_system', 'unknown')}`",
        f"- Git commit: `{metadata.get('git_commit') or 'unavailable'}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Completed requests | {metrics.get('completed_count', 0)} |",
        f"| Failed requests | {metrics.get('failed_count', 0)} |",
        f"| Timeout count | {metrics.get('timeout_count', 0)} |",
        f"| Latency p50 | {latency.get('p50', 0.0):.3f} ms |",
        f"| Latency p95 | {latency.get('p95', 0.0):.3f} ms |",
        f"| Latency p99 | {latency.get('p99', 0.0):.3f} ms |",
        f"| TTFT p50 | {ttft.get('p50', 0.0):.3f} ms |",
        f"| TPOT p50 | {tpot.get('p50', 0.0):.3f} ms |",
        f"| Output tokens/sec | {throughput.get('output_tokens_per_second', 0.0):.3f} |",
        f"| Requests/sec | {throughput.get('requests_per_second', 0.0):.3f} |",
        "",
        "## Memory",
        "",
        f"- GPU memory telemetry available: `{memory.get('available', False)}`",
        f"- Delta used memory: `{memory.get('delta_used_mib', 'unknown')}` MiB",
        "",
        "## Warnings",
        "",
        *([f"- {warning}" for warning in warnings] if warnings else ["- None"]),
        "",
        "## Artifacts",
        "",
        "- Raw request records: `raw_requests.jsonl`",
        "- Raw request CSV: `raw_requests.csv`",
        "- Machine-readable summary: `summary.json`",
        "- Latency plot: `plots/latency.svg`",
        "",
        "## Notes",
        "",
        "- This report separates measured request data from later analysis.",
        "- Mock backend results are examples for workflow validation, not hardware performance claims.",
    ]
    return "\n".join(lines) + "\n"


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_summary_markdown(summary), encoding="utf-8")


def render_aggregate_markdown(aggregate: dict[str, object]) -> str:
    rows = []
    for run in aggregate.get("runs", []):  # type: ignore[assignment]
        metrics = run["metrics"]
        metadata = run["metadata"]
        latency = metrics["latency_ms"]
        throughput = metrics["throughput"]
        rows.append(
            f"| `{run['run_id']}` | {metadata['concurrency']} | {metadata['input_tokens']} | "
            f"{metadata['output_tokens']} | {latency['p95']:.3f} | "
            f"{throughput['output_tokens_per_second']:.3f} | {metrics['failed_count']} |"
        )

    return "\n".join(
        [
            "# Sweep Aggregate Summary",
            "",
            f"- Run name: `{aggregate.get('run_name', 'unknown')}`",
            f"- Run count: `{aggregate.get('run_count', 0)}`",
            "",
            "| Run | Concurrency | Input tokens | Output tokens | Latency p95 ms | Output tokens/sec | Failed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "Raw per-run results are stored in each run directory. Aggregate comparisons should only be made across comparable model, backend, dtype, and hardware metadata.",
            "",
        ]
    )


def write_aggregate_markdown(path: Path, aggregate: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_aggregate_markdown(aggregate), encoding="utf-8")
