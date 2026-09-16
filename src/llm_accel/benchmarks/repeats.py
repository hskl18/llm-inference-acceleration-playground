"""Repeat one benchmark configuration and report the spread of its run-level metrics.

A single benchmark run reports percentiles over requests; it says nothing about how much the
result moves when the same configuration is run again. Repeating the run and publishing the
mean with a 95% confidence interval separates a real difference from run-to-run noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from llm_accel.metrics.aggregation import distribution
from llm_accel.metrics.io import write_json, write_text_atomic
from llm_accel.metrics.manifest import write_run_manifest


REPEAT_METRIC_PATHS: tuple[tuple[str, ...], ...] = (
    ("throughput", "output_tokens_per_second"),
    ("throughput", "requests_per_second"),
    ("latency_ms", "p50"),
    ("latency_ms", "p95"),
    ("ttft_ms", "p50"),
    ("ttft_ms", "p95"),
    ("tpot_ms", "p50"),
    ("inter_token_latency_ms", "p50"),
    ("inter_token_latency_ms", "p95"),
    ("goodput", "attainment"),
    ("goodput", "good_requests_per_second"),
)


def run_repeated_benchmark(
    runner: Callable[..., dict[str, object]],
    *,
    repeats: int,
    output_dir: str | Path,
    **benchmark_kwargs: Any,
) -> dict[str, object]:
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to report variance")
    base_dir = Path(output_dir)
    summaries: list[dict[str, object]] = []
    repeat_dirs: list[str] = []
    for repetition in range(1, repeats + 1):
        repeat_id = f"repeat-{repetition:02d}"
        summaries.append(runner(output_dir=base_dir / repeat_id, **benchmark_kwargs))
        repeat_dirs.append(repeat_id)
    return aggregate_repeats(summaries, repeat_dirs=repeat_dirs, output_dir=base_dir)


def aggregate_repeats(
    summaries: list[dict[str, object]],
    *,
    repeat_dirs: list[str],
    output_dir: str | Path,
) -> dict[str, object]:
    """Aggregate repetitions that already exist on disk under `output_dir`.

    Some workloads cannot repeat the identical configuration: a server with a prompt cache
    turns a second pass over the same prompts into a cache-hit measurement, so each repetition
    must use its own fresh prompt set. Those repetitions are separate runs that still belong to
    one aggregate, which is why aggregation is available apart from execution.
    """
    if len(summaries) < 2:
        raise ValueError("at least two repetitions are required to report variance")
    if len(summaries) != len(repeat_dirs):
        raise ValueError("every repetition needs its own directory name")
    base_dir = Path(output_dir)
    repeats = len(summaries)

    payload = {
        "schema_version": "0.2",
        "repeats": repeats,
        "repeat_dirs": repeat_dirs,
        "metadata": summaries[0].get("metadata", {}),
        "metrics": _aggregate_metrics(summaries),
        "failed_request_count": sum(
            int(_nested(summary.get("metrics"), ("failed_count",)) or 0) for summary in summaries
        ),
        "warnings": sorted({warning for summary in summaries for warning in _warnings(summary)}),
        "notes": [
            "Each repetition is an independent run in its own directory under this one.",
            "Confidence intervals are two-sided 95% Student t intervals for the mean of the repetitions.",
            "Repetitions run back to back on one host, so they capture run-to-run noise, not day-to-day drift.",
        ],
    }
    write_json(base_dir / "repeats_summary.json", payload)
    write_text_atomic(base_dir / "repeats_summary.md", _render_markdown(payload))
    write_run_manifest(
        base_dir,
        run_type="repeated_benchmark",
        artifacts=["manifest.json", "repeats_summary.json", "repeats_summary.md"],
    )
    return payload


def _aggregate_metrics(summaries: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}
    for path in REPEAT_METRIC_PATHS:
        values = [_nested(summary.get("metrics"), path) for summary in summaries]
        numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        # A metric that any repetition failed to report cannot be aggregated honestly.
        if len(numbers) == len(summaries):
            aggregated[".".join(path)] = distribution(numbers)
    return aggregated


def _warnings(summary: dict[str, object]) -> list[str]:
    warnings = summary.get("warnings")
    return [str(item) for item in warnings] if isinstance(warnings, list) else []


def _nested(payload: object, path: tuple[str, ...]) -> object:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _render_markdown(payload: dict[str, object]) -> str:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    lines = [
        "# Repeated Benchmark Summary",
        "",
        f"- Model: `{metadata.get('model', 'unknown')}`",
        f"- Backend: `{metadata.get('backend', 'unknown')}`",
        f"- Hardware label: `{metadata.get('hardware_label', 'unknown')}`",
        f"- Concurrency: `{metadata.get('concurrency', 'unknown')}`",
        f"- Output tokens: `{metadata.get('output_tokens', 'unknown')}`",
        f"- Repetitions: `{payload.get('repeats', 0)}`",
        f"- Failed requests across repetitions: `{payload.get('failed_request_count', 0)}`",
        "",
        "| Metric | Mean | 95% CI | Sample stddev | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(metrics):
        stats = metrics[name]
        lines.append(
            f"| `{name}` | {stats['mean']:.3f} | "
            f"[{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}] | "
            f"{stats['sample_stddev']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
        )
    warnings = payload.get("warnings")
    lines.extend(["", "## Warnings", ""])
    lines.extend(
        [f"- {warning}" for warning in warnings] if isinstance(warnings, list) and warnings else ["- None"]
    )
    notes = payload.get("notes")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {note}" for note in notes] if isinstance(notes, list) else [])
    return "\n".join(lines) + "\n"
