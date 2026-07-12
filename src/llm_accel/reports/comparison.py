from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

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
                "backend_version": metadata.get("backend_version"),
                "project_version": metadata.get("project_version"),
                "git_commit": metadata.get("git_commit"),
                "python_version": metadata.get("python_version"),
                "operating_system": metadata.get("operating_system"),
                "api_kind": metadata.get("api_kind", "chat"),
                "stream": metadata.get("stream", True),
                "dtype": metadata.get("dtype", "unknown"),
                "quantization": metadata.get("quantization", "unknown"),
                "model_revision": metadata.get("model_revision"),
                "optimization_profile": metadata.get("optimization_profile", "baseline"),
                "server_command_sha256": metadata.get("server_command_sha256"),
                "hardware_label": metadata.get("hardware_label", "unknown"),
                "gpu_driver_version": metadata.get("gpu_driver_version"),
                "cuda_version": metadata.get("cuda_version"),
                "torch_version": metadata.get("torch_version"),
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
    has_failed_runs = any(int(row.get("failed_count", 0)) > 0 for row in rows)
    profile_aggregates = _profile_aggregates(rows) if comparable and not has_failed_runs else []
    report = {
        "summary_count": len(rows),
        "runs": rows,
        "comparable": comparable,
        "ranking_allowed": ranking_allowed,
        "warnings": warnings,
        "profile_aggregates": profile_aggregates,
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


def _profile_aggregates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("optimization_profile", "baseline"))].append(row)
    aggregates = []
    for profile, profile_rows in sorted(groups.items()):
        if len(profile_rows) < 3:
            continue
        throughput = [float(row["output_tokens_per_second"]) for row in profile_rows]
        latency_p95 = [float(row["latency_p95_ms"]) for row in profile_rows]
        aggregates.append(
            {
                "optimization_profile": profile,
                "repetitions": len(profile_rows),
                "output_tokens_per_second": _distribution(throughput),
                "latency_p95_ms": _distribution(latency_p95),
            }
        )
    return aggregates


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "stddev": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _comparison_warnings(rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    compatibility_keys = [
        "model",
        "backend",
        "backend_version",
        "project_version",
        "git_commit",
        "python_version",
        "operating_system",
        "api_kind",
        "stream",
        "dtype",
        "quantization",
        "model_revision",
        "hardware_label",
        "gpu_driver_version",
        "cuda_version",
        "torch_version",
        "workload_mode",
        "workload_fingerprint",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "request_count",
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
    profile_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        profile_counts[str(row.get("optimization_profile", "baseline"))] += 1
    for profile, count in sorted(profile_counts.items()):
        if count < 3:
            warnings.append(
                f"Optimization profile {profile!r} has only {count} repetitions; at least 3 are required for ranking."
            )
    for profile in sorted(profile_counts):
        hashes = {
            row.get("server_command_sha256")
            for row in rows
            if str(row.get("optimization_profile", "baseline")) == profile
        }
        if len(hashes) > 1:
            warnings.append(
                f"Runs are not comparable: server command fingerprint differs within profile {profile!r}."
            )
    profiles_by_hash: dict[object, set[str]] = defaultdict(set)
    for row in rows:
        profiles_by_hash[row.get("server_command_sha256")].add(
            str(row.get("optimization_profile", "baseline"))
        )
    for command_hash, profiles in profiles_by_hash.items():
        if command_hash is not None and len(profiles) > 1:
            warnings.append(
                "Runs are not comparable: the same server command fingerprint is labeled as multiple optimization profiles."
            )
    return warnings


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    warnings = report.get("warnings", [])
    rows = [
        f"| `{run['summary_path']}` | `{run['optimization_profile']}` | `{run['backend']}` | {run['concurrency']} | "
        f"{run['request_count']} | {run['latency_p95_ms']:.3f} | "
        f"{run['output_tokens_per_second']:.3f} | {run['relative_to_first']:.3f} | {run['failed_count']} |"
        for run in report["runs"]  # type: ignore[index]
    ]
    aggregate_rows = [
        f"| `{aggregate['optimization_profile']}` | {aggregate['repetitions']} | "
        f"{aggregate['output_tokens_per_second']['mean']:.3f} | "
        f"{aggregate['output_tokens_per_second']['stddev']:.3f} | "
        f"{aggregate['latency_p95_ms']['mean']:.3f} | "
        f"{aggregate['latency_p95_ms']['stddev']:.3f} |"
        for aggregate in report["profile_aggregates"]  # type: ignore[index]
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
            "| Summary | Profile | Backend | Concurrency | Requests | p95 latency ms | Output tokens/sec | Relative throughput | Failed |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Repeated-run aggregates",
            "",
            "| Profile | Repetitions | Mean output tokens/sec | Stddev | Mean p95 latency ms | Stddev |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *aggregate_rows,
            "",
            "Relative throughput is an inspection aid, not a ranking, when warnings are present.",
            "Only compare runs with compatible model, backend, dtype, hardware, and workload metadata.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
