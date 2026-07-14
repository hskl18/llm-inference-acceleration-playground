from __future__ import annotations

from pathlib import Path

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.metrics.io import write_json, write_text_atomic
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.quantization.sanity import DEFAULT_SANITY_PROMPTS, run_quality_sanity_check
from llm_accel.serving.capabilities import get_capability


def compare_quantization_modes(
    *,
    base_url: str,
    model: str,
    modes: list[str],
    output_dir: str | Path,
    concurrency: int = 1,
    input_tokens: int = 128,
    output_tokens: int = 64,
    request_count: int = 8,
    backend: str = "openai-compatible",
    hardware_label: str = "local",
    sanity_prompts: list[str] | None = None,
) -> dict[str, object]:
    if not modes:
        raise ValueError("at least one quantization mode is required")

    out_dir = Path(output_dir)
    runs: list[dict[str, object]] = []
    warnings: list[str] = []
    baseline_tokens_per_second: float | None = None
    supported_modes = [str(mode) for mode in get_capability(backend).get("quantization_modes", ["unknown"])]
    for mode in modes:
        support_status = _support_status(mode, supported_modes)
        if support_status == "unsupported":
            warning = f"Quantization mode {mode!r} is not listed as supported for backend {backend!r}."
            warnings.append(warning)
            runs.append(
                {
                    "quantization": mode,
                    "support_status": support_status,
                    "measured": False,
                    "summary_path": None,
                    "output_tokens_per_second": None,
                    "relative_to_first_mode": None,
                    "failed_count": None,
                    "quality_sanity": None,
                    "warning": warning,
                }
            )
            continue
        if support_status == "unknown":
            warnings.append(
                f"Quantization mode support is unknown for backend {backend!r}; benchmark result is endpoint-defined."
            )
        run_dir = out_dir / mode
        summary = run_latency_benchmark(
            base_url=base_url,
            model=model,
            concurrency=concurrency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_dir=run_dir,
            request_count=request_count,
            quantization=mode,
            backend=backend,
            hardware_label=hardware_label,
        )
        metrics = summary["metrics"]
        sanity = run_quality_sanity_check(
            base_url=base_url,
            model=model,
            backend=backend,
            quantization=mode,
            prompts=sanity_prompts or DEFAULT_SANITY_PROMPTS,
        )
        throughput = metrics["throughput"]["output_tokens_per_second"]  # type: ignore[index]
        if baseline_tokens_per_second is None:
            baseline_tokens_per_second = float(throughput)
        relative = float(throughput) / baseline_tokens_per_second if baseline_tokens_per_second else 0.0
        runs.append(
            {
                "quantization": mode,
                "support_status": support_status,
                "measured": True,
                "summary_path": str(run_dir / "summary.json"),
                "output_tokens_per_second": throughput,
                "relative_to_first_mode": relative,
                "failed_count": metrics["failed_count"],  # type: ignore[index]
                "quality_sanity": sanity,
            }
        )

    report = {
        "model": model,
        "backend": backend,
        "base_url": base_url if base_url.startswith(("mock://", "http://localhost", "http://127.0.0.1")) else "redacted",
        "hardware_label": hardware_label,
        "modes": modes,
        "supported_modes": supported_modes,
        "runs": runs,
        "warnings": warnings,
        "notes": [
            "This comparison records benchmark metadata for each mode.",
            "Mock backend comparisons validate workflow only; they are not hardware quantization claims.",
        ],
    }
    write_json(out_dir / "quantization_comparison.json", report)
    _write_markdown(out_dir / "quantization_comparison.md", report)
    write_run_manifest(
        out_dir,
        run_type="quantization_comparison",
        artifacts=[
            "manifest.json",
            "quantization_comparison.json",
            "quantization_comparison.md",
        ],
    )
    return report


def _support_status(mode: str, supported_modes: list[str]) -> str:
    if supported_modes == ["unknown"]:
        return "unknown"
    return "supported" if mode in supported_modes else "unsupported"


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    rows = []
    for run in report["runs"]:  # type: ignore[index]
        sanity = run["quality_sanity"] or {}
        throughput = run["output_tokens_per_second"]
        relative = run["relative_to_first_mode"]
        throughput_text = f"{throughput:.3f}" if isinstance(throughput, (int, float)) else "not measured"
        relative_text = f"{relative:.3f}" if isinstance(relative, (int, float)) else "not measured"
        failed_count = run["failed_count"] if run["failed_count"] is not None else "not measured"
        rows.append(
            f"| `{run['quantization']}` | {run['support_status']} | {run['measured']} | {throughput_text} | "
            f"{relative_text} | {failed_count} | {sanity.get('passed', 'not measured')} |"
        )
    warning_lines = [f"- {warning}" for warning in report.get("warnings", [])] or ["- None"]
    text = "\n".join(
        [
            "# Quantization Comparison",
            "",
            f"- Model: `{report['model']}`",
            f"- Backend: `{report['backend']}`",
            f"- Supported modes: `{', '.join(report['supported_modes'])}`",
            "",
            "## Warnings",
            "",
            *warning_lines,
            "",
            "| Mode | Support status | Measured | Output tokens/sec | Relative to first measured mode | Failed requests | Quality sanity passed |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "Unsupported modes are not benchmarked and do not produce performance claims.",
            "Mock backend results are workflow validation only, not hardware performance claims.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, text)
