from __future__ import annotations

import json
from pathlib import Path
from urllib import error as urllib_error
from urllib import request

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.metrics.execution_identity import displayed_base_url, endpoint_sha256
from llm_accel.metrics.io import write_json, write_text_atomic
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.quantization.sanity import (
    DEFAULT_PERPLEXITY_TEXT,
    DEFAULT_SANITY_PROMPTS,
    measure_prompt_perplexity,
    run_quality_sanity_check,
)
from llm_accel.serving.capabilities import get_capability
from llm_accel.serving.openai_client import DEFAULT_API_KEY_ENV, bearer_auth_header


def compare_quantization_modes(
    *,
    model: str,
    mode_endpoints: dict[str, str],
    output_dir: str | Path,
    concurrency: int = 1,
    input_tokens: int = 128,
    output_tokens: int = 64,
    request_count: int = 8,
    backend: str = "openai-compatible",
    hardware_label: str = "local",
    sanity_prompts: list[str] | None = None,
    perplexity_text: str = DEFAULT_PERPLEXITY_TEXT,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, object]:
    """Benchmark one quantization mode per endpoint, each served by its own process.

    A quantization mode is a property of the loaded weights, not of the request, so the same
    endpoint cannot represent two modes. Identical endpoints are rejected instead of being
    relabelled, and quality is scored against the first (baseline) mode.
    """
    if not mode_endpoints:
        raise ValueError("at least one quantization mode and endpoint is required")
    _require_distinct_endpoints(mode_endpoints)

    modes = list(mode_endpoints)
    baseline_mode = modes[0]
    out_dir = Path(output_dir)
    runs: list[dict[str, object]] = []
    warnings: list[str] = []
    baseline_tokens_per_second: float | None = None
    baseline_outputs: list[str | None] | None = None
    baseline_perplexity: float | None = None
    supported_modes = [str(mode) for mode in get_capability(backend).get("quantization_modes", ["unknown"])]
    selected_prompts = sanity_prompts or DEFAULT_SANITY_PROMPTS
    for mode in modes:
        base_url = mode_endpoints[mode]
        served = _served_models(base_url, api_key_env)
        if served["reachable"] and model not in served["models"]:
            warnings.append(
                f"Endpoint for mode {mode!r} does not list model {model!r} in /models; "
                "the served model identity could not be confirmed."
            )
        support_status = _support_status(mode, supported_modes)
        if support_status == "unsupported":
            warning = f"Quantization mode {mode!r} is not listed as supported for backend {backend!r}."
            warnings.append(warning)
            runs.append(
                {
                    "quantization": mode,
                    "base_url": displayed_base_url(base_url),
                    "endpoint_sha256": endpoint_sha256(base_url),
                    "served_models": served,
                    "support_status": support_status,
                    "measured": False,
                    "summary_path": None,
                    "output_tokens_per_second": None,
                    "relative_to_baseline_mode": None,
                    "failed_count": None,
                    "quality_sanity": None,
                    "quality_vs_baseline": None,
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
            api_key_env=api_key_env,
        )
        metrics = summary["metrics"]
        sanity = run_quality_sanity_check(
            base_url=base_url,
            model=model,
            backend=backend,
            quantization=mode,
            prompts=selected_prompts,
            api_key_env=api_key_env,
        )
        perplexity = measure_prompt_perplexity(
            base_url=base_url,
            model=model,
            text=perplexity_text,
            api_key_env=api_key_env,
        )
        outputs = [check["output_text"] for check in sanity["checks"]]  # type: ignore[index]
        if mode == baseline_mode:
            baseline_outputs = outputs
            baseline_perplexity = perplexity["perplexity"] if perplexity["supported"] else None  # type: ignore[index]
        quality_vs_baseline = _quality_vs_baseline(
            baseline_mode=baseline_mode,
            mode=mode,
            outputs=outputs,
            baseline_outputs=baseline_outputs,
            perplexity=perplexity,
            baseline_perplexity=baseline_perplexity,
        )
        if not perplexity["supported"]:
            warnings.append(
                f"Endpoint for mode {mode!r} does not support echoed prompt logprobs; "
                "no perplexity delta is reported."
            )
        if quality_vs_baseline["exact_match_rate"] is not None and quality_vs_baseline["exact_match_rate"] < 1.0:
            warnings.append(
                f"Mode {mode!r} does not reproduce the baseline mode's temperature-0 outputs "
                f"(exact-match rate {quality_vs_baseline['exact_match_rate']:.3f})."
            )
        throughput = metrics["throughput"]["output_tokens_per_second"]  # type: ignore[index]
        if baseline_tokens_per_second is None:
            baseline_tokens_per_second = float(throughput)
        relative = float(throughput) / baseline_tokens_per_second if baseline_tokens_per_second else 0.0
        runs.append(
            {
                "quantization": mode,
                "base_url": displayed_base_url(base_url),
                "endpoint_sha256": endpoint_sha256(base_url),
                "served_models": served,
                "support_status": support_status,
                "measured": True,
                "summary_path": str(run_dir / "summary.json"),
                "output_tokens_per_second": throughput,
                "relative_to_baseline_mode": relative,
                "failed_count": metrics["failed_count"],  # type: ignore[index]
                "quality_sanity": sanity,
                "quality_vs_baseline": quality_vs_baseline,
            }
        )

    report = {
        "model": model,
        "backend": backend,
        "baseline_mode": baseline_mode,
        "hardware_label": hardware_label,
        "modes": modes,
        "supported_modes": supported_modes,
        "perplexity_text_chars": len(perplexity_text),
        "runs": runs,
        "warnings": warnings,
        "notes": [
            "Each mode is measured on its own endpoint; one endpoint cannot represent two quantization modes.",
            "Quality is scored against the baseline mode with temperature-0 exact match, plus a perplexity delta where the backend returns prompt logprobs.",
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


def _require_distinct_endpoints(mode_endpoints: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    for mode, base_url in mode_endpoints.items():
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(f"quantization mode {mode!r} requires a non-empty endpoint URL")
        digest = endpoint_sha256(base_url)
        if digest in seen:
            raise ValueError(
                f"quantization modes {seen[digest]!r} and {mode!r} share one endpoint; "
                "each mode needs its own server started with that quantization"
            )
        seen[digest] = mode


def _served_models(base_url: str, api_key_env: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    if base_url.startswith("mock://"):
        return {"reachable": False, "models": [], "error": "mock endpoint"}
    req = request.Request(
        f"{base_url.rstrip('/')}/models",
        method="GET",
        headers={"Accept": "application/json", **bearer_auth_header(api_key_env)},
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        entries = body["data"]
    except (urllib_error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
        return {"reachable": False, "models": [], "error": f"{type(exc).__name__}: {exc}"}
    models = [entry["id"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)]
    return {"reachable": True, "models": models, "error": None}


def _quality_vs_baseline(
    *,
    baseline_mode: str,
    mode: str,
    outputs: list[str | None],
    baseline_outputs: list[str | None] | None,
    perplexity: dict[str, object],
    baseline_perplexity: float | None,
) -> dict[str, object]:
    exact_match_rate: float | None = None
    if baseline_outputs is not None and len(baseline_outputs) == len(outputs) and outputs:
        matches = sum(
            1
            for actual, expected in zip(outputs, baseline_outputs, strict=True)
            if actual is not None and actual == expected
        )
        exact_match_rate = matches / len(outputs)
    value = perplexity["perplexity"] if perplexity["supported"] else None
    delta = (
        float(value) - baseline_perplexity
        if isinstance(value, float) and baseline_perplexity is not None
        else None
    )
    return {
        "baseline_mode": baseline_mode,
        "is_baseline": mode == baseline_mode,
        "compared_prompt_count": len(outputs),
        "exact_match_rate": exact_match_rate,
        "prompt_logprobs_supported": bool(perplexity["supported"]),
        "perplexity": value,
        "perplexity_delta_from_baseline": delta,
        "perplexity_error": perplexity["error"],
    }


def _support_status(mode: str, supported_modes: list[str]) -> str:
    if supported_modes == ["unknown"]:
        return "unknown"
    return "supported" if mode in supported_modes else "unsupported"


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    rows = []
    for run in report["runs"]:  # type: ignore[index]
        sanity = run["quality_sanity"] or {}
        quality = run["quality_vs_baseline"] or {}
        throughput = run["output_tokens_per_second"]
        relative = run["relative_to_baseline_mode"]
        throughput_text = f"{throughput:.3f}" if isinstance(throughput, (int, float)) else "not measured"
        relative_text = f"{relative:.3f}" if isinstance(relative, (int, float)) else "not measured"
        failed_count = run["failed_count"] if run["failed_count"] is not None else "not measured"
        rows.append(
            f"| `{run['quantization']}` | {run['support_status']} | {run['measured']} | {throughput_text} | "
            f"{relative_text} | {failed_count} | {sanity.get('passed', 'not measured')} | "
            f"{_number(quality.get('exact_match_rate'))} | {_number(quality.get('perplexity_delta_from_baseline'))} |"
        )
    warning_lines = [f"- {warning}" for warning in report.get("warnings", [])] or ["- None"]
    endpoint_lines = [
        f"- `{run['quantization']}`: `{run['base_url']}` (endpoint SHA-256 `{run['endpoint_sha256']}`)"
        for run in report["runs"]  # type: ignore[index]
    ]
    text = "\n".join(
        [
            "# Quantization Comparison",
            "",
            f"- Model: `{report['model']}`",
            f"- Backend: `{report['backend']}`",
            f"- Baseline mode: `{report['baseline_mode']}`",
            f"- Supported modes: `{', '.join(report['supported_modes'])}`",
            "",
            "## Endpoints",
            "",
            *endpoint_lines,
            "",
            "## Warnings",
            "",
            *warning_lines,
            "",
            "| Mode | Support status | Measured | Output tokens/sec | Relative to baseline mode | "
            "Failed requests | Quality sanity passed | Exact match vs baseline | Perplexity delta |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
            *rows,
            "",
            "Each mode was measured on its own endpoint, so a mode label reflects a separately started server.",
            "Unsupported modes are not benchmarked and do not produce performance claims.",
            "Mock backend results are workflow validation only, not hardware performance claims.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, text)


def _number(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "not measured"
