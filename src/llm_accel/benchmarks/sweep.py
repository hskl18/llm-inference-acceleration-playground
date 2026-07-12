from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.config.loader import get_path, load_config, sanitize_resolved_config, validate_benchmark_config
from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.reports.markdown import write_aggregate_markdown
from llm_accel.reports.plots import write_latency_throughput_svg, write_sweep_svg
from llm_accel.workloads.prompts import load_prompt_file


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def run_sweep(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, object]:
    config = load_config(config_path)
    validate_benchmark_config(config)
    config_file = Path(config_path)
    run_name = get_path(config, "run.name", "sweep")
    base_output_dir = Path(output_dir or get_path(config, "run.output_dir", f"results/runs/{run_name}"))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    base_url = get_path(config, "endpoint.base_url", "mock://local")
    backend = get_path(config, "endpoint.backend", "openai-compatible")
    api_kind = get_path(config, "endpoint.api_kind", "chat")
    model = get_path(config, "model.name", "mock-model")
    dtype = get_path(config, "model.dtype", "unknown")
    quantization = get_path(config, "model.quantization", "none")
    model_revision = get_path(config, "model.revision")
    request_count = int(get_path(config, "run.measured_requests", 8))
    warmup_count = int(get_path(config, "run.warmup_requests", 0))
    timeout_seconds = float(get_path(config, "run.timeout_seconds", 120))
    hardware_label = str(get_path(config, "run.hardware_label", "local"))
    optimization_profile = str(get_path(config, "run.optimization_profile", "baseline"))
    server_command_sha256 = get_path(config, "run.server_command_sha256")
    seed = int(get_path(config, "workload.seed", 42))
    prompts_path = get_path(config, "workload.prompts_path")
    if isinstance(prompts_path, str):
        prompts_path = _resolve_config_path(config_file, prompts_path)
    prompt_texts = load_prompt_file(prompts_path) if isinstance(prompts_path, str) else None

    input_lengths = _as_list(get_path(config, "workload.input_tokens", ["prompts"] if prompt_texts is not None else [128]))
    output_lengths = _as_list(get_path(config, "workload.output_tokens", [64]))
    concurrencies = _as_list(get_path(config, "workload.concurrency", [1]))

    runs: list[dict[str, object]] = []
    for input_tokens in input_lengths:
        for output_tokens in output_lengths:
            for concurrency in concurrencies:
                run_id = f"c{concurrency}-{'prompts' if prompt_texts is not None else f'in{input_tokens}'}-out{output_tokens}"
                summary = run_latency_benchmark(
                    base_url=base_url,
                    model=model,
                    concurrency=int(concurrency),
                    input_tokens=128 if prompt_texts is not None and input_tokens == "prompts" else int(input_tokens),
                    output_tokens=int(output_tokens),
                    output_dir=base_output_dir / run_id,
                    request_count=request_count,
                    warmup_count=warmup_count,
                    timeout_seconds=timeout_seconds,
                    dtype=dtype,
                    quantization=quantization,
                    backend=backend,
                    seed=seed,
                    hardware_label=hardware_label,
                    api_kind=api_kind,
                    prompt_texts=prompt_texts,
                    model_revision=str(model_revision) if model_revision is not None else None,
                    optimization_profile=optimization_profile,
                    server_command_sha256=(
                        str(server_command_sha256) if server_command_sha256 is not None else None
                    ),
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "summary_path": str(base_output_dir / run_id / "summary.json"),
                        "metrics": summary["metrics"],
                        "metadata": summary["metadata"],
                    }
                )

    aggregate = {
        "run_name": run_name,
        "config_path": str(config_path),
        "output_dir": str(base_output_dir),
        "run_count": len(runs),
        "runs": runs,
    }
    write_json(base_output_dir / "resolved_config.json", sanitize_resolved_config(config))
    write_json(base_output_dir / "aggregate_summary.json", aggregate)
    write_aggregate_markdown(base_output_dir / "aggregate_summary.md", aggregate)
    write_sweep_svg(base_output_dir / "plots" / "sweep_throughput.svg", aggregate)
    write_latency_throughput_svg(base_output_dir / "plots" / "latency_throughput.svg", aggregate)
    write_run_manifest(
        base_output_dir,
        run_type="benchmark_sweep",
        artifacts=[
            "manifest.json",
            "resolved_config.json",
            "aggregate_summary.json",
            "aggregate_summary.md",
            "plots/sweep_throughput.svg",
            "plots/latency_throughput.svg",
        ],
    )
    return aggregate


def _resolve_config_path(config_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(config_path.parent / path)
