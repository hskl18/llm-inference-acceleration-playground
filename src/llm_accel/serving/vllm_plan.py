from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.serving.vllm import build_vllm_command, optimization_profile_name


def create_vllm_benchmark_plan(
    *,
    model: str,
    base_url: str,
    output_dir: str | Path,
    config_path: str = "configs/benchmark_vllm_small.yaml",
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str,
    revision: str,
    hardware_label: str = "REQUIRED_HARDWARE_LABEL",
    quantization: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    speculative_model: str | None = None,
    num_speculative_tokens: int | None = None,
) -> dict[str, object]:
    command = build_vllm_command(
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        revision=revision,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
    )
    server_command_text = command.shell_command() + "\n"
    server_command_sha256 = hashlib.sha256(server_command_text.encode("utf-8")).hexdigest()
    out_dir = Path(output_dir)
    server_command_path = out_dir / "server_command.txt"
    revision_value = revision
    optimization_profile = optimization_profile_name(
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        speculative_model=speculative_model,
        quantization=quantization,
    )
    validation_command = _validation_command(
        model=model,
        base_url=base_url,
        revision=revision,
        dtype=command.dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
    )
    latency_command = _benchmark_command(
        kind="latency",
        base_url=base_url,
        model=model,
        revision=revision,
        server_command_sha256=server_command_sha256,
        server_command_path=server_command_path,
        dtype=command.dtype,
        quantization=quantization or "none",
        hardware_label=hardware_label,
        optimization_profile=optimization_profile,
        concurrency=4,
        output_dir="results/runs/vllm-latency",
    )
    throughput_command = _benchmark_command(
        kind="throughput",
        base_url=base_url,
        model=model,
        revision=revision,
        server_command_sha256=server_command_sha256,
        server_command_path=server_command_path,
        dtype=command.dtype,
        quantization=quantization or "none",
        hardware_label=hardware_label,
        optimization_profile=optimization_profile,
        concurrency=8,
        output_dir="results/runs/vllm-throughput",
    )
    plan = {
        "model": model,
        "model_revision": revision_value,
        "base_url": base_url if base_url.startswith(("http://localhost", "http://127.0.0.1")) else "redacted",
        "config_path": config_path,
        "server_command": command.to_dict(),
        "server_command_sha256": server_command_sha256,
        "steps": [
            {
                "name": "start_vllm_server",
                "command": shlex.join(["bash", str(server_command_path)]),
            },
            {
                "name": "validate_environment",
                "command": validation_command,
            },
            {
                "name": "run_latency_benchmark",
                "command": latency_command,
            },
            {
                "name": "run_throughput_benchmark",
                "command": throughput_command,
            },
            {
                "name": "run_config_sweep",
                "command": shlex.join(["llm-accel", "bench", "sweep", "--config", config_path]),
            },
            {
                "name": "run_quality_sanity",
                "command": shlex.join(
                    [
                        "llm-accel",
                        "eval",
                        "task",
                        "--base-url",
                        base_url,
                        "--backend",
                        "vllm",
                        "--model",
                        model,
                        "--tasks",
                        "configs/task_eval_small.jsonl",
                        "--output-dir",
                        "results/runs/vllm-task-eval",
                    ]
                ),
            },
            {
                "name": "validate_latency_run",
                "command": shlex.join(
                    ["llm-accel", "report", "validate", "--run-dir", "results/runs/vllm-latency"]
                ),
            },
            {
                "name": "validate_throughput_run",
                "command": shlex.join(
                    ["llm-accel", "report", "validate", "--run-dir", "results/runs/vllm-throughput"]
                ),
            },
        ],
        "required_artifacts": [
            "results/runs/vllm-validation/vllm_validation.json",
            "results/runs/vllm-latency/summary.json",
            "results/runs/vllm-latency/raw_requests.jsonl",
            "results/runs/vllm-latency/plots/latency.svg",
            "results/runs/vllm-throughput/throughput_summary.json",
            "results/runs/vllm-throughput/raw_requests.jsonl",
            "results/runs/vllm-task-eval/task_eval.json",
            str(server_command_path),
        ],
        "claim_rules": [
            "Do not publish performance claims if vllm validate reports blockers.",
            "Do not compare runs unless model, backend, dtype, quantization, hardware, and workload metadata are compatible.",
            "Report failed request counts and timeout counts with benchmark results.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    server_command_path.write_text(server_command_text, encoding="utf-8")
    write_json(out_dir / "vllm_benchmark_plan.json", plan)
    _write_markdown(out_dir / "vllm_benchmark_plan.md", plan)
    write_run_manifest(
        out_dir,
        run_type="vllm_benchmark_plan",
        artifacts=[
            "manifest.json",
            "vllm_benchmark_plan.json",
            "vllm_benchmark_plan.md",
            "server_command.txt",
        ],
    )
    return plan


def _benchmark_command(
    *,
    kind: str,
    base_url: str,
    model: str,
    revision: str,
    server_command_sha256: str,
    server_command_path: Path,
    dtype: str,
    quantization: str,
    hardware_label: str,
    optimization_profile: str,
    concurrency: int,
    output_dir: str,
) -> str:
    return shlex.join(
        [
            "llm-accel",
            "bench",
            kind,
            "--base-url",
            base_url,
            "--backend",
            "vllm",
            "--model",
            model,
            "--model-revision",
            revision,
            "--server-command-sha256",
            server_command_sha256,
            "--server-command-file",
            str(server_command_path),
            "--dtype",
            dtype,
            "--quantization",
            quantization,
            "--hardware-label",
            hardware_label,
            "--optimization-profile",
            optimization_profile,
            "--concurrency",
            str(concurrency),
            "--input-tokens",
            "512",
            "--output-tokens",
            "128",
            "--warmup-count",
            "8",
            "--request-count",
            "128",
            "--output-dir",
            output_dir,
        ]
    )


def _validation_command(
    *,
    model: str,
    base_url: str,
    revision: str,
    dtype: str,
    quantization: str | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
    speculative_model: str | None,
    num_speculative_tokens: int | None,
) -> str:
    argv = [
        "llm-accel",
        "vllm",
        "validate",
        "--model",
        model,
        "--base-url",
        base_url,
        "--revision",
        revision,
        "--dtype",
        dtype,
    ]
    optional_values = [
        ("--quantization", quantization if quantization != "none" else None),
        ("--max-model-len", max_model_len),
        ("--gpu-memory-utilization", gpu_memory_utilization),
        ("--max-num-batched-tokens", max_num_batched_tokens),
        ("--max-num-seqs", max_num_seqs),
        ("--speculative-model", speculative_model),
        ("--num-speculative-tokens", num_speculative_tokens),
    ]
    for flag, value in optional_values:
        if value is not None:
            argv.extend([flag, str(value)])
    if enable_prefix_caching:
        argv.append("--enable-prefix-caching")
    if enable_chunked_prefill:
        argv.append("--enable-chunked-prefill")
    argv.extend(["--output-dir", "results/runs/vllm-validation", "--smoke"])
    return shlex.join(argv)


def _write_markdown(path: Path, plan: dict[str, object]) -> None:
    step_lines = []
    for index, step in enumerate(plan["steps"], start=1):  # type: ignore[index]
        step_lines.extend(
            [
                f"## {index}. {step['name']}",
                "",
                "```bash",
                str(step["command"]),
                "```",
                "",
            ]
        )
    artifacts = [f"- `{artifact}`" for artifact in plan["required_artifacts"]]  # type: ignore[index]
    rules = [f"- {rule}" for rule in plan["claim_rules"]]  # type: ignore[index]
    text = "\n".join(
        [
            "# vLLM Benchmark Plan",
            "",
            f"- Model: `{plan['model']}`",
            f"- Model revision: `{plan['model_revision']}`",
            f"- Server command SHA-256: `{plan['server_command_sha256']}`",
            f"- Base URL: `{plan['base_url']}`",
            f"- Sweep config: `{plan['config_path']}`",
            "",
            *step_lines,
            "## Required Artifacts",
            "",
            *artifacts,
            "",
            "## Claim Rules",
            "",
            *rules,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
