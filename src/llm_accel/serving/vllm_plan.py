from __future__ import annotations

from pathlib import Path

from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.serving.vllm import build_vllm_command


def create_vllm_benchmark_plan(
    *,
    model: str,
    base_url: str,
    output_dir: str | Path,
    config_path: str = "configs/benchmark_vllm_small.yaml",
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str = "auto",
    quantization: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> dict[str, object]:
    command = build_vllm_command(
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    plan = {
        "model": model,
        "base_url": base_url if base_url.startswith(("http://localhost", "http://127.0.0.1")) else "redacted",
        "config_path": config_path,
        "server_command": command.to_dict(),
        "steps": [
            {
                "name": "validate_environment",
                "command": (
                    f"llm-accel vllm validate --model {model} --base-url {base_url} "
                    "--output-dir results/runs/vllm-validation --smoke"
                ),
            },
            {
                "name": "start_vllm_server",
                "command": command.shell_command(),
            },
            {
                "name": "run_latency_benchmark",
                "command": (
                    f"llm-accel bench latency --base-url {base_url} --backend vllm "
                    f"--model {model} --concurrency 4 --input-tokens 512 --output-tokens 128 "
                    "--request-count 16 --output-dir results/runs/vllm-latency"
                ),
            },
            {
                "name": "run_throughput_benchmark",
                "command": (
                    f"llm-accel bench throughput --base-url {base_url} --backend vllm "
                    f"--model {model} --concurrency 8 --input-tokens 512 --output-tokens 128 "
                    "--request-count 32 --output-dir results/runs/vllm-throughput"
                ),
            },
            {
                "name": "run_config_sweep",
                "command": f"llm-accel bench sweep --config {config_path}",
            },
            {
                "name": "run_quality_sanity",
                "command": (
                    f"llm-accel eval task --base-url {base_url} --backend vllm --model {model} "
                    "--tasks configs/task_eval_small.jsonl --output-dir results/runs/vllm-task-eval"
                ),
            },
            {
                "name": "validate_latency_run",
                "command": "llm-accel report validate --run-dir results/runs/vllm-latency",
            },
            {
                "name": "validate_throughput_run",
                "command": "llm-accel report validate --run-dir results/runs/vllm-throughput",
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
        ],
        "claim_rules": [
            "Do not publish performance claims if vllm validate reports blockers.",
            "Do not compare runs unless model, backend, dtype, quantization, hardware, and workload metadata are compatible.",
            "Report failed request counts and timeout counts with benchmark results.",
        ],
    }
    out_dir = Path(output_dir)
    write_json(out_dir / "vllm_benchmark_plan.json", plan)
    _write_markdown(out_dir / "vllm_benchmark_plan.md", plan)
    write_run_manifest(
        out_dir,
        run_type="vllm_benchmark_plan",
        artifacts=["manifest.json", "vllm_benchmark_plan.json", "vllm_benchmark_plan.md"],
    )
    return plan


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
