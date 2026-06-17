from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LATENCY_OUT = ROOT / "results" / "runs" / "smoke-local"
THROUGHPUT_OUT = ROOT / "results" / "runs" / "smoke-throughput"
SWEEP_OUT = ROOT / "results" / "runs" / "smoke-sweep"
PROMPT_SWEEP_OUT = ROOT / "results" / "runs" / "smoke-prompt-sweep"
PREFIX_SWEEP_OUT = ROOT / "results" / "runs" / "smoke-prefix-sweep"
COMPARISON_OUT = ROOT / "results" / "runs" / "smoke-comparison"
EVAL_OUT = ROOT / "results" / "runs" / "smoke-eval"
TASK_OUT = ROOT / "results" / "runs" / "smoke-task"
QUANT_OUT = ROOT / "results" / "runs" / "smoke-quantization"
SPEC_OUT = ROOT / "results" / "runs" / "smoke-speculative"
VLLM_VALIDATE_OUT = ROOT / "results" / "runs" / "smoke-vllm-validation"
VLLM_PLAN_OUT = ROOT / "results" / "runs" / "smoke-vllm-plan"
EXAMPLES_OUT = ROOT / "results" / "runs" / "smoke-examples"


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing expected smoke artifact: {path}")


def main() -> int:
    output_dirs = [
        LATENCY_OUT,
        THROUGHPUT_OUT,
        SWEEP_OUT,
        PROMPT_SWEEP_OUT,
        PREFIX_SWEEP_OUT,
        COMPARISON_OUT,
        EVAL_OUT,
        TASK_OUT,
        QUANT_OUT,
        SPEC_OUT,
        VLLM_VALIDATE_OUT,
        VLLM_PLAN_OUT,
        EXAMPLES_OUT,
    ]
    for output_dir in output_dirs:
        if output_dir.exists():
            shutil.rmtree(output_dir)

    run([sys.executable, "-m", "llm_accel.cli", "doctor"])
    run([sys.executable, "-m", "llm_accel.cli", "examples", "list"])
    run([sys.executable, "-m", "llm_accel.cli", "examples", "write", "--output-dir", str(EXAMPLES_OUT)])
    require_file(EXAMPLES_OUT / "benchmark_small.yaml")
    require_file(EXAMPLES_OUT / "spec_prompts.jsonl")
    run([sys.executable, "-m", "llm_accel.cli", "backend", "list"])
    run([sys.executable, "-m", "llm_accel.cli", "backend", "profile", "--backend", "vllm", "--base-url", "mock://local"])
    run([sys.executable, "-m", "llm_accel.cli", "kv-cache", "presets"])
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "kv-cache",
            "estimate",
            "--preset",
            "llama-3-8b",
            "--seq-len",
            "128",
            "--batch-size",
            "2",
            "--dtype",
            "fp16",
            "--json",
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "bench",
            "latency",
            "--base-url",
            "mock://local",
            "--model",
            "mock-model",
            "--request-count",
            "2",
            "--output-dir",
            str(LATENCY_OUT),
        ]
    )
    run([sys.executable, "-m", "llm_accel.cli", "report", "validate", "--run-dir", str(LATENCY_OUT)])
    run([sys.executable, "-m", "llm_accel.cli", "report", "generate", "--run-dir", str(LATENCY_OUT)])
    summary = json.loads((LATENCY_OUT / "summary.json").read_text(encoding="utf-8"))
    if summary["metrics"]["failed_count"] != 0:
        raise SystemExit("smoke benchmark has failed requests")
    require_file(LATENCY_OUT / "plots" / "latency.svg")

    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "bench",
            "throughput",
            "--base-url",
            "mock://local",
            "--model",
            "mock-model",
            "--request-count",
            "2",
            "--output-dir",
            str(THROUGHPUT_OUT),
        ]
    )
    run([sys.executable, "-m", "llm_accel.cli", "report", "validate", "--run-dir", str(THROUGHPUT_OUT)])
    throughput = json.loads((THROUGHPUT_OUT / "throughput_summary.json").read_text(encoding="utf-8"))
    if throughput["failed_count"] != 0:
        raise SystemExit("smoke throughput benchmark has failed requests")
    require_file(THROUGHPUT_OUT / "summary.json")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "bench",
            "sweep",
            "--config",
            "configs/benchmark_small.yaml",
            "--output-dir",
            str(SWEEP_OUT),
        ]
    )
    run([sys.executable, "-m", "llm_accel.cli", "report", "validate", "--run-dir", str(SWEEP_OUT)])
    require_file(SWEEP_OUT / "aggregate_summary.json")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "bench",
            "sweep",
            "--config",
            "configs/benchmark_prompts.yaml",
            "--output-dir",
            str(PROMPT_SWEEP_OUT),
        ]
    )
    run([sys.executable, "-m", "llm_accel.cli", "report", "validate", "--run-dir", str(PROMPT_SWEEP_OUT)])
    require_file(PROMPT_SWEEP_OUT / "c1-prompts-out64" / "summary.json")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "bench",
            "sweep",
            "--config",
            "configs/benchmark_prefix_cache.yaml",
            "--output-dir",
            str(PREFIX_SWEEP_OUT),
        ]
    )
    run([sys.executable, "-m", "llm_accel.cli", "report", "validate", "--run-dir", str(PREFIX_SWEEP_OUT)])
    prefix_summary = json.loads((PREFIX_SWEEP_OUT / "c1-prompts-out64" / "summary.json").read_text(encoding="utf-8"))
    if prefix_summary["metadata"]["shared_prefix_tokens_estimate"] <= 0:
        raise SystemExit("prefix-cache smoke benchmark did not record shared prefix metadata")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "report",
            "compare",
            "--summary",
            str(LATENCY_OUT / "summary.json"),
            "--summary",
            str(THROUGHPUT_OUT / "summary.json"),
            "--output-dir",
            str(COMPARISON_OUT),
        ]
    )
    require_file(COMPARISON_OUT / "comparison.json")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "eval",
            "sanity",
            "--base-url",
            "mock://local",
            "--model",
            "mock-model",
            "--prompts",
            "configs/spec_prompts.jsonl",
            "--output-dir",
            str(EVAL_OUT),
        ]
    )
    require_file(EVAL_OUT / "quality_eval.json")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "eval",
            "task",
            "--base-url",
            "mock://local",
            "--model",
            "mock-model",
            "--tasks",
            "configs/task_eval_small.jsonl",
            "--output-dir",
            str(TASK_OUT),
        ]
    )
    require_file(TASK_OUT / "task_eval.json")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "quantization",
            "compare",
            "--base-url",
            "mock://local",
            "--model",
            "mock-model",
            "--modes",
            "none,int8,fp8",
            "--request-count",
            "1",
            "--output-dir",
            str(QUANT_OUT),
        ]
    )
    require_file(QUANT_OUT / "quantization_comparison.json")
    run([sys.executable, "-m", "llm_accel.cli", "speculative", "run", "--lookahead", "4", "--output-dir", str(SPEC_OUT)])
    require_file(SPEC_OUT / "baseline_comparison.json")
    run([sys.executable, "-m", "llm_accel.cli", "vllm", "command", "--model", "mock-model", "--dtype", "auto"])
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "vllm",
            "validate",
            "--model",
            "mock-model",
            "--base-url",
            "mock://local",
            "--output-dir",
            str(VLLM_VALIDATE_OUT),
        ]
    )
    require_file(VLLM_VALIDATE_OUT / "vllm_validation.json")
    run(
        [
            sys.executable,
            "-m",
            "llm_accel.cli",
            "vllm",
            "plan",
            "--model",
            "mock-model",
            "--base-url",
            "http://localhost:8000/v1",
            "--output-dir",
            str(VLLM_PLAN_OUT),
        ]
    )
    require_file(VLLM_PLAN_OUT / "vllm_benchmark_plan.json")
    print("smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
