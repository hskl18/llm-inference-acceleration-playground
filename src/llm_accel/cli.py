from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_accel import __version__
from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.benchmarks.matrix import run_matrix
from llm_accel.benchmarks.sweep import run_sweep
from llm_accel.benchmarks.throughput import run_throughput_benchmark
from llm_accel.config.loader import ConfigError
from llm_accel.evaluation.quality import evaluate_prompts
from llm_accel.evaluation.tasks import evaluate_tasks, load_task_specs
from llm_accel.examples import list_example_files, write_example_files
from llm_accel.kv_cache.estimator import estimate_kv_cache, estimate_kv_cache_from_preset
from llm_accel.kv_cache.presets import list_kv_cache_presets
from llm_accel.metrics.memory import sample_gpu_memory
from llm_accel.quantization.comparison import compare_quantization_modes
from llm_accel.reports.comparison import compare_run_summaries
from llm_accel.reports.claim_audit import audit_hardware_claim
from llm_accel.reports.markdown import write_summary_markdown
from llm_accel.reports.regenerate import regenerate_run_report
from llm_accel.reports.ranking_audit import audit_performance_ranking
from llm_accel.reports.validation import validate_run_dir
from llm_accel.serving.capabilities import get_capability, list_capabilities
from llm_accel.serving.health import check_endpoint_health
from llm_accel.serving.profiles import backend_profile
from llm_accel.serving.vllm import build_vllm_command
from llm_accel.serving.vllm_plan import create_vllm_benchmark_plan
from llm_accel.serving.vllm_validation import validate_vllm_environment
from llm_accel.speculative_decoding.analysis import acceptance_curve, write_speculative_reports
from llm_accel.speculative_decoding.vanilla import run_toy_speculative
from llm_accel.workloads.prompts import load_prompt_file


def _add_bench_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="mock://local")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--dtype", default="unknown")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--backend", default="openai-compatible")
    parser.add_argument("--output-dir")
    parser.add_argument("--hardware-label", default="local")
    parser.add_argument("--api-kind", choices=["chat", "completion"], default="chat")
    parser.add_argument("--prompts", default="", help="Optional plain-text or JSONL prompt file for fixed-prompt benchmarks")
    parser.add_argument("--no-stream", action="store_true", help="Use non-streaming endpoint calls")
    parser.add_argument("--model-revision", help="Exact model revision or immutable artifact identifier")
    parser.add_argument("--tokenizer", help="Exact tokenizer name or artifact identifier")
    parser.add_argument("--tokenizer-revision", help="Exact immutable tokenizer revision")
    parser.add_argument("--optimization-profile", default="baseline", help="Named server configuration under test")
    parser.add_argument("--server-command-sha256", help="SHA-256 of the exact serving command record")
    parser.add_argument("--server-command-file", help="Exact serving command record to copy into the run")
    parser.add_argument(
        "--request-schedule",
        choices=["closed-loop", "open-loop"],
        default="closed-loop",
        help="Closed-loop concurrency or open-loop scheduled arrivals",
    )
    parser.add_argument("--request-rate-rps", type=float, help="Target arrivals per second for open-loop scheduling")
    parser.add_argument("--client-processes", type=int, default=1, help="Load-generator process count")
    parser.add_argument(
        "--queue-delay-warning-ms",
        type=float,
        default=10.0,
        help="Warn when client queue delay p95 exceeds this threshold",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-accel", description="LLM inference acceleration benchmark toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local environment and endpoint configuration")
    doctor.add_argument("--backend", default="mock")
    doctor.add_argument("--base-url", default="mock://local")
    doctor.add_argument("--timeout-seconds", type=float, default=5.0)
    doctor.set_defaults(func=cmd_doctor)

    bench = subparsers.add_parser("bench", help="Run benchmarks")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    latency = bench_sub.add_parser("latency", help="Run a latency benchmark")
    _add_bench_common(latency)
    latency.set_defaults(func=cmd_bench_latency)

    throughput = bench_sub.add_parser("throughput", help="Run a throughput benchmark")
    _add_bench_common(throughput)
    throughput.set_defaults(func=cmd_bench_throughput)

    sweep = bench_sub.add_parser("sweep", help="Run a config-defined benchmark sweep")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--output-dir")
    sweep.set_defaults(func=cmd_bench_sweep)
    matrix = bench_sub.add_parser("matrix", help="Run a randomized, resumable optimization matrix")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--output-dir")
    matrix.add_argument("--resume", action="store_true")
    matrix.set_defaults(func=cmd_bench_matrix)

    report = subparsers.add_parser("report", help="Generate reports from existing summaries")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    generate = report_sub.add_parser("generate", help="Regenerate reports from existing run artifacts")
    generate.add_argument("--run-dir", help="Existing benchmark run directory with summary.json and raw_requests.jsonl")
    generate.add_argument("--summary", help="Input summary.json for direct Markdown regeneration")
    generate.add_argument("--output", help="Output Markdown path for --summary mode")
    generate.set_defaults(func=cmd_report_generate)
    validate = report_sub.add_parser("validate", help="Validate a generated run directory")
    validate.add_argument("--run-dir", required=True)
    validate.set_defaults(func=cmd_report_validate)
    compare_report = report_sub.add_parser("compare", help="Compare two or more summary.json files")
    compare_report.add_argument("--summary", action="append", required=True)
    compare_report.add_argument("--output-dir", required=True)
    compare_report.add_argument("--baseline-profile", default="baseline")
    compare_report.add_argument("--mode", choices=["strict", "stratified"], default="strict")
    compare_report.set_defaults(func=cmd_report_compare)
    claim_audit = report_sub.add_parser("claim-audit", help="Audit whether one run can support a hardware claim")
    claim_audit.add_argument("--run-dir", required=True)
    claim_audit.set_defaults(func=cmd_report_claim_audit)
    ranking_audit = report_sub.add_parser(
        "ranking-audit",
        help="Audit whether a matrix can support a performance ranking",
    )
    ranking_audit.add_argument("--matrix-dir", required=True)
    ranking_audit.set_defaults(func=cmd_report_ranking_audit)

    examples = subparsers.add_parser("examples", help="List or write packaged example configs")
    examples_sub = examples.add_subparsers(dest="examples_command", required=True)
    examples_list = examples_sub.add_parser("list", help="List packaged example files")
    examples_list.set_defaults(func=cmd_examples_list)
    examples_write = examples_sub.add_parser("write", help="Write packaged example files to a directory")
    examples_write.add_argument("--output-dir", required=True)
    examples_write.add_argument("--overwrite", action="store_true")
    examples_write.set_defaults(func=cmd_examples_write)

    kv_cache = subparsers.add_parser("kv-cache", help="KV cache utilities")
    kv_sub = kv_cache.add_subparsers(dest="kv_command", required=True)
    estimate = kv_sub.add_parser("estimate", help="Estimate KV cache memory")
    estimate.add_argument("--preset", help="Model-shape preset, for example llama-3-8b")
    estimate.add_argument("--layers", type=int)
    estimate.add_argument("--seq-len", type=int, required=True)
    estimate.add_argument("--batch-size", type=int, required=True)
    estimate.add_argument("--kv-heads", type=int)
    estimate.add_argument("--head-dim", type=int)
    estimate.add_argument("--dtype", required=True)
    estimate.add_argument("--json", action="store_true")
    estimate.set_defaults(func=cmd_kv_cache_estimate)
    presets = kv_sub.add_parser("presets", help="List built-in model-shape presets")
    presets.set_defaults(func=cmd_kv_cache_presets)

    backend = subparsers.add_parser("backend", help="Inspect backend capability metadata")
    backend_sub = backend.add_subparsers(dest="backend_command", required=True)
    backend_list = backend_sub.add_parser("list", help="List known backend capabilities")
    backend_list.set_defaults(func=cmd_backend_list)
    backend_show = backend_sub.add_parser("show", help="Show one backend capability entry")
    backend_show.add_argument("--backend", required=True)
    backend_show.set_defaults(func=cmd_backend_show)
    backend_profile_parser = backend_sub.add_parser("profile", help="Show backend adapter profile")
    backend_profile_parser.add_argument("--backend", required=True)
    backend_profile_parser.add_argument("--base-url", default="mock://local")
    backend_profile_parser.set_defaults(func=cmd_backend_profile)

    vllm = subparsers.add_parser("vllm", help="vLLM helper commands")
    vllm_sub = vllm.add_subparsers(dest="vllm_command", required=True)
    vllm_command = vllm_sub.add_parser("command", help="Print a vLLM OpenAI server command")
    vllm_command.add_argument("--model", required=True)
    vllm_command.add_argument("--host", default="0.0.0.0")
    vllm_command.add_argument("--port", type=int, default=8000)
    vllm_command.add_argument("--dtype", default="auto")
    vllm_command.add_argument("--revision")
    vllm_command.add_argument("--tokenizer")
    vllm_command.add_argument("--tokenizer-revision")
    vllm_command.add_argument("--quantization")
    vllm_command.add_argument("--max-model-len", type=int)
    vllm_command.add_argument("--gpu-memory-utilization", type=float)
    _add_vllm_optimization_args(vllm_command)
    vllm_command.add_argument("--json", action="store_true")
    vllm_command.set_defaults(func=cmd_vllm_command)
    vllm_validate = vllm_sub.add_parser("validate", help="Validate local readiness for vLLM benchmarking")
    vllm_validate.add_argument("--model", required=True)
    vllm_validate.add_argument("--base-url", default="http://localhost:8000/v1")
    vllm_validate.add_argument("--host", default="0.0.0.0")
    vllm_validate.add_argument("--port", type=int, default=8000)
    vllm_validate.add_argument("--dtype", default="auto")
    vllm_validate.add_argument("--revision")
    vllm_validate.add_argument("--tokenizer")
    vllm_validate.add_argument("--tokenizer-revision")
    vllm_validate.add_argument("--quantization")
    vllm_validate.add_argument("--max-model-len", type=int)
    vllm_validate.add_argument("--gpu-memory-utilization", type=float)
    _add_vllm_optimization_args(vllm_validate)
    vllm_validate.add_argument("--timeout-seconds", type=float, default=5.0)
    vllm_validate.add_argument("--smoke", action="store_true")
    vllm_validate.add_argument("--output-dir", default="results/runs/vllm-validation")
    vllm_validate.set_defaults(func=cmd_vllm_validate)
    vllm_plan = vllm_sub.add_parser("plan", help="Write a vLLM hardware benchmark runbook")
    vllm_plan.add_argument("--model", required=True)
    vllm_plan.add_argument("--base-url", default="http://localhost:8000/v1")
    vllm_plan.add_argument("--config", default="configs/benchmark_vllm_small.yaml")
    vllm_plan.add_argument("--host", default="0.0.0.0")
    vllm_plan.add_argument("--port", type=int, default=8000)
    vllm_plan.add_argument("--dtype", required=True)
    vllm_plan.add_argument("--revision", required=True)
    vllm_plan.add_argument("--tokenizer")
    vllm_plan.add_argument("--tokenizer-revision")
    vllm_plan.add_argument("--hardware-label", required=True)
    vllm_plan.add_argument("--quantization")
    vllm_plan.add_argument("--max-model-len", type=int)
    vllm_plan.add_argument("--gpu-memory-utilization", type=float)
    _add_vllm_optimization_args(vllm_plan)
    vllm_plan.add_argument("--output-dir", default="results/runs/vllm-plan")
    vllm_plan.set_defaults(func=cmd_vllm_plan)

    quant = subparsers.add_parser("quantization", help="Quantization comparison helpers")
    quant_sub = quant.add_subparsers(dest="quant_command", required=True)
    compare = quant_sub.add_parser("compare", help="Compare benchmark metadata across quantization modes")
    compare.add_argument("--base-url", default="mock://local")
    compare.add_argument("--model", default="mock-model")
    compare.add_argument("--modes", required=True, help="Comma-separated modes, for example none,int8,int4")
    compare.add_argument("--concurrency", type=int, default=1)
    compare.add_argument("--input-tokens", type=int, default=128)
    compare.add_argument("--output-tokens", type=int, default=64)
    compare.add_argument("--request-count", type=int, default=8)
    compare.add_argument("--backend", default="mock")
    compare.add_argument("--output-dir", default="results/runs/quantization-comparison")
    compare.add_argument("--hardware-label", default="local")
    compare.add_argument("--sanity-prompts", default="", help="Optional plain-text or JSONL prompt file")
    compare.set_defaults(func=cmd_quantization_compare)

    eval_parser = subparsers.add_parser("eval", help="Run lightweight output quality checks")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    sanity = eval_sub.add_parser("sanity", help="Run fixed-prompt sanity evaluation")
    sanity.add_argument("--base-url", default="mock://local")
    sanity.add_argument("--model", default="mock-model")
    sanity.add_argument("--backend", default="mock")
    sanity.add_argument("--prompts", default="")
    sanity.add_argument("--output-dir", default="results/runs/quality-sanity")
    sanity.add_argument("--max-tokens", type=int, default=64)
    sanity.add_argument("--stream", action="store_true")
    sanity.set_defaults(func=cmd_eval_sanity)
    task = eval_sub.add_parser("task", help="Run validator-based task evaluation")
    task.add_argument("--base-url", default="mock://local")
    task.add_argument("--model", default="mock-model")
    task.add_argument("--backend", default="mock")
    task.add_argument("--tokenizer")
    task.add_argument("--tokenizer-revision")
    task.add_argument("--tasks", required=True)
    task.add_argument("--output-dir", default="results/runs/task-eval")
    task.add_argument("--max-tokens", type=int, default=64)
    task.add_argument("--stream", action="store_true")
    task.set_defaults(func=cmd_eval_task)

    speculative = subparsers.add_parser("speculative", help="Speculative decoding experiments")
    spec_sub = speculative.add_subparsers(dest="spec_command", required=True)
    spec_run = spec_sub.add_parser("run", help="Run toy speculative decoding accounting")
    spec_run.add_argument("--lookahead", type=int, default=4)
    spec_run.add_argument("--prompts", default="")
    spec_run.add_argument("--output-dir", default="results/runs/speculative-toy")
    spec_run.add_argument("--draft-model", default="mock-draft")
    spec_run.add_argument("--target-model", default="mock-target")
    spec_run.add_argument("--acceptance-mod", type=int, default=3)
    spec_run.set_defaults(func=cmd_speculative_run)

    return parser


def _add_vllm_optimization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--speculative-model")
    parser.add_argument("--num-speculative-tokens", type=int)


def cmd_doctor(args: argparse.Namespace) -> int:
    capability = get_capability(args.backend)
    memory = sample_gpu_memory()
    endpoint = check_endpoint_health(args.base_url, timeout_seconds=args.timeout_seconds)
    payload = {
        "project_version": __version__,
        "python": sys.version.split()[0],
        "gpu_required_for_tests": False,
        "mock_endpoint_available": True,
        "backend": args.backend,
        "backend_capability": capability,
        "endpoint_health": endpoint,
        "gpu_memory": memory.to_dict(),
        "status": "ok" if endpoint["healthy"] else "degraded",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_bench_latency(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or _default_run_dir(args.bench_command)
    prompt_texts = load_prompt_file(args.prompts) if args.prompts else None
    summary = run_latency_benchmark(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        output_dir=output_dir,
        request_count=args.request_count,
        warmup_count=args.warmup_count,
        timeout_seconds=args.timeout_seconds,
        dtype=args.dtype,
        quantization=args.quantization,
        backend=args.backend,
        stream=not args.no_stream,
        hardware_label=args.hardware_label,
        api_kind=args.api_kind,
        prompt_texts=prompt_texts,
        model_revision=args.model_revision,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        optimization_profile=args.optimization_profile,
        server_command_sha256=args.server_command_sha256,
        server_command_file=args.server_command_file,
        request_schedule=args.request_schedule,
        request_rate_rps=args.request_rate_rps,
        client_processes=args.client_processes,
        queue_delay_warning_ms=args.queue_delay_warning_ms,
    )
    print(json.dumps({"output_dir": output_dir, "metrics": summary["metrics"]}, indent=2, sort_keys=True))
    return 0


def cmd_bench_throughput(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or _default_run_dir(args.bench_command)
    prompt_texts = load_prompt_file(args.prompts) if args.prompts else None
    summary = run_throughput_benchmark(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        output_dir=output_dir,
        request_count=args.request_count,
        warmup_count=args.warmup_count,
        timeout_seconds=args.timeout_seconds,
        dtype=args.dtype,
        quantization=args.quantization,
        backend=args.backend,
        stream=not args.no_stream,
        hardware_label=args.hardware_label,
        api_kind=args.api_kind,
        prompt_texts=prompt_texts,
        model_revision=args.model_revision,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        optimization_profile=args.optimization_profile,
        server_command_sha256=args.server_command_sha256,
        server_command_file=args.server_command_file,
        request_schedule=args.request_schedule,
        request_rate_rps=args.request_rate_rps,
        client_processes=args.client_processes,
        queue_delay_warning_ms=args.queue_delay_warning_ms,
    )
    print(json.dumps({"output_dir": output_dir, "throughput": summary["metrics"]["throughput"]}, indent=2, sort_keys=True))
    return 0


def _default_run_dir(run_type: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return str(Path("results") / "runs" / f"{timestamp}-{run_type}")


def cmd_bench_sweep(args: argparse.Namespace) -> int:
    aggregate = run_sweep(args.config, args.output_dir)
    print(json.dumps({"output_dir": aggregate["output_dir"], "run_count": aggregate["run_count"]}, indent=2, sort_keys=True))
    return 0


def cmd_bench_matrix(args: argparse.Namespace) -> int:
    result = run_matrix(args.config, args.output_dir, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["execution_failed_run_count"] == 0 and result["evidence_failed_run_count"] == 0 else 1


def cmd_report_generate(args: argparse.Namespace) -> int:
    if args.run_dir:
        result = regenerate_run_report(args.run_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not args.summary or not args.output:
        raise ValueError("report generate requires either --run-dir or both --summary and --output")
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    write_summary_markdown(Path(args.output), summary)
    print(json.dumps({"output": args.output}, indent=2, sort_keys=True))
    return 0


def cmd_report_validate(args: argparse.Namespace) -> int:
    result = validate_run_dir(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def cmd_report_compare(args: argparse.Namespace) -> int:
    result = compare_run_summaries(
        args.summary,
        args.output_dir,
        baseline_profile=args.baseline_profile,
        comparison_mode=args.mode,
    )
    print(json.dumps({"output_dir": args.output_dir, "summary_count": result["summary_count"]}, indent=2, sort_keys=True))
    return 0


def cmd_report_claim_audit(args: argparse.Namespace) -> int:
    result = audit_hardware_claim(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["publishable_hardware_claim"] else 1


def cmd_report_ranking_audit(args: argparse.Namespace) -> int:
    result = audit_performance_ranking(args.matrix_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["publishable_performance_ranking"] else 1


def cmd_examples_list(args: argparse.Namespace) -> int:
    print(json.dumps({"examples": list_example_files()}, indent=2, sort_keys=True))
    return 0


def cmd_examples_write(args: argparse.Namespace) -> int:
    try:
        written = write_example_files(args.output_dir, overwrite=args.overwrite)
    except FileExistsError as exc:
        raise ValueError(str(exc)) from exc
    print(json.dumps({"output_dir": args.output_dir, "files": written}, indent=2, sort_keys=True))
    return 0


def cmd_kv_cache_estimate(args: argparse.Namespace) -> int:
    if args.preset:
        estimate = estimate_kv_cache_from_preset(
            preset=args.preset,
            sequence_length=args.seq_len,
            batch_size=args.batch_size,
            dtype=args.dtype,
            layers=args.layers,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
        )
    else:
        missing = [
            name
            for name, value in {
                "--layers": args.layers,
                "--kv-heads": args.kv_heads,
                "--head-dim": args.head_dim,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"{', '.join(missing)} required unless --preset is provided")
        estimate = estimate_kv_cache(
            layers=args.layers,
            sequence_length=args.seq_len,
            batch_size=args.batch_size,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=args.dtype,
        )
    payload = estimate.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"KV cache: {payload['bytes']} bytes ({payload['mib']:.2f} MiB, {payload['gib']:.2f} GiB)")
    return 0


def cmd_kv_cache_presets(args: argparse.Namespace) -> int:
    print(json.dumps({"presets": list_kv_cache_presets()}, indent=2, sort_keys=True))
    return 0


def cmd_backend_list(args: argparse.Namespace) -> int:
    print(json.dumps(list_capabilities(), indent=2, sort_keys=True))
    return 0


def cmd_backend_show(args: argparse.Namespace) -> int:
    print(json.dumps({args.backend: get_capability(args.backend)}, indent=2, sort_keys=True))
    return 0


def cmd_backend_profile(args: argparse.Namespace) -> int:
    print(json.dumps(backend_profile(args.backend, base_url=args.base_url), indent=2, sort_keys=True))
    return 0


def cmd_vllm_command(args: argparse.Namespace) -> int:
    command = build_vllm_command(
        model=args.model,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        revision=args.revision,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
    )
    if args.json:
        print(json.dumps(command.to_dict(), indent=2, sort_keys=True))
    else:
        print(command.shell_command())
    return 0


def cmd_vllm_validate(args: argparse.Namespace) -> int:
    report = validate_vllm_environment(
        model=args.model,
        base_url=args.base_url,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        revision=args.revision,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        timeout_seconds=args.timeout_seconds,
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "ready_for_hardware_benchmark": report["ready_for_hardware_benchmark"],
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["ready_for_hardware_benchmark"] else 1


def cmd_vllm_plan(args: argparse.Namespace) -> int:
    plan = create_vllm_benchmark_plan(
        model=args.model,
        base_url=args.base_url,
        output_dir=args.output_dir,
        config_path=args.config,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        revision=args.revision,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        hardware_label=args.hardware_label,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
    )
    print(json.dumps({"output_dir": args.output_dir, "steps": len(plan["steps"])}, indent=2, sort_keys=True))
    return 0


def cmd_quantization_compare(args: argparse.Namespace) -> int:
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    sanity_prompts = load_prompt_file(args.sanity_prompts) if args.sanity_prompts else None
    report = compare_quantization_modes(
        base_url=args.base_url,
        model=args.model,
        modes=modes,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        request_count=args.request_count,
        backend=args.backend,
        hardware_label=args.hardware_label,
        sanity_prompts=sanity_prompts,
    )
    print(json.dumps({"output_dir": args.output_dir, "modes": report["modes"]}, indent=2, sort_keys=True))
    return 0


def cmd_eval_sanity(args: argparse.Namespace) -> int:
    prompts = load_prompt_file(args.prompts) if args.prompts else ["Explain KV cache in one sentence."]
    report = evaluate_prompts(
        base_url=args.base_url,
        model=args.model,
        backend=args.backend,
        prompts=prompts,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        stream=args.stream,
    )
    print(json.dumps({"output_dir": args.output_dir, "passed": report["passed"]}, indent=2, sort_keys=True))
    return 0


def cmd_eval_task(args: argparse.Namespace) -> int:
    report = evaluate_tasks(
        base_url=args.base_url,
        model=args.model,
        backend=args.backend,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        task_specs=load_task_specs(args.tasks),
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        stream=args.stream,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "passed": report["passed"],
                "mean_score": report["mean_score"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_speculative_run(args: argparse.Namespace) -> int:
    prompts = ["synthetic prompt"]
    if args.prompts:
        path = Path(args.prompts)
        if path.exists():
            prompts = load_prompt_file(path)
    result = run_toy_speculative(prompts, lookahead=args.lookahead, acceptance_mod=args.acceptance_mod)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "result": result.to_dict(),
        "acceptance_curve": acceptance_curve(prompts=prompts, lookahead=args.lookahead),
    }
    write_speculative_reports(output_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError) as exc:
        print(f"llm-accel: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
