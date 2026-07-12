from __future__ import annotations

import argparse
import hashlib
import shlex
import shutil
from pathlib import Path

from llm_accel.benchmarks.throughput import run_throughput_benchmark
from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.reports.claim_audit import audit_hardware_claim
from llm_accel.reports.comparison import compare_run_summaries
from llm_accel.serving.vllm import (
    normalize_vllm_dtype,
    optimization_profile_name,
    require_immutable_revision,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect repeated runs for one already-running hardware serving profile."
    )
    parser.add_argument("--profile", required=True, help="Stable profile name, such as baseline or prefix-cache")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--hardware-label", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--request-count", type=int, default=128)
    parser.add_argument("--warmup-count", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--server-command-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    require_immutable_revision(args.model_revision)
    dtype = normalize_vllm_dtype(args.dtype)
    if args.repetitions < 3:
        raise ValueError("hardware evidence requires at least three repetitions")
    if args.request_count < 100:
        raise ValueError("hardware evidence requires at least 100 measured requests per repetition")
    if args.warmup_count < 5:
        raise ValueError("hardware evidence requires at least five warmup requests per repetition")

    prompt_texts = _load_prompts(args.prompts) if args.prompts else None
    profile_root = args.output_root / args.profile
    profile_root.mkdir(parents=True, exist_ok=True)
    server_command = args.server_command_file.read_bytes()
    server_command_sha256 = hashlib.sha256(server_command).hexdigest()
    _validate_server_command(
        server_command,
        model=args.model,
        revision=args.model_revision,
        dtype=dtype,
        quantization=args.quantization,
        profile=args.profile,
    )
    profile_command_path = (profile_root / "server_command.txt").resolve()
    if args.server_command_file.resolve() != profile_command_path:
        shutil.copyfile(args.server_command_file, profile_command_path)
    summaries: list[Path] = []
    audits: list[dict[str, object]] = []
    for repetition in range(1, args.repetitions + 1):
        run_dir = profile_root / f"repeat-{repetition:02d}"
        run_throughput_benchmark(
            base_url=args.base_url,
            model=args.model,
            model_revision=args.model_revision,
            concurrency=args.concurrency,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            output_dir=run_dir,
            request_count=args.request_count,
            warmup_count=args.warmup_count,
            timeout_seconds=args.timeout_seconds,
            dtype=dtype,
            quantization=args.quantization,
            backend="vllm",
            stream=True,
            hardware_label=args.hardware_label,
            api_kind="chat",
            prompt_texts=prompt_texts,
            optimization_profile=args.profile,
            server_command_sha256=server_command_sha256,
            server_command_file=args.server_command_file,
        )
        summaries.append(run_dir / "summary.json")
        audits.append(audit_hardware_claim(run_dir))

    comparison = compare_run_summaries(summaries, profile_root / "comparison")
    report = {
        "profile": args.profile,
        "repetitions": args.repetitions,
        "request_count_per_repetition": args.request_count,
        "server_command_sha256": server_command_sha256,
        "all_runs_publishable": all(audit["publishable_hardware_claim"] for audit in audits),
        "audits": audits,
        "comparison": comparison,
        "limitations": [
            "This bundle measures one server profile only.",
            "Cross-profile claims require a separate server restart and comparison of compatible bundles.",
            "Quality evidence must be collected and reported separately.",
        ],
    }
    write_json(profile_root / "profile_evidence.json", report)
    artifacts = [
        "manifest.json",
        "profile_evidence.json",
        "server_command.txt",
        "comparison/manifest.json",
        "comparison/comparison.json",
        "comparison/comparison.md",
    ]
    for repetition in range(1, args.repetitions + 1):
        repeat = f"repeat-{repetition:02d}"
        artifacts.extend(
            [
                f"{repeat}/manifest.json",
                f"{repeat}/summary.json",
                f"{repeat}/raw_requests.jsonl",
                f"{repeat}/server_command.txt",
            ]
        )
    write_run_manifest(profile_root, run_type="hardware_profile_evidence", artifacts=artifacts)
    print(profile_root / "profile_evidence.json")
    return 0 if report["all_runs_publishable"] and comparison["ranking_allowed"] else 1


def _load_prompts(path: Path) -> list[str]:
    from llm_accel.workloads.prompts import load_prompt_file

    return load_prompt_file(path)


def _validate_server_command(
    command: bytes,
    *,
    model: str,
    revision: str,
    dtype: str,
    quantization: str,
    profile: str,
) -> None:
    argv = shlex.split(command.decode("utf-8").strip())
    if argv[:3] != ["python", "-m", "vllm.entrypoints.openai.api_server"]:
        raise ValueError("server command file is not a vLLM API server command")
    expected = {"--model": model, "--revision": revision, "--dtype": dtype}
    for flag, value in expected.items():
        if _flag_value(argv, flag) != value:
            raise ValueError(f"server command {flag} does not match collector arguments")
    if (_flag_value(argv, "--quantization") or "none") != quantization:
        raise ValueError("server command --quantization does not match collector arguments")
    command_profile = optimization_profile_name(
        enable_prefix_caching="--enable-prefix-caching" in argv,
        enable_chunked_prefill="--enable-chunked-prefill" in argv,
        speculative_model=_flag_value(argv, "--speculative-model"),
        quantization=_flag_value(argv, "--quantization"),
    )
    if command_profile != profile:
        raise ValueError(
            f"server command implies optimization profile {command_profile!r}, not {profile!r}"
        )


def _flag_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


if __name__ == "__main__":
    raise SystemExit(main())
