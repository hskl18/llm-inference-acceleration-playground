from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from pathlib import Path

from llm_accel.metrics.aggregation import summarize_requests
from llm_accel.metrics.environment import environment_fingerprint
from llm_accel.metrics.optimization_profile import (
    OptimizationProfile,
    load_bound_optimization_profile,
)
from llm_accel.metrics.schemas import SCHEMA_VERSION, RequestMetrics
from llm_accel.metrics.token_counting import is_local_tokenizer_reference
from llm_accel.reports.validation import validate_run_dir
from llm_accel.serving.vllm import normalize_vllm_dtype, optimization_profile_name


MIN_MEASURED_REQUESTS = 100
MIN_WARMUP_REQUESTS = 5


def audit_hardware_claim(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir)
    try:
        validation = validate_run_dir(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _report(path, [f"artifact validation could not read the run: {exc}"], [], {})
    blockers = [f"artifact validation: {error}" for error in validation["errors"]]
    warnings = list(validation["warnings"])
    summary_path = path / "summary.json"
    raw_path = path / "raw_requests.jsonl"

    if not summary_path.exists():
        blockers.append("summary.json is required")
        return _report(path, blockers, warnings, {})

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"summary.json could not be read: {exc}")
        return _report(path, blockers, warnings, {})
    metadata = summary.get("metadata", {})
    metrics = summary.get("metrics", {})
    memory = summary.get("memory", {})
    if not isinstance(metadata, dict) or not isinstance(metrics, dict) or not isinstance(memory, dict):
        blockers.append("summary metadata, metrics, and memory must be objects")
        return _report(path, blockers, warnings, {})
    if summary.get("schema_version") != SCHEMA_VERSION:
        blockers.append(
            f"summary schema {summary.get('schema_version')!r} is unsupported; expected {SCHEMA_VERSION!r}"
        )

    required_metadata = {
        "backend_version": "backend version",
        "model_revision": "exact model revision",
        "gpu_name": "GPU name",
        "gpu_driver_version": "GPU driver version",
        "cuda_version": "CUDA version",
        "torch_version": "PyTorch version",
        "git_commit": "benchmark code commit",
        "server_command_sha256": "exact server command fingerprint",
        "tokenizer_revision": "exact tokenizer revision",
        "environment_fingerprint": "environment fingerprint",
        "endpoint_sha256": "endpoint fingerprint",
        "token_count_method": "token count method",
        "optimization_profile_fingerprint": "optimization profile fingerprint",
    }
    for key, label in required_metadata.items():
        if metadata.get(key) in {None, "", "unknown"}:
            blockers.append(f"missing {label} ({key})")
    if metadata.get("environment_fingerprint") != environment_fingerprint(metadata):
        blockers.append("environment_fingerprint does not match canonical environment metadata")

    git_commit = str(metadata.get("git_commit") or "")
    if git_commit and not re.fullmatch(r"[0-9a-f]{40,64}", git_commit):
        blockers.append("git_commit must be a full hexadecimal commit identifier")
    model_revision = str(metadata.get("model_revision") or "")
    if model_revision and not re.fullmatch(r"[0-9a-f]{40,64}", model_revision):
        blockers.append("model_revision must be a full immutable hexadecimal revision")
    tokenizer_revision = str(metadata.get("tokenizer_revision") or "")
    if tokenizer_revision and not re.fullmatch(r"[0-9a-f]{40,64}", tokenizer_revision):
        blockers.append("tokenizer_revision must be a full immutable hexadecimal revision")
    tokenizer = str(metadata.get("tokenizer") or "")
    if tokenizer and is_local_tokenizer_reference(tokenizer):
        blockers.append("local tokenizer paths are mutable and cannot support hardware claims")
    server_command_sha256 = str(metadata.get("server_command_sha256") or "")
    if server_command_sha256 and not re.fullmatch(r"[0-9a-f]{64}", server_command_sha256):
        blockers.append("server_command_sha256 must be a lowercase 64-character SHA-256 digest")

    if metadata.get("backend") != "vllm" or str(metadata.get("base_url", "")).startswith("mock://"):
        blockers.append("hardware claims require a vLLM endpoint run, not mock or relabeled output")
    if (
        metadata.get("backend") == "vllm"
        and metadata.get("token_count_method")
        != "prompt=server_usage;output=tokenizers.encode(add_special_tokens=false)"
    ):
        blockers.append(
            "vLLM hardware claims require server prompt usage and resolved-tokenizer output counts"
        )
    if metadata.get("dtype") in {None, "", "unknown", "auto"}:
        blockers.append("an exact dtype must be recorded")
    else:
        try:
            normalized_dtype = normalize_vllm_dtype(str(metadata["dtype"]))
        except ValueError as exc:
            blockers.append(str(exc))
        else:
            if normalized_dtype != metadata["dtype"]:
                blockers.append(f"dtype must use canonical vLLM spelling {normalized_dtype!r}")
    if metadata.get("quantization") in {None, "", "unknown"}:
        blockers.append("quantization mode must be recorded")
    if metadata.get("hardware_label") in {None, "", "local", "unknown"}:
        blockers.append("hardware_label must identify the measured host or hardware class")
    if metadata.get("optimization_profile") in {None, "", "unknown"}:
        blockers.append("optimization_profile must identify the server configuration")

    try:
        request_count = int(metrics.get("request_count", 0))
        completed_count = int(metrics.get("completed_count", 0))
        failed_count = int(metrics.get("failed_count", 0))
        warmup_count = int(metadata.get("warmup_count", 0))
    except (TypeError, ValueError):
        blockers.append("request, completion, failure, and warmup counts must be integers")
        request_count = completed_count = failed_count = warmup_count = 0
    if completed_count + failed_count != request_count:
        blockers.append("completed_count plus failed_count does not match request_count")
    if request_count < MIN_MEASURED_REQUESTS:
        blockers.append(
            f"only {request_count} measured requests; at least {MIN_MEASURED_REQUESTS} are required"
        )
    if warmup_count < MIN_WARMUP_REQUESTS:
        blockers.append(
            f"only {warmup_count} warmup requests; at least {MIN_WARMUP_REQUESTS} are required"
        )
    if not memory.get("available", False):
        blockers.append("GPU memory telemetry is unavailable")
    else:
        _validate_memory(memory, blockers)
        for snapshot_name in ["before", "after"]:
            snapshot = memory.get(snapshot_name)
            if isinstance(snapshot, dict) and snapshot.get("gpu_name") != metadata.get("gpu_name"):
                blockers.append(f"memory.{snapshot_name}.gpu_name does not match summary metadata")

    for metric_group in ["latency_ms", "ttft_ms", "tpot_ms"]:
        values = metrics.get(metric_group)
        if not isinstance(values, dict):
            blockers.append(f"metrics.{metric_group} is missing")
            continue
        for percentile_name in ["p50", "p95", "p99"]:
            if percentile_name not in values:
                blockers.append(f"metrics.{metric_group}.{percentile_name} is missing")
            elif not _finite_non_negative(values[percentile_name]):
                blockers.append(f"metrics.{metric_group}.{percentile_name} must be finite and non-negative")
    throughput = metrics.get("throughput")
    if not isinstance(throughput, dict):
        blockers.append("metrics.throughput is missing")
    else:
        for key in ["output_tokens_per_second", "requests_per_second", "measured_elapsed_seconds"]:
            if throughput.get(key) is None:
                blockers.append(f"metrics.throughput.{key} is missing")
            elif not _finite_positive(throughput[key]):
                blockers.append(f"metrics.throughput.{key} must be finite and positive")

    _validate_server_command(path, metadata, blockers)
    _validate_optimization_profile(path, metadata, blockers)
    _validate_quality_execution_identity(metadata, blockers)

    client_configuration = metadata.get("client_configuration")
    if not isinstance(client_configuration, dict):
        blockers.append("structured client_configuration is required")
    if metadata.get("request_schedule") not in {"closed-loop", "open-loop"}:
        blockers.append("request_schedule must identify closed-loop or open-loop scheduling")
    queue_delay = metrics.get("queue_delay_ms")
    if not isinstance(queue_delay, dict):
        blockers.append("metrics.queue_delay_ms is missing")
    else:
        for percentile_name in ["p50", "p95", "p99", "max"]:
            if not _finite_non_negative(queue_delay.get(percentile_name)):
                blockers.append(f"metrics.queue_delay_ms.{percentile_name} must be finite and non-negative")

    raw_result = _read_raw_requests(raw_path)
    if raw_result is None:
        blockers.append("raw_requests.jsonl is required")
    elif isinstance(raw_result, str):
        blockers.append(raw_result)
    elif len(raw_result) != request_count:
        blockers.append(
            f"raw request count {len(raw_result)} does not match summary request count {request_count}"
        )
    elif len({str(row.get("request_id")) for row in raw_result}) != len(raw_result):
        blockers.append("raw request IDs are not unique")
    else:
        _validate_raw_requests(raw_result, metadata, metrics, blockers)

    if "error_rate" not in metrics:
        blockers.append("metrics.error_rate is missing")
        error_rate = 0.0
    else:
        try:
            error_rate = float(metrics["error_rate"])
        except (TypeError, ValueError):
            blockers.append("metrics.error_rate must be numeric")
            error_rate = 0.0
        expected_error_rate = failed_count / request_count if request_count else 0.0
        if abs(error_rate - expected_error_rate) > 1e-9:
            blockers.append("metrics.error_rate does not match failed_count divided by request_count")
    if error_rate > 0:
        warnings.append("The run contains request errors; publish the error rate and failure details.")
    if error_rate > 0.05:
        blockers.append("error rate exceeds the 5 percent hardware-claim ceiling")
    warnings.append(
        "This audit covers one performance run; publish repeated runs, a compatible comparison, and separate quality evidence."
    )
    warnings.append(
        "Artifact checks cannot independently attest that the endpoint process matched the recorded command; preserve operator or platform launch evidence."
    )
    if metadata.get("request_schedule") == "closed-loop":
        warnings.append(
            "Closed-loop scheduling is susceptible to coordinated omission and is not sufficient for a cross-profile performance ranking."
        )
    evidence = {
        "backend": metadata.get("backend"),
        "model": metadata.get("model"),
        "model_revision": metadata.get("model_revision"),
        "optimization_profile": metadata.get("optimization_profile"),
        "request_count": request_count,
        "warmup_count": warmup_count,
        "error_rate": metrics.get("error_rate"),
        "gpu_name": metadata.get("gpu_name"),
    }
    return _report(path, blockers, warnings, evidence)


def _validate_quality_execution_identity(
    metadata: dict[str, object],
    blockers: list[str],
) -> None:
    quality_gate = metadata.get("quality_gate")
    if quality_gate is None:
        return
    if not isinstance(quality_gate, dict):
        blockers.append("quality_gate must be an object")
        return
    expected = {
        "profile": metadata.get("optimization_profile"),
        "model": metadata.get("model"),
        "backend": metadata.get("backend"),
        "base_url": metadata.get("base_url"),
        "endpoint_sha256": metadata.get("endpoint_sha256"),
    }
    if quality_gate.get("execution_identity") != expected:
        blockers.append("quality evidence execution identity does not match summary metadata")


def _validate_server_command(
    path: Path,
    metadata: dict[str, object],
    blockers: list[str],
) -> None:
    command_path = path / "server_command.txt"
    if not command_path.exists():
        blockers.append("server_command.txt is required in the run directory")
        return
    try:
        command_bytes = command_path.read_bytes()
        argv = shlex.split(command_bytes.decode("utf-8").strip())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        blockers.append(f"server_command.txt could not be parsed: {exc}")
        return
    digest = hashlib.sha256(command_bytes).hexdigest()
    if digest != metadata.get("server_command_sha256"):
        blockers.append("server_command.txt hash does not match summary metadata")
    if argv[:3] != ["python", "-m", "vllm.entrypoints.openai.api_server"]:
        blockers.append("server_command.txt is not the expected vLLM API server command")
    expected = {
        "--model": metadata.get("model"),
        "--revision": metadata.get("model_revision"),
        "--tokenizer": metadata.get("tokenizer"),
        "--tokenizer-revision": metadata.get("tokenizer_revision"),
        "--dtype": metadata.get("dtype"),
    }
    for flag, expected_value in expected.items():
        if expected_value is None and flag in {"--tokenizer", "--tokenizer-revision"}:
            continue
        actual = _flag_value(argv, flag)
        if actual != expected_value:
            blockers.append(f"server command {flag} does not match summary metadata")
    command_quantization = _flag_value(argv, "--quantization") or "none"
    if command_quantization != metadata.get("quantization"):
        blockers.append("server command --quantization does not match summary metadata")
    if not (path / "optimization_profile.json").exists():
        command_profile = optimization_profile_name(
            enable_prefix_caching="--enable-prefix-caching" in argv,
            enable_chunked_prefill="--enable-chunked-prefill" in argv,
            speculative_model=_flag_value(argv, "--speculative-model"),
            quantization=_flag_value(argv, "--quantization"),
        )
        if command_profile != metadata.get("optimization_profile"):
            blockers.append("optimization_profile does not match the recorded server command flags")


def _flag_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def _validate_optimization_profile(
    path: Path,
    metadata: dict[str, object],
    blockers: list[str],
) -> None:
    try:
        profile = load_bound_optimization_profile(
            path,
            metadata.get("optimization_profile_spec"),
            require_artifact=True,
        )
        if profile is None:
            raise ValueError("optimization profile is missing")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        blockers.append(f"optimization_profile.json is required and must be valid: {exc}")
        return
    expected = {
        "model": profile.model,
        "model_revision": profile.model_revision,
        "tokenizer": profile.tokenizer,
        "tokenizer_revision": profile.tokenizer_revision,
        "backend": profile.backend,
        "backend_version": profile.backend_version,
        "dtype": profile.dtype,
        "quantization": profile.quantization,
        "environment_fingerprint": profile.environment_fingerprint,
        "optimization_profile_fingerprint": profile.semantic_fingerprint,
        "server_command_sha256": profile.server_command_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            blockers.append(f"summary metadata {key} does not match optimization_profile.json")
    if metadata.get("optimization_profile") != profile.name:
        blockers.append("summary optimization_profile name does not match optimization_profile.json")
    if profile.backend == "vllm":
        _validate_vllm_profile_flags(profile, blockers)


def _validate_vllm_profile_flags(profile: OptimizationProfile, blockers: list[str]) -> None:
    argv = list(profile.server_command_argv)
    boolean_flags = {
        "--enable-prefix-caching": profile.prefix_cache,
        "--enable-chunked-prefill": profile.chunked_prefill,
    }
    for flag, expected in boolean_flags.items():
        if (flag in argv) != expected:
            blockers.append(f"optimization profile {flag} does not match the exact server command")
    expected_values: dict[str, object] = {
        "--model": profile.model,
        "--revision": profile.model_revision,
        "--tokenizer": profile.tokenizer,
        "--tokenizer-revision": profile.tokenizer_revision,
        "--dtype": profile.dtype,
        "--quantization": None if profile.quantization == "none" else profile.quantization,
        "--speculative-model": profile.speculative_model,
        "--num-speculative-tokens": profile.num_speculative_tokens,
        "--max-num-batched-tokens": profile.max_num_batched_tokens,
        "--max-num-seqs": profile.max_num_seqs,
        "--max-model-len": profile.max_model_len,
        "--gpu-memory-utilization": profile.gpu_memory_utilization,
    }
    for flag, expected in expected_values.items():
        actual = _flag_value(argv, flag)
        expected_text = None if expected is None else str(expected)
        if actual != expected_text:
            blockers.append(f"optimization profile {flag} does not match the exact server command")


def _validate_raw_requests(
    rows: list[dict[str, object]],
    metadata: dict[str, object],
    metrics: dict[str, object],
    blockers: list[str],
) -> None:
    methods = {row.get("token_count_method") for row in rows}
    if methods != {metadata.get("token_count_method")}:
        blockers.append("summary token_count_method does not match raw requests")
    completed = [row for row in rows if row.get("completed") is True]
    failed = [row for row in rows if row.get("completed") is False]
    if len(completed) != metrics.get("completed_count") or len(failed) != metrics.get("failed_count"):
        blockers.append("raw completed and failed counts do not match summary metrics")
    if len(completed) < MIN_MEASURED_REQUESTS:
        blockers.append(f"only {len(completed)} completed requests; at least {MIN_MEASURED_REQUESTS} are required")
    output_tokens = 0
    for row in rows:
        if row.get("model") != metadata.get("model") or row.get("backend") != metadata.get("backend"):
            blockers.append("raw request model or backend does not match summary metadata")
            break
        if row.get("completed") is True:
            if not _finite_positive(row.get("total_latency_ms")):
                blockers.append("completed raw requests must have positive finite latency")
                break
            if not isinstance(row.get("output_tokens"), int) or int(row["output_tokens"]) <= 0:
                blockers.append("completed raw requests must have positive output tokens")
                break
            output_tokens += int(row["output_tokens"])
        if not _finite_non_negative(row.get("started_offset_ms")):
            blockers.append("raw requests must include a finite non-negative started_offset_ms")
            break
        if not _finite_non_negative(row.get("completed_offset_ms")):
            blockers.append("raw requests must include a finite non-negative completed_offset_ms")
            break
        if float(row["completed_offset_ms"]) < float(row["started_offset_ms"]):
            blockers.append("raw request completion offsets must not precede start offsets")
            break
        for key in ["scheduled_offset_ms", "dispatch_offset_ms", "queue_delay_ms", "end_to_end_latency_ms"]:
            if not _finite_non_negative(row.get(key)):
                blockers.append(f"raw requests must include a finite non-negative {key}")
                break
        else:
            if float(row["scheduled_offset_ms"]) > float(row["dispatch_offset_ms"]):
                blockers.append("raw request dispatch offsets must not precede scheduled offsets")
                break
            if float(row["dispatch_offset_ms"]) != float(row["started_offset_ms"]):
                blockers.append("raw request dispatch offsets must match started offsets")
                break
            if not math.isclose(
                float(row["queue_delay_ms"]),
                float(row["dispatch_offset_ms"]) - float(row["scheduled_offset_ms"]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                blockers.append("raw request queue delay does not match dispatch minus schedule")
                break
            dispatch_span = float(row["completed_offset_ms"]) - float(row["dispatch_offset_ms"])
            scheduled_span = float(row["completed_offset_ms"]) - float(row["scheduled_offset_ms"])
            timing_tolerance_ms = max(5.0, dispatch_span * 0.01)
            if not math.isclose(
                float(row["total_latency_ms"]),
                dispatch_span,
                rel_tol=0.01,
                abs_tol=timing_tolerance_ms,
            ):
                blockers.append("raw request total latency does not match completion minus dispatch")
                break
            if not math.isclose(
                float(row["end_to_end_latency_ms"]),
                scheduled_span,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                blockers.append("raw request end-to-end latency does not match completion minus schedule")
                break
    if output_tokens != metrics.get("output_tokens"):
        blockers.append("raw completed output tokens do not match summary metrics")
    _validate_recomputed_metrics(rows, metrics, blockers)


def _validate_recomputed_metrics(
    rows: list[dict[str, object]],
    metrics: dict[str, object],
    blockers: list[str],
) -> None:
    try:
        records = [
            RequestMetrics(
                request_id=str(row["request_id"]),
                model=str(row["model"]),
                backend=str(row["backend"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                concurrency=int(row["concurrency"]),
                ttft_ms=float(row["ttft_ms"]),
                tpot_ms=float(row["tpot_ms"]),
                total_latency_ms=float(row["total_latency_ms"]),
                completed=bool(row["completed"]),
                error=str(row["error"]) if row.get("error") is not None else None,
                started_offset_ms=float(row["started_offset_ms"]),
                completed_offset_ms=float(row["completed_offset_ms"]),
                scheduled_offset_ms=float(row["scheduled_offset_ms"]),
                dispatch_offset_ms=float(row["dispatch_offset_ms"]),
                queue_delay_ms=float(row["queue_delay_ms"]),
                end_to_end_latency_ms=float(row["end_to_end_latency_ms"]),
            )
            for row in rows
        ]
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(f"raw requests cannot reproduce summary metrics: {exc}")
        return
    elapsed_seconds = _raw_span_seconds(records)
    recomputed = summarize_requests(records, elapsed_seconds=elapsed_seconds)
    paths = [
        ("request_count",),
        ("completed_count",),
        ("failed_count",),
        ("timeout_count",),
        ("error_rate",),
        ("output_tokens",),
        ("latency_ms", "mean"),
        ("latency_ms", "p50"),
        ("latency_ms", "p95"),
        ("latency_ms", "p99"),
        ("ttft_ms", "mean"),
        ("ttft_ms", "p50"),
        ("ttft_ms", "p95"),
        ("ttft_ms", "p99"),
        ("tpot_ms", "mean"),
        ("tpot_ms", "p50"),
        ("tpot_ms", "p95"),
        ("tpot_ms", "p99"),
        ("throughput", "output_tokens_per_second"),
        ("throughput", "requests_per_second"),
        ("throughput", "estimated_elapsed_seconds"),
        ("throughput", "measured_elapsed_seconds"),
        ("queue_delay_ms", "mean"),
        ("queue_delay_ms", "p50"),
        ("queue_delay_ms", "p95"),
        ("queue_delay_ms", "p99"),
        ("queue_delay_ms", "max"),
        ("end_to_end_latency_ms", "mean"),
        ("end_to_end_latency_ms", "p50"),
        ("end_to_end_latency_ms", "p95"),
        ("end_to_end_latency_ms", "p99"),
    ]
    for path in paths:
        expected = _nested(recomputed, path)
        actual = _nested(metrics, path)
        if not _numbers_match(actual, expected):
            blockers.append(f"summary metric {'.'.join(path)} does not match raw requests")


def _nested(payload: dict[str, object], path: tuple[str, ...]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _numbers_match(actual: object, expected: object) -> bool:
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        return actual == expected
    return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)


def _raw_span_seconds(records: list[RequestMetrics]) -> float:
    if not records:
        return 0.0
    started = min(record.scheduled_offset_ms for record in records)
    completed = max(record.completed_offset_ms for record in records)
    return max(completed - started, 0.0) / 1000


def _validate_memory(memory: dict[str, object], blockers: list[str]) -> None:
    for snapshot_name in ["before", "after"]:
        snapshot = memory.get(snapshot_name)
        if not isinstance(snapshot, dict):
            blockers.append(f"memory.{snapshot_name} must be an object")
            continue
        if not snapshot.get("gpu_name"):
            blockers.append(f"memory.{snapshot_name}.gpu_name is required")
        for key in ["total_mib", "used_mib", "free_mib"]:
            if not _finite_non_negative(snapshot.get(key)):
                blockers.append(f"memory.{snapshot_name}.{key} must be finite and non-negative")


def _finite_non_negative(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0


def _finite_positive(value: object) -> bool:
    return _finite_non_negative(value) and float(value) > 0


def _read_raw_requests(path: Path) -> list[dict[str, object]] | str | None:
    if not path.exists():
        return None
    rows: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                return f"raw_requests.jsonl line {line_number} is not an object"
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        return f"raw_requests.jsonl could not be read: {exc}"
    return rows


def _report(
    path: Path,
    blockers: list[str],
    warnings: list[str],
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "run_dir": str(path),
        "publishable_hardware_claim": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "evidence": evidence,
    }
