from __future__ import annotations

import hashlib
import json
import math
import random
import re
from itertools import product
from pathlib import Path
from typing import Any

from llm_accel.benchmarks.throughput import run_throughput_benchmark
from llm_accel.config.loader import get_path, load_config, sanitize_resolved_config, validate_benchmark_config
from llm_accel.evaluation.tasks import evaluate_tasks, load_task_specs
from llm_accel.evaluation.validators import validate_output
from llm_accel.metrics.io import write_bytes_atomic, write_json, write_json_atomic
from llm_accel.metrics.environment import environment_fingerprint
from llm_accel.metrics.execution_identity import execution_identity
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.metrics.optimization_profile import (
    create_optimization_profile,
    load_optimization_profile,
    write_optimization_profile,
)
from llm_accel.reports.comparison import compare_run_summaries
from llm_accel.reports.validation import validate_run_dir
from llm_accel.serving.versions import detect_backend_version
from llm_accel.workloads.prompts import load_prompt_file


REQUIRED_PROFILE_ROLES = {
    "baseline",
    "prefix-cache",
    "chunked-prefill",
    "quantized",
    "speculative",
}
MIN_REPETITIONS = 3
PROFILE_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def run_matrix(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    resume: bool = False,
) -> dict[str, object]:
    config_file = Path(config_path)
    config = load_config(config_file)
    validate_benchmark_config(config)
    profiles = _validate_matrix_config(config)
    matrix_name = str(get_path(config, "matrix.name", "optimization-matrix"))
    base_output = Path(
        output_dir or get_path(config, "run.output_dir", f"results/runs/{matrix_name}")
    ).expanduser().resolve()
    base_output.mkdir(parents=True, exist_ok=True)
    config_digest = _config_digest(config, config_file)
    plan_path = base_output / "matrix_plan.json"
    state_path = base_output / "matrix_state.json"

    plan = _build_plan(config, profiles, config_file, base_output, config_digest)
    if resume:
        _validate_resume(plan_path, state_path, config_digest)
        persisted_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if persisted_plan.get("runs") != plan.get("runs"):
            raise ValueError("matrix plan changed; resume requires the original randomized plan")
        plan = persisted_plan
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for entry in state.get("runs", []):
            if entry.get("status") == "running":
                entry["status"] = "pending"
                entry["error"] = "previous execution stopped while this run was active"
    else:
        if plan_path.exists() or state_path.exists():
            raise ValueError("matrix output already contains state; pass resume=True to continue it")
        state = _initial_state(plan)
        write_json_atomic(plan_path, plan)
    write_json_atomic(state_path, state)
    write_json(base_output / "resolved_config.json", sanitize_resolved_config(config))
    quality_by_profile = _run_quality_gates(config, config_file, base_output, profiles, resume=resume)

    state_by_id = {str(entry["run_id"]): entry for entry in state["runs"]}
    summary_paths: list[Path] = []
    skipped_on_resume = 0
    for planned in plan["runs"]:
        run_id = str(planned["run_id"])
        entry = state_by_id[run_id]
        run_dir = _contained_path(base_output, run_id)
        if resume and entry.get("status") == "succeeded" and _valid_existing_run(
            run_dir,
            planned,
            matrix_name=matrix_name,
            quality=quality_by_profile.get(str(planned["profile"])),
        ):
            skipped_on_resume += 1
            summary_paths.append(run_dir / "summary.json")
            continue
        entry.update(
            {
                "status": "running",
                "attempts": int(entry.get("attempts", 0)) + 1,
                "error": None,
            }
        )
        write_json_atomic(state_path, state)
        try:
            summary = _run_planned_cell(
                config,
                config_file,
                base_output,
                planned,
                quality_by_profile.get(str(planned["profile"])),
            )
        except Exception as exc:
            entry.update(
                {
                    "status": "execution_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "summary_path": None,
                    "failed_request_count": None,
                }
            )
        else:
            failed_count = int(summary["metrics"]["failed_count"])
            entry.update(
                {
                    "status": "evidence_failed" if failed_count else "succeeded",
                    "error": None,
                    "summary_path": str(Path(run_id) / "summary.json"),
                    "failed_request_count": failed_count,
                }
            )
            summary_paths.append(run_dir / "summary.json")
        finally:
            write_json_atomic(state_path, state)

    comparison: dict[str, object] | None = None
    comparison_error: str | None = None
    if len(summary_paths) >= 2:
        try:
            comparison = compare_run_summaries(
                summary_paths,
                base_output / "comparison",
                source_root=base_output,
            )
        except Exception as exc:
            comparison_error = f"{type(exc).__name__}: {exc}"

    counts = _status_counts(state)
    matrix_summary = {
        "schema_version": "0.2",
        "matrix_name": matrix_name,
        "config_sha256": config_digest,
        "planned_run_count": len(plan["runs"]),
        "successful_run_count": counts["succeeded"],
        "evidence_failed_run_count": counts["evidence_failed"],
        "execution_failed_run_count": counts["execution_failed"],
        "pending_run_count": counts["pending"] + counts["running"],
        "skipped_on_resume_count": skipped_on_resume,
        "complete": all(
            counts[status] == 0
            for status in ["pending", "running", "execution_failed", "evidence_failed"]
        ),
        "ranking_allowed": bool(comparison and comparison.get("ranking_allowed")),
        "comparison_error": comparison_error,
        "comparison_path": "comparison/comparison.json" if comparison is not None else None,
        "quality": list(quality_by_profile.values()),
        "runs": state["runs"],
        "notes": [
            "Profile order is randomized within each repetition from the persisted matrix seed.",
            "Warmup requests are executed before each measured cell and are excluded from raw measured rows.",
            "Mock matrix results validate experiment orchestration only and are not model or backend performance evidence.",
        ],
    }
    write_json(base_output / "matrix_summary.json", matrix_summary)
    artifacts = [
        "manifest.json",
        "resolved_config.json",
        "matrix_plan.json",
        "matrix_state.json",
        "matrix_summary.json",
    ]
    if comparison is not None:
        artifacts.extend(
            [
                "comparison/manifest.json",
                "comparison/comparison.json",
                "comparison/comparison.md",
            ]
        )
    for profile_name in sorted(quality_by_profile):
        artifacts.extend(
            [
                f"quality/{profile_name}/manifest.json",
                f"quality/{profile_name}/task_specs.jsonl",
                f"quality/{profile_name}/task_outputs.jsonl",
                f"quality/{profile_name}/task_eval.json",
            ]
        )
    write_run_manifest(base_output, run_type="optimization_matrix", artifacts=artifacts)
    return matrix_summary


def _validate_matrix_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    errors: list[str] = []
    repetitions = get_path(config, "matrix.repetitions")
    if not isinstance(repetitions, int) or repetitions < MIN_REPETITIONS:
        errors.append(f"matrix.repetitions must be an integer of at least {MIN_REPETITIONS}")
    seed = get_path(config, "matrix.seed", 42)
    if not isinstance(seed, int):
        errors.append("matrix.seed must be an integer")
    baseline = get_path(config, "matrix.baseline_profile", "baseline")
    if baseline != "baseline":
        errors.append("matrix.baseline_profile must be 'baseline' for the v0.2 matrix")
    raw_profiles = config.get("profiles")
    if not isinstance(raw_profiles, dict):
        errors.append("profiles section must be a mapping")
        profiles: dict[str, dict[str, Any]] = {}
    else:
        profiles = {}
        for name, payload in raw_profiles.items():
            if not isinstance(name, str) or not isinstance(payload, dict):
                errors.append("every profile must be a named mapping")
                continue
            if PROFILE_SLUG.fullmatch(name) is None:
                errors.append(
                    f"profile name {name!r} must be a lowercase hyphen-separated slug"
                )
                continue
            profiles[name] = payload
    missing = sorted(REQUIRED_PROFILE_ROLES - set(profiles))
    if missing:
        errors.append(f"profiles section is missing required roles: {', '.join(missing)}")
    for name, profile in profiles.items():
        command = profile.get("server_command")
        command_file = profile.get("server_command_file")
        if not isinstance(command, str) and not isinstance(command_file, str):
            errors.append(f"profiles.{name} requires server_command or server_command_file")
    real_endpoints: list[str] = []
    for name, profile in profiles.items():
        backend = str(profile.get("backend", get_path(config, "endpoint.backend", "openai-compatible")))
        if backend == "mock":
            continue
        base_url = profile.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            errors.append(f"profiles.{name}.base_url is required for a real matrix endpoint")
        else:
            real_endpoints.append(base_url)
    if len(real_endpoints) != len(set(real_endpoints)):
        errors.append("real matrix profiles must use distinct pre-launched endpoint URLs")
    model_revision = get_path(config, "model.revision")
    global_tokenizer = get_path(config, "model.tokenizer", get_path(config, "model.name"))
    tokenizer_revision = get_path(config, "model.tokenizer_revision")
    for field, value in [("model.revision", model_revision), ("model.tokenizer_revision", tokenizer_revision)]:
        if not isinstance(value, str) or len(value) < 40:
            errors.append(f"{field} must be an immutable hexadecimal revision")
    for name, profile in profiles.items():
        profile_tokenizer = profile.get("tokenizer")
        profile_revision = profile.get("tokenizer_revision")
        if (
            isinstance(profile_tokenizer, str)
            and profile_tokenizer != global_tokenizer
            and profile_revision is None
        ):
            errors.append(
                f"profiles.{name}.tokenizer_revision is required for a profile-specific tokenizer"
            )
        if profile_revision is not None and (
            not isinstance(profile_revision, str) or len(profile_revision) < 40
        ):
            errors.append(
                f"profiles.{name}.tokenizer_revision must be an immutable hexadecimal revision"
            )
    for field in ["workload.input_tokens", "workload.output_tokens", "workload.concurrency"]:
        values = _as_list(get_path(config, field))
        if len(values) != len(set(values)):
            errors.append(f"{field} must not contain duplicate values")
    if errors:
        raise ValueError("invalid matrix config: " + "; ".join(errors))
    return profiles


def _build_plan(
    config: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    config_file: Path,
    base_output: Path,
    config_digest: str,
) -> dict[str, object]:
    repetitions = int(get_path(config, "matrix.repetitions"))
    seed = int(get_path(config, "matrix.seed", 42))
    inputs = _as_list(get_path(config, "workload.input_tokens", [128]))
    outputs = _as_list(get_path(config, "workload.output_tokens", [64]))
    concurrencies = _as_list(get_path(config, "workload.concurrency", [1]))
    cells = list(product(inputs, outputs, concurrencies))
    runs: list[dict[str, object]] = []
    run_ids: set[str] = set()
    plan_index = 0
    for repetition in range(1, repetitions + 1):
        order = sorted(profiles)
        random.Random(seed + repetition).shuffle(order)
        for randomized_order, profile_name in enumerate(order, start=1):
            profile = profiles[profile_name]
            profile_identity = execution_identity(
                profile=profile_name,
                model=str(get_path(config, "model.name")),
                backend=str(
                    profile.get(
                        "backend",
                        get_path(config, "endpoint.backend", "openai-compatible"),
                    )
                ),
                base_url=str(profile.get("base_url", get_path(config, "endpoint.base_url"))),
            )
            for cell_index, (input_tokens, output_tokens, concurrency) in enumerate(cells, start=1):
                cell_name = f"c{int(concurrency)}-in{int(input_tokens)}-out{int(output_tokens)}"
                run_id = f"{profile_name}/repeat-{repetition:02d}/{cell_name}"
                if run_id in run_ids:
                    raise ValueError(f"matrix plan contains duplicate run ID {run_id!r}")
                run_ids.add(run_id)
                runs.append(
                    {
                        "plan_index": plan_index,
                        "run_id": run_id,
                        "profile": profile_name,
                        "repetition": repetition,
                        "randomized_profile_order": randomized_order,
                        "cell_index": cell_index,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "concurrency": int(concurrency),
                        "execution_identity": profile_identity,
                        "server_command_file": _materialize_command(
                            config_file,
                            base_output,
                            profile_name,
                            profile,
                        ),
                    }
                )
                plan_index += 1
    return {
        "schema_version": "0.2",
        "matrix_name": str(get_path(config, "matrix.name", "optimization-matrix")),
        "config_path": str(config_file),
        "config_sha256": config_digest,
        "external_file_sha256": _external_file_hashes(config, config_file),
        "seed": seed,
        "repetitions": repetitions,
        "baseline_profile": "baseline",
        "profile_names": sorted(profiles),
        "runs": runs,
    }


def _run_planned_cell(
    config: dict[str, Any],
    config_file: Path,
    base_output: Path,
    planned: dict[str, object],
    quality: dict[str, object] | None,
) -> dict[str, object]:
    profile_name = str(planned["profile"])
    profile_config = config["profiles"][profile_name]
    base_url = str(profile_config.get("base_url", get_path(config, "endpoint.base_url")))
    backend = str(profile_config.get("backend", get_path(config, "endpoint.backend", "openai-compatible")))
    model = str(get_path(config, "model.name"))
    model_revision = str(profile_config.get("model_revision", get_path(config, "model.revision")))
    tokenizer, tokenizer_revision = _resolve_profile_tokenizer(config, profile_config)
    dtype = str(profile_config.get("dtype", get_path(config, "model.dtype", "unknown")))
    quantization = str(profile_config.get("quantization", "none"))
    run_dir = _contained_path(base_output, str(planned["run_id"]))
    command_path = Path(str(planned["server_command_file"]))
    command_text = command_path.read_text(encoding="utf-8")
    prompt_path = get_path(config, "workload.prompts_path")
    prompt_texts = None
    if isinstance(prompt_path, str):
        resolved_prompt = Path(prompt_path)
        if not resolved_prompt.is_absolute():
            resolved_prompt = config_file.parent / resolved_prompt
        prompt_texts = load_prompt_file(resolved_prompt)
    summary = run_throughput_benchmark(
        base_url=base_url,
        model=model,
        concurrency=int(planned["concurrency"]),
        input_tokens=int(planned["input_tokens"]),
        output_tokens=int(planned["output_tokens"]),
        output_dir=run_dir,
        request_count=int(get_path(config, "run.measured_requests", 8)),
        warmup_count=int(get_path(config, "run.warmup_requests", 0)),
        timeout_seconds=float(get_path(config, "run.timeout_seconds", 120.0)),
        dtype=dtype,
        quantization=quantization,
        backend=backend,
        seed=int(get_path(config, "workload.seed", 42)),
        stream=bool(get_path(config, "endpoint.stream", True)),
        hardware_label=str(get_path(config, "run.hardware_label", "local")),
        api_kind=str(get_path(config, "endpoint.api_kind", "chat")),
        prompt_texts=prompt_texts,
        model_revision=model_revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        optimization_profile=profile_name,
        server_command_file=command_path,
        request_schedule=str(get_path(config, "workload.request_schedule", "closed-loop")),
        request_rate_rps=_optional_float(get_path(config, "workload.request_rate_rps")),
        client_processes=int(get_path(config, "run.client_processes", 1)),
        queue_delay_warning_ms=float(get_path(config, "run.queue_delay_warning_ms", 10.0)),
    )
    summary_metadata = summary.get("metadata")
    if not isinstance(summary_metadata, dict):
        raise ValueError("summary metadata must be a mapping")
    optimization_profile = create_optimization_profile(
        name=profile_name,
        backend=backend,
        backend_version=detect_backend_version("mock" if base_url.startswith("mock://") else backend),
        server_command=command_text,
        model=model,
        model_revision=model_revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        dtype=dtype,
        quantization=quantization,
        environment_fingerprint=environment_fingerprint(summary_metadata),
        prefix_cache=bool(profile_config.get("prefix_cache", False)),
        chunked_prefill=bool(profile_config.get("chunked_prefill", False)),
        speculative_model=_optional_string(profile_config.get("speculative_model")),
        speculative_model_revision=_optional_string(profile_config.get("speculative_model_revision")),
        num_speculative_tokens=_optional_int(profile_config.get("num_speculative_tokens")),
        max_num_batched_tokens=_optional_int(profile_config.get("max_num_batched_tokens")),
        max_num_seqs=_optional_int(profile_config.get("max_num_seqs")),
        max_model_len=_optional_int(profile_config.get("max_model_len")),
        gpu_memory_utilization=_optional_float(profile_config.get("gpu_memory_utilization")),
    )
    write_optimization_profile(run_dir, optimization_profile)
    _bind_profile_to_summary(
        run_dir,
        summary,
        optimization_profile.to_dict(),
        matrix_name=str(get_path(config, "matrix.name", "optimization-matrix")),
        repetition=int(planned["repetition"]),
        randomized_order=int(planned["randomized_profile_order"]),
        quality=quality,
    )
    return summary


def _bind_profile_to_summary(
    run_dir: Path,
    summary: dict[str, object],
    profile: dict[str, object],
    *,
    matrix_name: str,
    repetition: int,
    randomized_order: int,
    quality: dict[str, object] | None,
) -> None:
    metadata = summary["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("summary metadata must be a mapping")
    metadata["optimization_profile_spec"] = profile
    metadata["optimization_profile_fingerprint"] = profile["semantic_fingerprint"]
    metadata["optimization_treatment_fingerprint"] = profile["treatment_fingerprint"]
    tokenizer = profile["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise ValueError("optimization profile tokenizer must be a mapping")
    metadata["tokenizer"] = tokenizer["name"]
    metadata["tokenizer_revision"] = tokenizer["revision"]
    metadata["environment_fingerprint"] = profile["environment_fingerprint"]
    metadata["matrix_name"] = matrix_name
    metadata["matrix_repetition"] = repetition
    metadata["matrix_randomized_order"] = randomized_order
    if quality is not None:
        metadata["quality_gate"] = {
            "task_set_sha256": quality["task_set_sha256"],
            "max_allowed_score_drop": quality["max_allowed_score_drop"],
            "execution_identity": quality["execution_identity"],
        }
        metadata["quality_score"] = quality["mean_score"]
        metadata["quality_score_drop_from_baseline"] = quality["score_drop_from_baseline"]
        metadata["quality_task_passed"] = quality["task_passed"]
        metadata["quality_passed"] = quality["quality_gate_passed"]
        metadata["quality_evidence_path"] = quality["evidence_path"]
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = list(manifest.get("artifacts", []))
    if "optimization_profile.json" not in artifacts:
        artifacts.append("optimization_profile.json")
    manifest["artifacts"] = artifacts
    write_json(manifest_path, manifest)


def _materialize_command(
    config_file: Path,
    base_output: Path,
    profile_name: str,
    profile: dict[str, Any],
) -> str:
    command_dir = _contained_path(base_output, "profile_commands")
    command_dir.mkdir(parents=True, exist_ok=True)
    destination = _contained_path(base_output, "profile_commands", f"{profile_name}.txt")
    command_file = profile.get("server_command_file")
    if isinstance(command_file, str):
        source = Path(command_file)
        if not source.is_absolute():
            source = config_file.parent / source
        command_bytes = source.read_bytes()
    else:
        command = str(profile["server_command"])
        command_bytes = (command if command.endswith("\n") else command + "\n").encode("utf-8")
    write_bytes_atomic(destination, command_bytes)
    return str(destination)


def _initial_state(plan: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "0.2",
        "config_sha256": plan["config_sha256"],
        "runs": [
            {
                "plan_index": run["plan_index"],
                "run_id": run["run_id"],
                "profile": run["profile"],
                "repetition": run["repetition"],
                "status": "pending",
                "attempts": 0,
                "summary_path": None,
                "failed_request_count": None,
                "error": None,
            }
            for run in plan["runs"]
        ],
    }


def _validate_resume(plan_path: Path, state_path: Path, config_digest: str) -> None:
    if not plan_path.exists() or not state_path.exists():
        raise ValueError("resume requires matrix_plan.json and matrix_state.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if plan.get("config_sha256") != config_digest or state.get("config_sha256") != config_digest:
        raise ValueError("matrix config changed; resume is blocked to preserve experiment identity")


def _valid_existing_run(
    run_dir: Path,
    planned: dict[str, object],
    *,
    matrix_name: str,
    quality: dict[str, object] | None,
) -> bool:
    if not (run_dir / "summary.json").exists() or not (run_dir / "optimization_profile.json").exists():
        return False
    try:
        result = validate_run_dir(run_dir)
        if not result["valid"]:
            return False
        profile = load_optimization_profile(run_dir / "optimization_profile.json")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (run_dir / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(summary, dict) or not all(isinstance(row, dict) for row in rows):
        return False
    metrics = summary.get("metrics")
    metadata = summary.get("metadata")
    if not isinstance(metrics, dict) or not isinstance(metadata, dict):
        return False
    expected_metadata = {
        "matrix_name": matrix_name,
        "matrix_repetition": planned.get("repetition"),
        "matrix_randomized_order": planned.get("randomized_profile_order"),
        "optimization_profile": planned.get("profile"),
        "concurrency": planned.get("concurrency"),
        "requested_input_tokens": planned.get("input_tokens"),
        "output_tokens": planned.get("output_tokens"),
    }
    if any(metadata.get(field) != expected for field, expected in expected_metadata.items()):
        return False
    actual_identity = {
        "profile": metadata.get("optimization_profile"),
        "model": metadata.get("model"),
        "backend": metadata.get("backend"),
        "base_url": metadata.get("base_url"),
        "endpoint_sha256": metadata.get("endpoint_sha256"),
    }
    if actual_identity != planned.get("execution_identity"):
        return False
    if {row.get("token_count_method") for row in rows} != {
        metadata.get("token_count_method")
    }:
        return False
    if quality is not None:
        expected_quality = {
            "quality_gate": {
                "task_set_sha256": quality.get("task_set_sha256"),
                "max_allowed_score_drop": quality.get("max_allowed_score_drop"),
                "execution_identity": quality.get("execution_identity"),
            },
            "quality_score": quality.get("mean_score"),
            "quality_score_drop_from_baseline": quality.get("score_drop_from_baseline"),
            "quality_task_passed": quality.get("task_passed"),
            "quality_passed": quality.get("quality_gate_passed"),
            "quality_evidence_path": quality.get("evidence_path"),
        }
        if any(metadata.get(field) != expected for field, expected in expected_quality.items()):
            return False
    if profile.name != planned.get("profile"):
        return False
    if metadata.get("optimization_profile_fingerprint") != profile.semantic_fingerprint:
        return False
    if metadata.get("optimization_treatment_fingerprint") != profile.treatment_fingerprint:
        return False
    command_path = run_dir / "server_command.txt"
    planned_command_path = Path(str(planned.get("server_command_file", "")))
    if not command_path.exists() or not planned_command_path.exists():
        return False
    command_sha256 = hashlib.sha256(command_path.read_bytes()).hexdigest()
    planned_command_sha256 = hashlib.sha256(planned_command_path.read_bytes()).hexdigest()
    if command_sha256 != planned_command_sha256 or command_sha256 != profile.server_command_sha256:
        return False
    if metrics.get("request_count") != len(rows):
        return False
    completed = sum(1 for row in rows if row.get("completed") is True)
    failed = sum(1 for row in rows if row.get("completed") is False)
    if metrics.get("completed_count") != completed or metrics.get("failed_count") != failed:
        return False
    request_ids = [row.get("request_id") for row in rows]
    if len(set(request_ids)) != len(request_ids) or None in request_ids:
        return False
    return all(_valid_raw_timing_row(row) for row in rows)


def _valid_raw_timing_row(row: dict[str, object]) -> bool:
    fields = [
        "scheduled_offset_ms",
        "dispatch_offset_ms",
        "completed_offset_ms",
        "queue_delay_ms",
        "end_to_end_latency_ms",
        "total_latency_ms",
    ]
    if any(
        not isinstance(row.get(field), (int, float))
        or isinstance(row.get(field), bool)
        or not math.isfinite(float(row[field]))
        or float(row[field]) < 0
        for field in fields
    ):
        return False
    scheduled = float(row["scheduled_offset_ms"])
    dispatch = float(row["dispatch_offset_ms"])
    completed = float(row["completed_offset_ms"])
    if scheduled > dispatch or dispatch > completed:
        return False
    if not math.isclose(float(row["queue_delay_ms"]), dispatch - scheduled, rel_tol=1e-9, abs_tol=1e-6):
        return False
    if not math.isclose(float(row["end_to_end_latency_ms"]), completed - scheduled, rel_tol=1e-9, abs_tol=1e-6):
        return False
    timing_tolerance_ms = max(5.0, (completed - dispatch) * 0.01)
    return math.isclose(
        float(row["total_latency_ms"]),
        completed - dispatch,
        rel_tol=0.01,
        abs_tol=timing_tolerance_ms,
    )


def _status_counts(state: dict[str, object]) -> dict[str, int]:
    counts = {status: 0 for status in ["pending", "running", "succeeded", "evidence_failed", "execution_failed"]}
    for entry in state["runs"]:
        status = str(entry["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _config_digest(config: dict[str, Any], config_file: Path) -> str:
    external_files = _external_file_hashes(config, config_file)
    identity = {
        "config": sanitize_resolved_config(config),
        "external_file_sha256": external_files,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _external_file_hashes(config: dict[str, Any], config_file: Path) -> dict[str, str]:
    external_files: dict[str, str] = {}
    referenced_paths: list[tuple[str, object]] = [
        ("workload.prompts_path", get_path(config, "workload.prompts_path")),
        ("quality.tasks_path", get_path(config, "quality.tasks_path")),
    ]
    profiles = config.get("profiles", {})
    if isinstance(profiles, dict):
        referenced_paths.extend(
            (f"profiles.{name}.server_command_file", profile.get("server_command_file"))
            for name, profile in profiles.items()
            if isinstance(profile, dict)
        )
    for label, value in referenced_paths:
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = config_file.parent / path
        external_files[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return external_files


def _run_quality_gates(
    config: dict[str, Any],
    config_file: Path,
    base_output: Path,
    profiles: dict[str, dict[str, Any]],
    *,
    resume: bool,
) -> dict[str, dict[str, object]]:
    tasks_value = get_path(config, "quality.tasks_path")
    if not isinstance(tasks_value, str):
        return {}
    tasks_path = Path(tasks_value)
    if not tasks_path.is_absolute():
        tasks_path = config_file.parent / tasks_path
    task_specs = load_task_specs(tasks_path)
    max_tokens = int(get_path(config, "quality.max_tokens", 64))
    max_allowed_score_drop = float(get_path(config, "quality.max_allowed_score_drop", 0.0))
    raw: dict[str, dict[str, object]] = {}
    for profile_name, profile in profiles.items():
        output_dir = _contained_path(base_output, "quality", profile_name)
        report_path = output_dir / "task_eval.json"
        base_url = str(profile.get("base_url", get_path(config, "endpoint.base_url")))
        backend = str(profile.get("backend", get_path(config, "endpoint.backend", "openai-compatible")))
        identity = execution_identity(
            profile=profile_name,
            model=str(get_path(config, "model.name")),
            backend=backend,
            base_url=base_url,
        )
        tokenizer, tokenizer_revision = _resolve_profile_tokenizer(config, profile)
        if resume and report_path.exists() and _valid_existing_quality_run(
            output_dir,
            task_specs,
            identity,
        ):
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = evaluate_tasks(
                base_url=base_url,
                model=str(get_path(config, "model.name")),
                task_specs=task_specs,
                output_dir=output_dir,
                backend=backend,
                profile=profile_name,
                tokenizer=tokenizer,
                tokenizer_revision=tokenizer_revision,
                max_tokens=max_tokens,
                stream=bool(get_path(config, "endpoint.stream", True)),
            )
        raw[profile_name] = {
            "profile": profile_name,
            "execution_identity": report["execution_identity"],
            "task_set_sha256": report["task_set_sha256"],
            "mean_score": float(report["mean_score"]),
            "task_passed": bool(report["passed"]),
            "max_allowed_score_drop": max_allowed_score_drop,
            "evidence_path": f"quality/{profile_name}/task_eval.json",
        }
    baseline_score = float(raw["baseline"]["mean_score"])
    for result in raw.values():
        score_drop = baseline_score - float(result["mean_score"])
        result["score_drop_from_baseline"] = score_drop
        result["quality_gate_passed"] = bool(result["task_passed"]) and score_drop <= max_allowed_score_drop
    return raw


def _valid_existing_quality_run(
    output_dir: Path,
    expected_specs: list[dict[str, object]],
    expected_identity: dict[str, str],
) -> bool:
    try:
        if not validate_run_dir(output_dir)["valid"]:
            return False
        specs = load_task_specs(output_dir / "task_specs.jsonl")
        outputs = [
            json.loads(line)
            for line in (output_dir / "task_outputs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = json.loads((output_dir / "task_eval.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if specs != expected_specs:
        return False
    if report.get("execution_identity") != expected_identity:
        return False
    for field in ["model", "backend", "base_url"]:
        if report.get(field) != expected_identity[field]:
            return False
    checks = report.get("checks") if isinstance(report, dict) else None
    if not isinstance(checks, list) or len(specs) != len(outputs) or len(specs) != len(checks) or not specs:
        return False
    scores: list[float] = []
    passed_count = 0
    for index, (spec, output, check) in enumerate(zip(specs, outputs, checks, strict=True)):
        if not isinstance(output, dict) or not isinstance(check, dict):
            return False
        case_id = spec.get("id")
        if (
            output.get("case_id") != case_id
            or check.get("case_id") != case_id
            or output.get("output_index") != index
            or check.get("output_index") != index
        ):
            return False
        validator = spec.get("validator")
        if not isinstance(validator, dict):
            return False
        output_text = output.get("output_text")
        if isinstance(output_text, str) and output.get("error") is None:
            validation = validate_output(output_text, validator)
            score = validation.score
            passed = validation.passed
        else:
            score = 0.0
            passed = False
        if not _numbers_close(check.get("score"), score) or check.get("passed") is not passed:
            return False
        scores.append(score)
        passed_count += int(passed)
    mean_score = sum(scores) / len(scores)
    canonical = json.dumps(specs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    task_set_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected = {
        "task_set_sha256": task_set_sha256,
        "task_count": len(scores),
        "passed_count": passed_count,
        "failed_count": len(scores) - passed_count,
        "passed": passed_count == len(scores),
    }
    if any(report.get(field) != value for field, value in expected.items()):
        return False
    return _numbers_close(report.get("mean_score"), mean_score)


def _numbers_close(value: object, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _resolve_profile_tokenizer(
    config: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[str, str]:
    model = str(get_path(config, "model.name"))
    global_tokenizer = str(get_path(config, "model.tokenizer", model))
    tokenizer = str(profile.get("tokenizer", global_tokenizer))
    if tokenizer == global_tokenizer:
        revision = profile.get("tokenizer_revision", get_path(config, "model.tokenizer_revision"))
    else:
        revision = profile.get("tokenizer_revision")
    if not isinstance(revision, str):
        raise ValueError("profile-specific tokenizer requires tokenizer_revision")
    return tokenizer, revision


def _contained_path(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"derived output path escapes matrix output root: {candidate}") from exc
    return candidate


def _optional_string(value: object) -> str | None:
    return str(value) if value not in {None, ""} else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None
