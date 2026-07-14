from __future__ import annotations

import json
from pathlib import Path

from llm_accel.metrics.optimization_profile import load_optimization_profile
from llm_accel.metrics.schemas import SCHEMA_VERSION


def validate_run_dir(path: str | Path) -> dict[str, object]:
    run_dir = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append("missing manifest.json")
        manifest: dict[str, object] = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            warnings.append(
                f"schema version mismatch: expected {SCHEMA_VERSION}, got {manifest.get('schema_version')}"
            )
        for artifact in manifest.get("artifacts", []):
            if not (run_dir / str(artifact)).exists():
                errors.append(f"missing artifact listed in manifest: {artifact}")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        _validate_summary(summary_path, errors, warnings)
    elif (run_dir / "aggregate_summary.json").exists():
        _validate_aggregate(run_dir / "aggregate_summary.json", errors)
    elif manifest:
        warnings.append("no standard summary artifact found")
    _validate_optional_artifacts(run_dir, errors, warnings)

    return {
        "path": str(run_dir),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _validate_summary(path: Path, errors: list[str], warnings: list[str]) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    for key in ["schema_version", "metadata", "metrics", "memory", "warnings"]:
        if key not in summary:
            errors.append(f"summary.json missing {key}")
    if summary.get("schema_version") != SCHEMA_VERSION:
        warnings.append(f"summary schema version mismatch: {summary.get('schema_version')}")
    metrics = summary.get("metrics", {})
    for key in ["request_count", "completed_count", "failed_count", "timeout_count", "latency_ms", "throughput"]:
        if key not in metrics:
            errors.append(f"summary.metrics missing {key}")
    metadata = summary.get("metadata", {})
    for key in ["api_kind", "backend_version", "project_version", "python_version", "operating_system", "hardware_label"]:
        if key not in metadata:
            errors.append(f"summary.metadata missing {key}")
    if "warnings" in summary and not isinstance(summary["warnings"], list):
        errors.append("summary.warnings must be a list")


def _validate_aggregate(path: Path, errors: list[str]) -> None:
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    if "runs" not in aggregate:
        errors.append("aggregate_summary.json missing runs")
    if aggregate.get("run_count") != len(aggregate.get("runs", [])):
        errors.append("aggregate run_count does not match runs length")


def _validate_optional_artifacts(run_dir: Path, errors: list[str], warnings: list[str]) -> None:
    validators = {
        "throughput_summary.json": _validate_throughput_summary,
        "quality_eval.json": _validate_quality_eval,
        "task_eval.json": _validate_task_eval,
        "quantization_comparison.json": _validate_quantization_comparison,
        "comparison.json": _validate_comparison,
        "matrix_summary.json": _validate_matrix_summary,
        "vllm_validation.json": _validate_vllm_validation,
        "vllm_benchmark_plan.json": _validate_vllm_plan,
        "speculative_summary.json": _validate_speculative_summary,
        "acceptance_curve.json": _validate_acceptance_curve,
        "baseline_comparison.json": _validate_baseline_comparison,
    }
    for artifact, validator in validators.items():
        path = run_dir / artifact
        if path.exists():
            validator(path, errors, warnings)
    profile_path = run_dir / "optimization_profile.json"
    if profile_path.exists():
        try:
            load_optimization_profile(profile_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"optimization_profile.json is invalid: {exc}")


def _validate_throughput_summary(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(path.name, payload, ["schema_version", "metadata", "throughput", "completed_count", "failed_count", "timeout_count", "warnings"], errors)
    _require_mapping(path.name, payload.get("metadata"), "metadata", errors)
    throughput = payload.get("throughput")
    _require_mapping(path.name, throughput, "throughput", errors)
    if isinstance(throughput, dict):
        for key in ["output_tokens_per_second", "requests_per_second", "measured_elapsed_seconds"]:
            if key not in throughput:
                errors.append(f"{path.name}.throughput missing {key}")
    _require_list(path.name, payload.get("warnings"), "warnings", errors)
    if payload.get("schema_version") != SCHEMA_VERSION:
        warnings.append(f"{path.name} schema version mismatch: {payload.get('schema_version')}")


def _validate_quality_eval(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(path.name, payload, ["schema_version", "model", "backend", "prompt_count", "prompt_set_sha256", "passed", "checks", "notes"], errors)
    _require_list(path.name, payload.get("checks"), "checks", errors)
    _require_list(path.name, payload.get("notes"), "notes", errors)
    if not isinstance(payload.get("passed"), bool):
        errors.append(f"{path.name}.passed must be a boolean")
    if isinstance(payload.get("checks"), list) and payload.get("prompt_count") != len(payload["checks"]):
        errors.append(f"{path.name} prompt_count does not match checks length")
    outputs_path = path.parent / "quality_outputs.jsonl"
    if not outputs_path.exists():
        errors.append("quality evaluation missing quality_outputs.jsonl")
    else:
        outputs = _read_mapping_jsonl(outputs_path, errors)
        if payload.get("prompt_count") != len(outputs):
            errors.append("quality_outputs.jsonl row count does not match prompt_count")


def _validate_task_eval(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(
        path.name,
        payload,
        [
            "schema_version",
            "model",
            "backend",
            "task_set_sha256",
            "task_count",
            "passed_count",
            "failed_count",
            "mean_score",
            "passed",
            "checks",
            "notes",
        ],
        errors,
    )
    _require_list(path.name, payload.get("checks"), "checks", errors)
    _require_list(path.name, payload.get("notes"), "notes", errors)
    if not isinstance(payload.get("passed"), bool):
        errors.append(f"{path.name}.passed must be a boolean")
    if isinstance(payload.get("checks"), list) and payload.get("task_count") != len(payload["checks"]):
        errors.append(f"{path.name} task_count does not match checks length")
    if isinstance(payload.get("task_count"), int) and isinstance(payload.get("passed_count"), int) and isinstance(payload.get("failed_count"), int):
        if payload["passed_count"] + payload["failed_count"] != payload["task_count"]:
            errors.append(f"{path.name} passed_count plus failed_count does not match task_count")
    if not isinstance(payload.get("mean_score"), (int, float)) or not 0.0 <= float(payload.get("mean_score", -1)) <= 1.0:
        errors.append(f"{path.name}.mean_score must be between 0 and 1")

    specs_path = path.parent / "task_specs.jsonl"
    outputs_path = path.parent / "task_outputs.jsonl"
    if not specs_path.exists():
        errors.append("task evaluation missing task_specs.jsonl")
    if not outputs_path.exists():
        errors.append("task evaluation missing task_outputs.jsonl")
    if specs_path.exists() and outputs_path.exists():
        specs = _read_mapping_jsonl(specs_path, errors)
        outputs = _read_mapping_jsonl(outputs_path, errors)
        checks = payload.get("checks", [])
        task_count = payload.get("task_count")
        if task_count != len(specs):
            errors.append("task_specs.jsonl row count does not match task_count")
        if task_count != len(outputs):
            errors.append("task_outputs.jsonl row count does not match task_count")
        if isinstance(checks, list):
            spec_ids = [row.get("id") for row in specs]
            output_ids = [row.get("case_id") for row in outputs]
            check_ids = [row.get("case_id") for row in checks if isinstance(row, dict)]
            if len(set(spec_ids)) != len(spec_ids):
                errors.append("task_specs.jsonl ids must be unique")
            if spec_ids != output_ids or spec_ids != check_ids:
                errors.append("task spec, output, and summary case ids do not match in order")
        for row in outputs:
            for key in ["case_id", "output_index", "output_text", "output_tokens", "ttft_ms", "total_latency_ms", "error"]:
                if key not in row:
                    errors.append(f"task_outputs.jsonl row missing {key}")


def _read_mapping_jsonl(path: Path, errors: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name} line {line_no} is invalid JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path.name} line {line_no} must be an object")
            continue
        rows.append(payload)
    return rows


def _validate_quantization_comparison(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(path.name, payload, ["model", "backend", "modes", "supported_modes", "runs", "warnings", "notes"], errors)
    for key in ["modes", "supported_modes", "runs", "warnings", "notes"]:
        _require_list(path.name, payload.get(key), key, errors)
    if isinstance(payload.get("runs"), list) and isinstance(payload.get("modes"), list) and len(payload["runs"]) != len(payload["modes"]):
        errors.append(f"{path.name} runs length does not match modes length")


def _validate_comparison(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(
        path.name,
        payload,
        [
            "comparison_schema_version",
            "summary_count",
            "runs",
            "comparable",
            "ranking_allowed",
            "blockers",
            "strata",
            "warnings",
            "notes",
        ],
        errors,
    )
    _require_list(path.name, payload.get("runs"), "runs", errors)
    _require_list(path.name, payload.get("warnings"), "warnings", errors)
    _require_list(path.name, payload.get("blockers"), "blockers", errors)
    _require_list(path.name, payload.get("strata"), "strata", errors)
    _require_list(path.name, payload.get("notes"), "notes", errors)
    for key in ["comparable", "ranking_allowed"]:
        if not isinstance(payload.get(key), bool):
            errors.append(f"{path.name}.{key} must be a boolean")
    if isinstance(payload.get("runs"), list) and payload.get("summary_count") != len(payload["runs"]):
        errors.append(f"{path.name} summary_count does not match runs length")


def _validate_matrix_summary(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(
        path.name,
        payload,
        [
            "schema_version",
            "matrix_name",
            "planned_run_count",
            "successful_run_count",
            "evidence_failed_run_count",
            "execution_failed_run_count",
            "pending_run_count",
            "runs",
            "quality",
            "complete",
            "ranking_allowed",
        ],
        errors,
    )
    _require_list(path.name, payload.get("runs"), "runs", errors)
    _require_list(path.name, payload.get("quality"), "quality", errors)
    runs = payload.get("runs")
    if isinstance(runs, list) and payload.get("planned_run_count") != len(runs):
        errors.append(f"{path.name} planned_run_count does not match runs length")
    if isinstance(runs, list):
        statuses = [run.get("status") for run in runs if isinstance(run, dict)]
        expected_counts = {
            "successful_run_count": statuses.count("succeeded"),
            "evidence_failed_run_count": statuses.count("evidence_failed"),
            "execution_failed_run_count": statuses.count("execution_failed"),
            "pending_run_count": statuses.count("pending") + statuses.count("running"),
        }
        for field, expected in expected_counts.items():
            if payload.get(field) != expected:
                errors.append(f"{path.name} {field} does not match run statuses")


def _validate_vllm_validation(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(path.name, payload, ["model", "base_url", "command", "checks", "ready_for_hardware_benchmark", "blockers"], errors)
    _require_mapping(path.name, payload.get("command"), "command", errors)
    _require_mapping(path.name, payload.get("checks"), "checks", errors)
    _require_list(path.name, payload.get("blockers"), "blockers", errors)
    if not isinstance(payload.get("ready_for_hardware_benchmark"), bool):
        errors.append(f"{path.name}.ready_for_hardware_benchmark must be a boolean")


def _validate_vllm_plan(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(path.name, payload, ["model", "base_url", "config_path", "server_command", "steps", "required_artifacts", "claim_rules"], errors)
    _require_mapping(path.name, payload.get("server_command"), "server_command", errors)
    for key in ["steps", "required_artifacts", "claim_rules"]:
        _require_list(path.name, payload.get(key), key, errors)


def _validate_speculative_summary(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(path.name, payload, ["draft_model", "target_model", "result", "acceptance_curve", "baseline_comparison"], errors)
    _require_mapping(path.name, payload.get("result"), "result", errors)
    _require_list(path.name, payload.get("acceptance_curve"), "acceptance_curve", errors)
    _require_mapping(path.name, payload.get("baseline_comparison"), "baseline_comparison", errors)


def _validate_acceptance_curve(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        errors.append(f"{path.name} must be a list")


def _validate_baseline_comparison(path: Path, errors: list[str], warnings: list[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(path.name, payload, ["baseline", "speculative", "estimated_speedup", "saved_steps", "relative_step_reduction", "interpretation"], errors)
    _require_mapping(path.name, payload.get("baseline"), "baseline", errors)
    _require_mapping(path.name, payload.get("speculative"), "speculative", errors)


def _require_keys(name: str, payload: object, keys: list[str], errors: list[str]) -> None:
    if not isinstance(payload, dict):
        errors.append(f"{name} must be a JSON object")
        return
    for key in keys:
        if key not in payload:
            errors.append(f"{name} missing {key}")


def _require_mapping(name: str, value: object, field: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{name}.{field} must be an object")


def _require_list(name: str, value: object, field: str, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{name}.{field} must be a list")
