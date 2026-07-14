from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

from llm_accel.evaluation.validators import validate_output
from llm_accel.metrics.environment import environment_fingerprint
from llm_accel.metrics.optimization_profile import load_bound_optimization_profile
from llm_accel.reports.claim_audit import audit_hardware_claim
from llm_accel.reports.comparison import compare_run_summaries
from llm_accel.reports.validation import validate_run_dir


MIN_PROFILE_REPETITIONS = 3


def audit_performance_ranking(matrix_dir: str | Path) -> dict[str, object]:
    root = Path(matrix_dir)
    blockers: list[dict[str, object]] = []
    warnings: list[str] = []
    matrix = _read_object(root / "matrix_summary.json", blockers, code="missing_matrix_summary")
    plan = _read_object(root / "matrix_plan.json", blockers, code="missing_matrix_plan")
    state_artifact = _read_object(root / "matrix_state.json", blockers, code="missing_matrix_state")
    comparison = _read_object(
        root / "comparison" / "comparison.json",
        blockers,
        code="missing_comparison",
    )
    if matrix is None:
        return _report(root, blockers, warnings, {})

    planned_by_id = _audit_matrix_chain(matrix, plan, state_artifact, blockers)

    runs = matrix.get("runs")
    if not isinstance(runs, list) or not runs:
        _add(blockers, "missing_repetitions", "matrix_summary.json must contain run evidence")
        runs = []
    profile_repetitions: dict[str, set[int]] = defaultdict(set)
    treatment_fingerprints: dict[str, set[str]] = defaultdict(set)
    run_execution_identities: dict[str, set[str]] = defaultdict(set)
    audited_summary_paths: list[Path] = []
    source_audits: list[dict[str, object]] = []
    for state in runs:
        if not isinstance(state, dict):
            _add(blockers, "invalid_matrix_state", "matrix run state must be an object")
            continue
        run_id = str(state.get("run_id", ""))
        profile = str(state.get("profile", ""))
        repetition = _integer(state.get("repetition"))
        planned = planned_by_id.get(run_id)
        if planned is None:
            _add(blockers, "unplanned_run", f"Run {run_id or '<unknown>'} is not present in matrix_plan.json.", run_id=run_id)
        else:
            _audit_planned_state(state, planned, blockers)
        status = state.get("status")
        if status != "succeeded":
            _add(
                blockers,
                "failed_repetition",
                f"Run {run_id or '<unknown>'} has status {status!r} and cannot support ranking.",
                run_id=run_id,
            )
            continue
        if repetition is not None:
            profile_repetitions[profile].add(repetition)
        run_dir = root / run_id
        audited_summary_paths.append(run_dir / "summary.json")
        run_evidence = _audit_run_evidence(run_dir, run_id, planned, blockers)
        treatment = run_evidence.get("treatment_fingerprint")
        if isinstance(treatment, str):
            treatment_fingerprints[profile].add(treatment)
        identity = run_evidence.get("execution_identity")
        if isinstance(identity, dict):
            run_execution_identities[profile].add(
                json.dumps(identity, sort_keys=True, separators=(",", ":"))
            )
        single = audit_hardware_claim(run_dir)
        source_audits.append(
            {
                "run_id": run_id,
                "publishable_hardware_claim": single["publishable_hardware_claim"],
                "blocker_count": len(single["blockers"]),
            }
        )
        if not single["publishable_hardware_claim"]:
            _add(
                blockers,
                "single_run_audit_failed",
                f"Run {run_id} does not pass the hardware claim audit.",
                run_id=run_id,
                source_blockers=single["blockers"],
            )

    for profile, repetitions in sorted(profile_repetitions.items()):
        if len(repetitions) < MIN_PROFILE_REPETITIONS:
            _add(
                blockers,
                "insufficient_repetitions",
                f"Profile {profile!r} has {len(repetitions)} valid repetitions; {MIN_PROFILE_REPETITIONS} are required.",
                profile=profile,
            )
    if "baseline" not in profile_repetitions:
        _add(blockers, "missing_baseline", "A valid baseline profile is required.")
    if len(profile_repetitions) < 2:
        _add(blockers, "missing_treatment", "A ranking requires a baseline and at least one treatment profile.")
    for profile, fingerprints in sorted(treatment_fingerprints.items()):
        if len(fingerprints) != 1:
            _add(
                blockers,
                "treatment_identity_mismatch",
                f"Profile {profile!r} does not resolve to exactly one treatment fingerprint.",
                profile=profile,
            )

    _audit_quality(
        root,
        matrix.get("quality"),
        profile_repetitions,
        run_execution_identities,
        blockers,
    )
    _audit_run_quality_bindings(root, runs, matrix.get("quality"), blockers)
    if comparison is not None:
        _audit_comparison_chain(root, comparison, audited_summary_paths, blockers)
        if not comparison.get("ranking_allowed"):
            _add(
                blockers,
                "comparison_blocked",
                "The comparison artifact does not allow ranking.",
                comparison_blockers=comparison.get("blockers", []),
            )
        strata = comparison.get("strata")
        if not isinstance(strata, list) or len(strata) != 1:
            _add(
                blockers,
                "multiple_comparison_strata",
                "A publishable ranking requires exactly one compatible comparison stratum.",
            )

    evidence = {
        "planned_run_count": matrix.get("planned_run_count"),
        "successful_run_count": matrix.get("successful_run_count"),
        "profile_repetitions": {
            profile: sorted(repetitions) for profile, repetitions in sorted(profile_repetitions.items())
        },
        "quality_profile_count": len(matrix.get("quality", [])) if isinstance(matrix.get("quality"), list) else 0,
        "source_audits": source_audits,
    }
    warnings.append(
        "Artifact auditing cannot attest that each live endpoint process matched its recorded server command; preserve operator or platform launch evidence."
    )
    return _report(root, blockers, warnings, evidence)


def _audit_run_quality_bindings(
    root: Path,
    runs: list[object],
    quality: object,
    blockers: list[dict[str, object]],
) -> None:
    if not isinstance(quality, list):
        return
    by_profile = {
        str(item.get("profile")): item
        for item in quality
        if isinstance(item, dict) and isinstance(item.get("profile"), str)
    }
    for state in runs:
        if not isinstance(state, dict) or state.get("status") != "succeeded":
            continue
        run_id = str(state.get("run_id", ""))
        profile = str(state.get("profile", ""))
        result = by_profile.get(profile)
        if result is None:
            continue
        summary = _read_object(
            root / run_id / "summary.json",
            blockers,
            code="missing_run_summary",
            run_id=run_id,
        )
        if summary is None or not isinstance(summary.get("metadata"), dict):
            continue
        metadata = summary["metadata"]
        expected = {
            "quality_gate": {
                "task_set_sha256": result.get("task_set_sha256"),
                "max_allowed_score_drop": result.get("max_allowed_score_drop"),
                "execution_identity": result.get("execution_identity"),
            },
            "quality_score": result.get("mean_score"),
            "quality_score_drop_from_baseline": result.get("score_drop_from_baseline"),
            "quality_task_passed": result.get("task_passed"),
            "quality_passed": result.get("quality_gate_passed"),
            "quality_evidence_path": result.get("evidence_path"),
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                _add(
                    blockers,
                    "run_quality_binding_mismatch",
                    f"Run {run_id} quality field {field!r} does not match audited profile evidence.",
                    run_id=run_id,
                    profile=profile,
                    field=field,
                )


def _audit_comparison_chain(
    root: Path,
    comparison: dict[str, object],
    summary_paths: list[Path],
    blockers: list[dict[str, object]],
) -> None:
    if len(summary_paths) < 2 or not all(path.exists() for path in summary_paths):
        _add(blockers, "comparison_source_mismatch", "Comparison sources do not match readable matrix summaries.")
        return
    run_rows = comparison.get("runs")
    if not isinstance(run_rows, list):
        _add(blockers, "invalid_comparison_source", "Comparison run sources must be a list.")
        return
    reported_paths: set[Path] = set()
    resolved_root = root.resolve()
    for row in run_rows:
        value = row.get("summary_path") if isinstance(row, dict) else None
        if not isinstance(value, str) or Path(value).is_absolute():
            _add(
                blockers,
                "invalid_comparison_source",
                "Comparison summary paths must be relative to the matrix output root.",
            )
            return
        resolved = (resolved_root / value).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            _add(
                blockers,
                "invalid_comparison_source",
                "Comparison summary paths must remain inside the matrix output root.",
            )
            return
        reported_paths.add(resolved)
    if reported_paths != {path.resolve() for path in summary_paths}:
        _add(
            blockers,
            "comparison_source_mismatch",
            "Comparison sources do not match matrix run summaries.",
        )
        return
    baseline_profile = comparison.get("baseline_profile", "baseline")
    comparison_mode = comparison.get("comparison_mode", "strict")
    if not isinstance(baseline_profile, str) or not isinstance(comparison_mode, str):
        _add(blockers, "invalid_comparison_identity", "Comparison baseline and mode must be strings.")
        return
    try:
        with tempfile.TemporaryDirectory(prefix="llm-accel-ranking-audit-") as temporary:
            recomputed = compare_run_summaries(
                summary_paths,
                Path(temporary),
                baseline_profile=baseline_profile,
                comparison_mode=comparison_mode,
                source_root=root,
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _add(blockers, "comparison_recompute_failed", f"Comparison could not be recomputed: {exc}")
        return
    if recomputed != comparison:
        _add(
            blockers,
            "comparison_evidence_mismatch",
            "comparison.json does not match a fresh comparison of the audited matrix summaries.",
        )


def _audit_matrix_chain(
    matrix: dict[str, object],
    plan: dict[str, object] | None,
    state: dict[str, object] | None,
    blockers: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if plan is None or state is None:
        return {}
    config_hashes = {matrix.get("config_sha256"), plan.get("config_sha256"), state.get("config_sha256")}
    if len(config_hashes) != 1 or None in config_hashes:
        _add(blockers, "matrix_config_mismatch", "Matrix plan, state, and summary config hashes must match.")
    plan_runs = plan.get("runs")
    state_runs = state.get("runs")
    summary_runs = matrix.get("runs")
    if not all(isinstance(value, list) for value in [plan_runs, state_runs, summary_runs]):
        _add(blockers, "invalid_matrix_chain", "Matrix plan, state, and summary must contain run lists.")
        return {}
    assert isinstance(plan_runs, list)
    assert isinstance(state_runs, list)
    assert isinstance(summary_runs, list)
    planned_by_id: dict[str, dict[str, object]] = {}
    plan_indices: set[int] = set()
    matrix_name = plan.get("matrix_name")
    for item in plan_runs:
        if not isinstance(item, dict):
            _add(blockers, "invalid_matrix_plan", "Every planned run must be an object.")
            continue
        run_id = item.get("run_id")
        plan_index = _integer(item.get("plan_index"))
        if not isinstance(run_id, str) or not run_id or run_id in planned_by_id:
            _add(blockers, "duplicate_planned_run", "Planned run IDs must be non-empty and unique.")
            continue
        if plan_index is None or plan_index in plan_indices:
            _add(blockers, "duplicate_plan_index", "Plan indices must be integers and unique.", run_id=run_id)
        else:
            plan_indices.add(plan_index)
        planned = dict(item)
        planned["matrix_name"] = matrix_name
        planned_by_id[run_id] = planned
    expected_ids = set(planned_by_id)
    for label, rows in [("matrix_state.json", state_runs), ("matrix_summary.json", summary_runs)]:
        ids = [row.get("run_id") for row in rows if isinstance(row, dict)]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids or len(ids) != len(rows):
            _add(
                blockers,
                "matrix_run_set_mismatch",
                f"{label} must contain each planned run exactly once.",
            )
    state_by_id = {
        str(item["run_id"]): item
        for item in state_runs
        if isinstance(item, dict) and isinstance(item.get("run_id"), str)
    }
    summary_by_id = {
        str(item["run_id"]): item
        for item in summary_runs
        if isinstance(item, dict) and isinstance(item.get("run_id"), str)
    }
    for run_id in sorted(expected_ids & set(state_by_id) & set(summary_by_id)):
        persisted = state_by_id[run_id]
        summarized = summary_by_id[run_id]
        for field in [
            "plan_index",
            "profile",
            "repetition",
            "status",
            "summary_path",
            "failed_request_count",
        ]:
            if persisted.get(field) != summarized.get(field):
                _add(
                    blockers,
                    "matrix_state_mismatch",
                    f"Run {run_id} field {field!r} differs between matrix state and summary.",
                    run_id=run_id,
                    field=field,
                )
    if matrix.get("planned_run_count") != len(plan_runs):
        _add(blockers, "planned_count_mismatch", "Matrix planned_run_count does not match matrix_plan.json.")
    return planned_by_id


def _audit_planned_state(
    state: dict[str, object],
    planned: dict[str, object],
    blockers: list[dict[str, object]],
) -> None:
    run_id = str(state.get("run_id", ""))
    for field in ["plan_index", "profile", "repetition"]:
        if state.get(field) != planned.get(field):
            _add(
                blockers,
                "planned_run_mismatch",
                f"Run {run_id} field {field!r} does not match matrix_plan.json.",
                run_id=run_id,
                field=field,
            )
    expected_summary = f"{run_id}/summary.json"
    if state.get("status") == "succeeded" and state.get("summary_path") != expected_summary:
        _add(
            blockers,
            "summary_path_mismatch",
            f"Run {run_id} summary path does not match its planned run ID.",
            run_id=run_id,
        )


def _audit_run_evidence(
    run_dir: Path,
    run_id: str,
    planned: dict[str, object] | None,
    blockers: list[dict[str, object]],
) -> dict[str, object]:
    evidence: dict[str, object] = {}
    summary = _read_object(run_dir / "summary.json", blockers, code="missing_run_summary", run_id=run_id)
    if summary is None:
        return evidence
    metadata = summary.get("metadata")
    metrics = summary.get("metrics")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        _add(blockers, "invalid_run_summary", f"Run {run_id} has invalid metadata or metrics.", run_id=run_id)
        return evidence
    evidence["execution_identity"] = {
        "profile": metadata.get("optimization_profile"),
        "model": metadata.get("model"),
        "backend": metadata.get("backend"),
        "base_url": metadata.get("base_url"),
        "endpoint_sha256": metadata.get("endpoint_sha256"),
    }
    if planned is not None:
        expected_metadata = {
            "matrix_name": planned.get("matrix_name"),
            "matrix_repetition": planned.get("repetition"),
            "matrix_randomized_order": planned.get("randomized_profile_order"),
            "optimization_profile": planned.get("profile"),
            "concurrency": planned.get("concurrency"),
            "requested_input_tokens": planned.get("input_tokens"),
            "output_tokens": planned.get("output_tokens"),
        }
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                _add(
                    blockers,
                    "planned_run_mismatch",
                    f"Run {run_id} metadata field {field!r} does not match matrix_plan.json.",
                    run_id=run_id,
                    field=field,
                )
        if evidence["execution_identity"] != planned.get("execution_identity"):
            _add(
                blockers,
                "planned_execution_identity_mismatch",
                f"Run {run_id} execution identity does not match matrix_plan.json.",
                run_id=run_id,
            )
    try:
        profile = load_bound_optimization_profile(
            run_dir,
            metadata.get("optimization_profile_spec"),
            require_artifact=True,
        )
        if profile is None:
            raise ValueError("optimization profile is missing")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        _add(
            blockers,
            "invalid_optimization_profile",
            f"Run {run_id} optimization profile is invalid: {exc}",
            run_id=run_id,
        )
    else:
        evidence["treatment_fingerprint"] = profile.treatment_fingerprint
        if planned is not None and profile.server_command_sha256 != planned.get(
            "server_command_sha256"
        ):
            _add(
                blockers,
                "planned_server_command_mismatch",
                f"Run {run_id} server command does not match matrix_plan.json.",
                run_id=run_id,
            )
        command_path = run_dir / "server_command.txt"
        if not command_path.exists():
            _add(blockers, "missing_server_command", f"Run {run_id} lacks server_command.txt.", run_id=run_id)
        elif hashlib.sha256(command_path.read_bytes()).hexdigest() != profile.server_command_sha256:
            _add(
                blockers,
                "server_command_mismatch",
                f"Run {run_id} server command does not match its optimization profile.",
                run_id=run_id,
            )
        if metadata.get("optimization_profile_fingerprint") != profile.semantic_fingerprint:
            _add(
                blockers,
                "profile_fingerprint_mismatch",
                f"Run {run_id} summary is not bound to its optimization profile.",
                run_id=run_id,
            )
        if metadata.get("environment_fingerprint") != profile.environment_fingerprint:
            _add(
                blockers,
                "environment_fingerprint_mismatch",
                f"Run {run_id} environment fingerprint differs from its optimization profile.",
                run_id=run_id,
            )
        recomputed_environment = environment_fingerprint(metadata)
        if recomputed_environment != profile.environment_fingerprint:
            _add(
                blockers,
                "environment_fingerprint_mismatch",
                f"Run {run_id} environment metadata does not reproduce its recorded fingerprint.",
                run_id=run_id,
            )

    schedule = metadata.get("request_schedule")
    if schedule != "open-loop":
        _add(
            blockers,
            "coordinated_omission_risk",
            f"Run {run_id} uses {schedule!r}; ranking evidence requires open-loop scheduling.",
            run_id=run_id,
        )
    client = metadata.get("client_configuration")
    if not isinstance(client, dict):
        _add(
            blockers,
            "missing_client_configuration",
            f"Run {run_id} lacks structured client configuration evidence.",
            run_id=run_id,
        )
    queue = metrics.get("queue_delay_ms")
    threshold = metadata.get("queue_delay_warning_ms")
    if not isinstance(queue, dict) or not _finite_non_negative(queue.get("p95")):
        _add(
            blockers,
            "missing_queue_delay",
            f"Run {run_id} lacks client queue-delay evidence.",
            run_id=run_id,
        )
    elif not _finite_non_negative(threshold):
        _add(
            blockers,
            "missing_saturation_threshold",
            f"Run {run_id} lacks a client saturation threshold.",
            run_id=run_id,
        )
    else:
        request_rate = client.get("request_rate_rps") if isinstance(client, dict) else None
        if not _finite_number(request_rate) or float(request_rate) <= 0:
            _add(
                blockers,
                "missing_request_rate",
                f"Run {run_id} lacks a positive open-loop request rate.",
                run_id=run_id,
            )
            effective_threshold = _effective_saturation_threshold(float(threshold), None)
        else:
            effective_threshold = _effective_saturation_threshold(
                float(threshold),
                float(request_rate),
            )
        if float(queue["p95"]) > effective_threshold:
            _add(
                blockers,
                "client_saturation",
                f"Run {run_id} queue-delay p95 exceeds the derived saturation ceiling.",
                run_id=run_id,
                effective_threshold_ms=effective_threshold,
            )
    raw_path = run_dir / "raw_requests.jsonl"
    raw_rows = _read_jsonl(raw_path, blockers, run_id)
    if raw_rows is not None:
        expected_count = _integer(metrics.get("request_count"))
        if expected_count != len(raw_rows):
            _add(
                blockers,
                "raw_request_count_mismatch",
                f"Run {run_id} raw request count does not match its summary.",
                run_id=run_id,
            )
        _audit_dispatch_rows(raw_rows, run_id, blockers)
    return evidence


def _audit_dispatch_rows(
    rows: list[dict[str, object]],
    run_id: str,
    blockers: list[dict[str, object]],
) -> None:
    for index, row in enumerate(rows, start=1):
        scheduled = row.get("scheduled_offset_ms")
        dispatch = row.get("dispatch_offset_ms")
        completed = row.get("completed_offset_ms")
        queue_delay = row.get("queue_delay_ms")
        total_latency = row.get("total_latency_ms")
        end_to_end = row.get("end_to_end_latency_ms")
        if not all(
            _finite_non_negative(value)
            for value in [scheduled, dispatch, completed, queue_delay, total_latency, end_to_end]
        ):
            _add(
                blockers,
                "missing_dispatch_evidence",
                f"Run {run_id} raw row {index} lacks finite schedule, dispatch, completion, or queue evidence.",
                run_id=run_id,
            )
            return
        if not math.isclose(
            float(end_to_end),
            float(completed) - float(scheduled),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            _add(
                blockers,
                "end_to_end_latency_mismatch",
                f"Run {run_id} raw row {index} end-to-end latency does not match completion minus schedule.",
                run_id=run_id,
            )
            return
        service_time = float(completed) - float(dispatch)
        if not math.isclose(
            float(total_latency),
            service_time,
            rel_tol=0.01,
            abs_tol=max(5.0, service_time * 0.01),
        ):
            _add(
                blockers,
                "total_latency_mismatch",
                f"Run {run_id} raw row {index} total latency does not match completion minus dispatch.",
                run_id=run_id,
            )
            return
        if float(scheduled) > float(dispatch) or float(dispatch) > float(completed):
            _add(
                blockers,
                "invalid_dispatch_order",
                f"Run {run_id} raw row {index} has invalid schedule and dispatch ordering.",
                run_id=run_id,
            )
            return
        if not math.isclose(
            float(queue_delay),
            float(dispatch) - float(scheduled),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            _add(
                blockers,
                "queue_delay_mismatch",
                f"Run {run_id} raw row {index} queue delay does not match dispatch minus schedule.",
                run_id=run_id,
            )
            return


def _audit_quality(
    root: Path,
    quality: object,
    profile_repetitions: dict[str, set[int]],
    run_execution_identities: dict[str, set[str]],
    blockers: list[dict[str, object]],
) -> None:
    if not isinstance(quality, list) or not quality:
        _add(blockers, "missing_quality_evidence", "Quality evidence is required for every profile.")
        return
    by_profile = {
        str(item.get("profile")): item
        for item in quality
        if isinstance(item, dict) and item.get("profile") is not None
    }
    fingerprints = set()
    recomputed_scores: dict[str, float] = {}
    recomputed_passed: dict[str, bool] = {}
    for profile in sorted(profile_repetitions):
        result = by_profile.get(profile)
        if result is None:
            _add(
                blockers,
                "missing_quality_profile",
                f"Profile {profile!r} lacks quality evidence.",
                profile=profile,
            )
            continue
        fingerprint = result.get("task_set_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            _add(
                blockers,
                "invalid_quality_fingerprint",
                f"Profile {profile!r} lacks a valid quality-suite fingerprint.",
                profile=profile,
            )
        else:
            fingerprints.add(fingerprint)
        if not _finite_number(result.get("score_drop_from_baseline")):
            _add(
                blockers,
                "missing_quality_delta",
                f"Profile {profile!r} lacks a quality delta from baseline.",
                profile=profile,
            )
        if result.get("quality_gate_passed") is not True:
            _add(
                blockers,
                "quality_gate_failed",
                f"Profile {profile!r} does not pass its quality gate.",
                profile=profile,
            )
        evidence_path = result.get("evidence_path")
        if not isinstance(evidence_path, str):
            _add(
                blockers,
                "missing_quality_result",
                f"Profile {profile!r} lacks a quality result path.",
                profile=profile,
            )
            continue
        relative_evidence = Path(evidence_path)
        if relative_evidence.is_absolute():
            _add(
                blockers,
                "invalid_quality_evidence_path",
                f"Profile {profile!r} quality evidence path must be relative to the matrix root.",
                profile=profile,
            )
            continue
        resolved_root = root.resolve()
        evidence_file = (resolved_root / relative_evidence).resolve()
        try:
            evidence_file.relative_to(resolved_root)
        except ValueError:
            _add(
                blockers,
                "invalid_quality_evidence_path",
                f"Profile {profile!r} quality evidence path escapes the matrix root.",
                profile=profile,
            )
            continue
        evidence_dir = evidence_file.parent
        try:
            validation = validate_run_dir(evidence_dir)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            _add(
                blockers,
                "invalid_quality_manifest",
                f"Profile {profile!r} quality evidence could not be validated: {exc}",
                profile=profile,
            )
            continue
        if not validation["valid"]:
            _add(
                blockers,
                "invalid_quality_manifest",
                f"Profile {profile!r} quality manifest is incomplete.",
                profile=profile,
                errors=validation["errors"],
            )
            continue
        evidence = _read_object(
            evidence_file,
            blockers,
            code="missing_quality_result",
        )
        if evidence is None:
            continue
        result_identity = result.get("execution_identity")
        evidence_identity = evidence.get("execution_identity")
        expected_identities = run_execution_identities.get(profile, set())
        expected_identity = None
        if len(expected_identities) == 1:
            expected_identity = json.loads(next(iter(expected_identities)))
        if (
            not isinstance(result_identity, dict)
            or evidence_identity != result_identity
            or expected_identity is None
            or result_identity != expected_identity
            or any(evidence.get(field) != result_identity.get(field) for field in ["model", "backend", "base_url"])
        ):
            _add(
                blockers,
                "quality_execution_identity_mismatch",
                f"Profile {profile!r} quality evidence is not bound to its measured execution identity.",
                profile=profile,
            )
        if evidence.get("task_set_sha256") != fingerprint:
            _add(
                blockers,
                "quality_result_mismatch",
                f"Profile {profile!r} quality result fingerprint does not match matrix evidence.",
                profile=profile,
            )
        if evidence.get("mean_score") != result.get("mean_score"):
            _add(
                blockers,
                "quality_result_mismatch",
                f"Profile {profile!r} quality score does not match matrix evidence.",
                profile=profile,
            )
        specs = _read_quality_jsonl(evidence_dir / "task_specs.jsonl", profile, blockers)
        outputs = _read_quality_jsonl(evidence_dir / "task_outputs.jsonl", profile, blockers)
        if specs is None or outputs is None:
            continue
        recomputed = _recompute_quality(profile, specs, outputs, evidence, blockers)
        if recomputed is None:
            continue
        recomputed_score, task_passed = recomputed
        recomputed_scores[profile] = recomputed_score
        recomputed_passed[profile] = task_passed
        canonical = json.dumps(specs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        recomputed_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if recomputed_fingerprint != fingerprint:
            _add(
                blockers,
                "quality_task_set_mismatch",
                f"Profile {profile!r} task_specs.jsonl does not reproduce its task-set fingerprint.",
                profile=profile,
            )
    if len(fingerprints) > 1:
        _add(
            blockers,
            "quality_suite_mismatch",
            "Profiles were evaluated with different quality suites.",
        )
    baseline_score = recomputed_scores.get("baseline")
    if baseline_score is not None:
        for profile, score in sorted(recomputed_scores.items()):
            result = by_profile[profile]
            expected_drop = baseline_score - score
            if not _numbers_close(result.get("score_drop_from_baseline"), expected_drop):
                _add(
                    blockers,
                    "quality_delta_mismatch",
                    f"Profile {profile!r} quality delta does not match recomputed evidence.",
                    profile=profile,
                )
            threshold = result.get("max_allowed_score_drop")
            expected_gate = (
                _finite_number(threshold)
                and recomputed_passed.get(profile) is True
                and expected_drop <= float(threshold)
            )
            if result.get("task_passed") is not recomputed_passed.get(profile):
                _add(
                    blockers,
                    "quality_task_pass_mismatch",
                    f"Profile {profile!r} task-pass state does not match recomputed evidence.",
                    profile=profile,
                )
            if result.get("quality_gate_passed") is not expected_gate:
                _add(
                    blockers,
                    "quality_gate_mismatch",
                    f"Profile {profile!r} quality gate does not match recomputed evidence.",
                    profile=profile,
                )


def _read_quality_jsonl(
    path: Path,
    profile: str,
    blockers: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    rows: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _add(
            blockers,
            "invalid_quality_artifact",
            f"Profile {profile!r} quality JSONL artifact could not be read: {exc}",
            profile=profile,
        )
        return None
    return rows


def _recompute_quality(
    profile: str,
    specs: list[dict[str, object]],
    outputs: list[dict[str, object]],
    evidence: dict[str, object],
    blockers: list[dict[str, object]],
) -> tuple[float, bool] | None:
    checks = evidence.get("checks")
    if not isinstance(checks, list) or len(specs) != len(outputs) or len(specs) != len(checks) or not specs:
        _add(
            blockers,
            "quality_chain_mismatch",
            f"Profile {profile!r} task specs, outputs, and checks must have the same non-zero length.",
            profile=profile,
        )
        return None
    scores: list[float] = []
    passed_count = 0
    seen_ids: set[str] = set()
    for index, (spec, output, check) in enumerate(zip(specs, outputs, checks, strict=True)):
        case_id = spec.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            _add(blockers, "quality_case_mismatch", f"Profile {profile!r} has invalid or duplicate task IDs.", profile=profile)
            return None
        seen_ids.add(case_id)
        if not isinstance(check, dict) or output.get("case_id") != case_id or check.get("case_id") != case_id:
            _add(blockers, "quality_case_mismatch", f"Profile {profile!r} task, output, and check IDs differ.", profile=profile)
            return None
        if output.get("output_index") != index or check.get("output_index") != index:
            _add(blockers, "quality_case_mismatch", f"Profile {profile!r} output indices are not canonical.", profile=profile)
            return None
        validator = spec.get("validator")
        if not isinstance(validator, dict):
            _add(blockers, "invalid_quality_validator", f"Profile {profile!r} has an invalid normalized validator.", profile=profile)
            return None
        output_text = output.get("output_text")
        if isinstance(output_text, str) and output.get("error") is None:
            validation = validate_output(output_text, validator)
            score = validation.score
            passed = validation.passed
        else:
            score = 0.0
            passed = False
        if not _numbers_close(check.get("score"), score) or check.get("passed") is not passed:
            _add(
                blockers,
                "quality_check_mismatch",
                f"Profile {profile!r} check for case {case_id!r} does not match the validator result.",
                profile=profile,
            )
            return None
        scores.append(score)
        passed_count += int(passed)
    mean_score = sum(scores) / len(scores)
    expected_values = {
        "task_count": len(scores),
        "passed_count": passed_count,
        "failed_count": len(scores) - passed_count,
        "passed": passed_count == len(scores),
    }
    for field, expected in expected_values.items():
        if evidence.get(field) != expected:
            _add(blockers, "quality_summary_mismatch", f"Profile {profile!r} field {field!r} does not match raw evidence.", profile=profile)
    if not _numbers_close(evidence.get("mean_score"), mean_score):
        _add(blockers, "quality_summary_mismatch", f"Profile {profile!r} mean score does not match raw evidence.", profile=profile)
    return mean_score, passed_count == len(scores)


def _read_object(
    path: Path,
    blockers: list[dict[str, object]],
    *,
    code: str,
    run_id: str | None = None,
) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _add(blockers, code, f"Could not read {path}: {exc}", run_id=run_id)
        return None
    if not isinstance(value, dict):
        _add(blockers, code, f"{path} must contain an object.", run_id=run_id)
        return None
    return value


def _read_jsonl(
    path: Path,
    blockers: list[dict[str, object]],
    run_id: str,
) -> list[dict[str, object]] | None:
    rows: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _add(
            blockers,
            "invalid_raw_trace",
            f"Run {run_id} raw request trace could not be read: {exc}",
            run_id=run_id,
        )
        return None
    return rows


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_non_negative(value: object) -> bool:
    return _finite_number(value) and float(value) >= 0


def _numbers_close(value: object, expected: float) -> bool:
    return _finite_number(value) and math.isclose(
        float(value),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _effective_saturation_threshold(
    configured_threshold_ms: float,
    request_rate_rps: float | None,
) -> float:
    if request_rate_rps is None:
        cadence_guard_ms = 10.0
    else:
        cadence_guard_ms = max(10.0, 0.1 * 1000.0 / request_rate_rps)
    return min(configured_threshold_ms, cadence_guard_ms)


def _add(
    blockers: list[dict[str, object]],
    code: str,
    message: str,
    **details: object,
) -> None:
    blocker = {"code": code, "message": message}
    blocker.update({key: value for key, value in details.items() if value is not None})
    blockers.append(blocker)


def _report(
    root: Path,
    blockers: list[dict[str, object]],
    warnings: list[str],
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "matrix_dir": str(root),
        "publishable_performance_ranking": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "evidence": evidence,
    }
