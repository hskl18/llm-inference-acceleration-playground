from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Mapping

from llm_accel.metrics.environment import (
    ENVIRONMENT_FINGERPRINT_FIELDS,
    environment_fingerprint,
)
from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.metrics.optimization_profile import (
    OPTIMIZATION_PROFILE_SCHEMA_VERSION,
    OptimizationProfile,
    OptimizationProfileMismatchError,
    fingerprint_payload,
    load_bound_optimization_profile,
)


MIN_RANKING_REQUESTS = 8
MIN_PROFILE_REPETITIONS = 3
COMPARISON_MODES = {"strict", "stratified"}


def compare_run_summaries(
    summary_paths: list[str | Path],
    output_dir: str | Path,
    *,
    baseline_profile: str = "baseline",
    comparison_mode: str = "strict",
    source_root: str | Path | None = None,
) -> dict[str, object]:
    if len(summary_paths) < 2:
        raise ValueError("at least two summary paths are required")
    if comparison_mode not in COMPARISON_MODES:
        raise ValueError("comparison_mode must be 'strict' or 'stratified'")
    if not baseline_profile.strip():
        raise ValueError("baseline_profile must be a non-empty string")

    out_dir = Path(output_dir)
    resolved_paths = [Path(path).resolve() for path in summary_paths]
    resolved_source_root = Path(source_root or out_dir.parent).resolve()
    relative_paths: list[str] = []
    for path in resolved_paths:
        try:
            relative_paths.append(path.relative_to(resolved_source_root).as_posix())
        except ValueError as exc:
            raise ValueError(
                f"summary path {path} must be contained by source root {resolved_source_root}"
            ) from exc
    rows = [
        _load_run_row(path, summary_identity=relative)
        for path, relative in zip(resolved_paths, relative_paths, strict=True)
    ]
    strata = _build_strata(rows, baseline_profile=baseline_profile)
    blockers: list[dict[str, object]] = []
    if len(set(resolved_paths)) != len(resolved_paths):
        blockers.append(
            _blocker(
                "duplicate_summary_path",
                "Each summary path may contribute at most one repetition.",
            )
        )
    warnings = _comparison_warnings(rows)
    if comparison_mode == "strict" and len(strata) > 1:
        fields = _differing_invariant_fields(rows)
        blockers.append(
            _blocker(
                "invariant_mismatch",
                "Runs are not comparable in strict mode because invariant metadata differs.",
                fields=fields,
            )
        )
        for field in fields:
            warnings.append(f"Runs are not comparable: {field} differs across summaries.")

    rankable_strata = [stratum for stratum in strata if stratum["ranking_allowed"]]
    ranking_allowed = bool(rankable_strata) if comparison_mode == "stratified" else (
        len(strata) == 1 and bool(rankable_strata) and not blockers
    )
    comparable = len(strata) == 1 and not blockers
    profile_aggregates = strata[0]["profile_aggregates"] if len(strata) == 1 else []
    report = {
        "comparison_schema_version": "0.2",
        "comparison_mode": comparison_mode,
        "baseline_profile": baseline_profile,
        "summary_count": len(rows),
        "runs": rows,
        "comparable": comparable,
        "cross_stratum_ranking_allowed": len(strata) == 1 and ranking_allowed,
        "ranking_allowed": ranking_allowed,
        "blockers": blockers,
        "warnings": _deduplicate(warnings),
        "strata": strata,
        "profile_aggregates": profile_aggregates,
        "notes": [
            "Only v0.2 summaries with structured optimization profiles can contribute valid repetitions.",
            "Optimization settings are treatment dimensions; workload, schedule, quality, client, and environment metadata are invariants.",
            "Relative performance uses the declared baseline aggregate, never input order.",
            "Stratified mode permits rankings within a stratum but never across strata.",
        ],
    }
    write_json(out_dir / "comparison.json", report)
    _write_markdown(out_dir / "comparison.md", report)
    write_run_manifest(
        out_dir,
        run_type="run_comparison",
        artifacts=["manifest.json", "comparison.json", "comparison.md"],
    )
    return report


def _load_run_row(path: Path, *, summary_identity: str) -> dict[str, object]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read summary {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"summary {path} must contain an object")
    metadata = summary.get("metadata")
    metrics = summary.get("metrics")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        raise ValueError(f"summary {path} metadata and metrics must be objects")

    evidence_blockers: list[dict[str, object]] = []
    if summary.get("schema_version") != "0.2":
        evidence_blockers.append(
            _blocker(
                "legacy_summary_schema",
                f"Summary schema {summary.get('schema_version')!r} cannot contribute v0.2 ranking evidence.",
            )
        )
    profile = _load_profile(path, metadata, evidence_blockers)
    effective_backend = profile.backend if profile else metadata.get("backend")
    if effective_backend == "mock":
        evidence_blockers.append(
            _blocker(
                "mock_evidence",
                "Mock runs are workflow evidence and cannot contribute to a performance ranking.",
            )
        )
    request_count = _integer(metrics.get("request_count"), default=0)
    failed_count = _integer(metrics.get("failed_count"), default=request_count)
    throughput = _nested_number(metrics, "throughput", "output_tokens_per_second")
    latency_p95 = _nested_number(metrics, "latency_ms", "p95")
    quality_score = metadata.get("quality_score")
    quality_score_drop = metadata.get("quality_score_drop_from_baseline")
    if metadata.get("quality_passed") is not True or metadata.get("quality_task_passed") is not True:
        evidence_blockers.append(
            _blocker("quality_gate_failed", "The run does not pass its recorded quality gate.")
        )
    if not _finite_score(quality_score):
        evidence_blockers.append(
            _blocker("invalid_quality_score", "A finite quality score between 0 and 1 is required.")
        )
    if not _finite_number(quality_score_drop):
        evidence_blockers.append(
            _blocker("missing_quality_delta", "A finite quality-score delta from baseline is required.")
        )
    if request_count < MIN_RANKING_REQUESTS:
        evidence_blockers.append(
            _blocker(
                "insufficient_requests",
                f"Only {request_count} measured requests are present; {MIN_RANKING_REQUESTS} are required.",
            )
        )
    if failed_count > 0:
        evidence_blockers.append(
            _blocker("failed_requests", f"The run contains {failed_count} failed requests.")
        )
    if throughput is None or throughput <= 0:
        evidence_blockers.append(
            _blocker("invalid_throughput", "A positive output token throughput is required.")
        )
    if latency_p95 is None or latency_p95 < 0:
        evidence_blockers.append(
            _blocker("invalid_latency", "A finite non-negative p95 latency is required.")
        )

    invariants = _comparison_invariants(metadata, metrics, profile, evidence_blockers)
    invariant_fingerprint = (
        fingerprint_payload(invariants)
        if not any(blocker["code"] == "missing_invariant" for blocker in evidence_blockers)
        else "invalid:" + fingerprint_payload({"summary_path": summary_identity})
    )
    profile_name = profile.name if profile else str(metadata.get("optimization_profile", "legacy"))
    profile_fingerprint = (
        profile.treatment_fingerprint
        if profile
        else "legacy:" + fingerprint_payload({"summary_path": summary_identity, "name": profile_name})
    )
    return {
        "summary_path": summary_identity,
        "schema_version": summary.get("schema_version"),
        "model": profile.model if profile else metadata.get("model"),
        "backend": profile.backend if profile else metadata.get("backend"),
        "backend_version": profile.backend_version if profile else metadata.get("backend_version"),
        "optimization_profile": profile_name,
        "optimization_profile_schema_version": profile.schema_version if profile else None,
        "optimization_profile_fingerprint": profile.semantic_fingerprint if profile else None,
        "treatment_fingerprint": profile_fingerprint,
        "invariant_fingerprint": invariant_fingerprint,
        "invariants": invariants,
        "concurrency": metadata.get("concurrency"),
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "request_count": request_count,
        "failed_count": failed_count,
        "latency_p95_ms": latency_p95 if latency_p95 is not None else 0.0,
        "output_tokens_per_second": throughput if throughput is not None else 0.0,
        "quality_score": quality_score,
        "quality_score_drop_from_baseline": quality_score_drop,
        "valid_repetition": not evidence_blockers,
        "evidence_blockers": evidence_blockers,
    }


def _load_profile(
    summary_path: Path,
    metadata: Mapping[str, object],
    blockers: list[dict[str, object]],
) -> OptimizationProfile | None:
    inline = metadata.get("optimization_profile_spec")
    if inline is None and isinstance(metadata.get("optimization_profile"), dict):
        inline = metadata.get("optimization_profile")
    try:
        profile = load_bound_optimization_profile(
            summary_path.parent,
            inline,
            require_artifact=bool(
                metadata.get("matrix_name")
                or metadata.get("optimization_profile_fingerprint")
            ),
        )
    except OptimizationProfileMismatchError as exc:
        blockers.append(_blocker("optimization_profile_mismatch", str(exc)))
        try:
            profile = load_bound_optimization_profile(
                summary_path.parent,
                None,
                require_artifact=True,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        blockers.append(_blocker("invalid_optimization_profile", str(exc)))
        return None
    if profile is None:
        blockers.append(
            _blocker(
                "missing_optimization_profile",
                "A structured v0.2 optimization profile is required for ranking.",
            )
        )
        return None
    if profile.schema_version != OPTIMIZATION_PROFILE_SCHEMA_VERSION:
        blockers.append(
            _blocker(
                "legacy_optimization_profile",
                f"Optimization profile schema {profile.schema_version!r} cannot be ranked as v0.2 evidence.",
            )
        )
    return profile


def _comparison_invariants(
    metadata: Mapping[str, object],
    metrics: Mapping[str, object],
    profile: OptimizationProfile | None,
    blockers: list[dict[str, object]],
) -> dict[str, object]:
    schedule = _first_present(metadata, "request_schedule", "workload_schedule", "schedule")
    client = _first_present(metadata, "client_configuration", "client_config", "load_generator")
    quality = _quality_comparison_identity(
        _first_present(metadata, "quality_gate", "quality_gate_fingerprint")
    )
    values: dict[str, object] = {
        "backend": profile.backend if profile else metadata.get("backend"),
        "backend_version": profile.backend_version if profile else metadata.get("backend_version"),
        "model": profile.model if profile else metadata.get("model"),
        "model_revision": profile.model_revision if profile else metadata.get("model_revision"),
        "tokenizer": profile.tokenizer if profile else metadata.get("tokenizer"),
        "tokenizer_revision": (
            profile.tokenizer_revision if profile else metadata.get("tokenizer_revision")
        ),
        "token_count_method": metadata.get("token_count_method"),
        "api_kind": metadata.get("api_kind"),
        "stream": metadata.get("stream"),
        "workload_mode": metadata.get("workload_mode"),
        "workload_fingerprint": metadata.get("workload_fingerprint"),
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "concurrency": metadata.get("concurrency"),
        "warmup_count": metadata.get("warmup_count"),
        "request_count": metrics.get("request_count"),
        "request_schedule": schedule,
        "client_configuration": client,
        "quality_gate": quality,
        "environment_fingerprint": (
            profile.environment_fingerprint if profile else metadata.get("environment_fingerprint")
        ),
    }
    missing = [key for key, value in values.items() if value is None or value == ""]
    if missing:
        blockers.append(
            _blocker(
                "missing_invariant",
                "Required comparison invariant metadata is missing.",
                fields=missing,
            )
        )
    metadata_environment = metadata.get("environment_fingerprint")
    if profile and metadata_environment not in {None, profile.environment_fingerprint}:
        blockers.append(
            _blocker(
                "environment_fingerprint_mismatch",
                "Summary and optimization profile environment fingerprints differ.",
            )
        )
    if all(field in metadata for field in ENVIRONMENT_FINGERPRINT_FIELDS):
        recomputed_environment = environment_fingerprint(dict(metadata))
        if metadata_environment != recomputed_environment:
            blockers.append(
                _blocker(
                    "environment_fingerprint_mismatch",
                    "Summary environment metadata does not reproduce its recorded fingerprint.",
                )
            )
    return values


def _quality_comparison_identity(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {
        key: item
        for key, item in value.items()
        if key != "execution_identity"
    }


def _build_strata(
    rows: list[dict[str, object]],
    *,
    baseline_profile: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["invariant_fingerprint"])].append(row)
    return [
        _build_stratum(fingerprint, grouped[fingerprint], baseline_profile=baseline_profile)
        for fingerprint in sorted(grouped)
    ]


def _build_stratum(
    invariant_fingerprint: str,
    rows: list[dict[str, object]],
    *,
    baseline_profile: str,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    warnings: list[str] = []
    profiles: dict[str, list[dict[str, object]]] = defaultdict(list)
    names_to_fingerprints: dict[str, set[str]] = defaultdict(set)
    fingerprints_to_names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        fingerprint = str(row["treatment_fingerprint"])
        name = str(row["optimization_profile"])
        profiles[fingerprint].append(row)
        names_to_fingerprints[name].add(fingerprint)
        fingerprints_to_names[fingerprint].add(name)
    for name, fingerprints in sorted(names_to_fingerprints.items()):
        if len(fingerprints) > 1:
            blockers.append(
                _blocker(
                    "profile_name_collision",
                    f"Optimization profile {name!r} maps to multiple treatment fingerprints.",
                )
            )
    for fingerprint, names in sorted(fingerprints_to_names.items()):
        if len(names) > 1:
            blockers.append(
                _blocker(
                    "profile_alias_collision",
                    "One treatment fingerprint is labeled as multiple optimization profiles.",
                    profiles=sorted(names),
                )
            )
    if len(profiles) < 2:
        blockers.append(
            _blocker(
                "missing_treatment",
                "A ranking stratum requires a baseline and at least one distinct treatment fingerprint.",
            )
        )

    aggregates = []
    for fingerprint, profile_rows in sorted(
        profiles.items(), key=lambda item: (str(item[1][0]["optimization_profile"]), item[0])
    ):
        name = str(profile_rows[0]["optimization_profile"])
        valid_rows = [row for row in profile_rows if row["valid_repetition"]]
        if len(valid_rows) < MIN_PROFILE_REPETITIONS:
            blockers.append(
                _blocker(
                    "insufficient_repetitions",
                    f"Optimization profile {name!r} has {len(valid_rows)} valid repetitions; {MIN_PROFILE_REPETITIONS} are required.",
                    profile=name,
                )
            )
        invalid_count = len(profile_rows) - len(valid_rows)
        if invalid_count:
            warnings.append(
                f"Optimization profile {name!r} has {invalid_count} invalid repetitions that were excluded from aggregates."
            )
        throughput = [float(row["output_tokens_per_second"]) for row in valid_rows]
        latency = [float(row["latency_p95_ms"]) for row in valid_rows]
        aggregates.append(
            {
                "optimization_profile": name,
                "treatment_fingerprint": fingerprint,
                "repetitions": len(profile_rows),
                "valid_repetitions": len(valid_rows),
                "invalid_repetitions": invalid_count,
                "output_tokens_per_second": _distribution(throughput) if throughput else None,
                "latency_p95_ms": _distribution(latency) if latency else None,
                "relative_to_baseline": None,
            }
        )

    baseline_aggregates = [
        aggregate
        for aggregate in aggregates
        if aggregate["optimization_profile"] == baseline_profile
    ]
    if len(baseline_aggregates) != 1:
        blockers.append(
            _blocker(
                "baseline_not_unique",
                f"Declared baseline profile {baseline_profile!r} must resolve to exactly one treatment fingerprint.",
            )
        )
    else:
        baseline_distribution = baseline_aggregates[0]["output_tokens_per_second"]
        baseline_mean = (
            float(baseline_distribution["mean"])
            if isinstance(baseline_distribution, dict)
            else 0.0
        )
        if baseline_mean <= 0:
            blockers.append(
                _blocker("invalid_baseline", "Declared baseline has no positive valid throughput aggregate.")
            )
        else:
            for aggregate in aggregates:
                distribution = aggregate["output_tokens_per_second"]
                if isinstance(distribution, dict):
                    aggregate["relative_to_baseline"] = float(distribution["mean"]) / baseline_mean

    aggregates.sort(
        key=lambda aggregate: (
            aggregate["optimization_profile"] != baseline_profile,
            str(aggregate["optimization_profile"]),
            str(aggregate["treatment_fingerprint"]),
        )
    )
    return {
        "invariant_fingerprint": invariant_fingerprint,
        "invariants": rows[0]["invariants"],
        "run_count": len(rows),
        "run_paths": sorted(str(row["summary_path"]) for row in rows),
        "ranking_allowed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "profile_aggregates": aggregates,
    }


def _comparison_warnings(rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        for blocker in row["evidence_blockers"]:  # type: ignore[union-attr]
            code = blocker.get("code") if isinstance(blocker, dict) else None
            if code == "insufficient_requests":
                warnings.append(
                    f"Run {row['summary_path']} has only {row['request_count']} measured requests; ranking is not justified."
                )
            elif code == "failed_requests":
                warnings.append(
                    f"Run {row['summary_path']} has failed requests; it is not a valid repetition."
                )
            elif code in {
                "missing_optimization_profile",
                "invalid_optimization_profile",
                "legacy_optimization_profile",
                "legacy_summary_schema",
                "missing_invariant",
            }:
                warnings.append(
                    f"Run {row['summary_path']} lacks complete v0.2 comparison evidence and remains inspection-only."
                )
    return warnings


def _differing_invariant_fields(rows: list[dict[str, object]]) -> list[str]:
    keys = sorted({key for row in rows for key in row["invariants"]})  # type: ignore[union-attr]
    return [
        key
        for key in keys
        if len({_canonical(row["invariants"].get(key)) for row in rows}) > 1  # type: ignore[union-attr]
    ]


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "stddev": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    blockers = report.get("blockers", [])
    warnings = report.get("warnings", [])
    lines = [
        "# Run Comparison",
        "",
        f"- Mode: `{report['comparison_mode']}`",
        f"- Declared baseline: `{report['baseline_profile']}`",
        f"- Comparable as one stratum: `{report['comparable']}`",
        f"- Ranking allowed: `{report['ranking_allowed']}`",
        "",
        "## Blockers",
        "",
        *(
            [f"- `{item['code']}`: {item['message']}" for item in blockers]  # type: ignore[index]
            if blockers
            else ["- None"]
        ),
        "",
        "## Warnings",
        "",
        *([f"- {warning}" for warning in warnings] if warnings else ["- None"]),
        "",
    ]
    for index, stratum in enumerate(report["strata"], start=1):  # type: ignore[index]
        lines.extend(
            [
                f"## Stratum {index}",
                "",
                f"- Invariant fingerprint: `{stratum['invariant_fingerprint']}`",
                f"- Ranking allowed: `{stratum['ranking_allowed']}`",
                "",
                "| Profile | Valid repetitions | Total repetitions | Mean output tokens/sec | Relative to baseline | Mean p95 latency ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for aggregate in stratum["profile_aggregates"]:
            throughput = aggregate["output_tokens_per_second"] or {}
            latency = aggregate["latency_p95_ms"] or {}
            relative = aggregate["relative_to_baseline"]
            relative_text = f"{relative:.3f}" if isinstance(relative, (int, float)) else "n/a"
            lines.append(
                f"| `{aggregate['optimization_profile']}` | {aggregate['valid_repetitions']} | "
                f"{aggregate['repetitions']} | {throughput.get('mean', 0.0):.3f} | "
                f"{relative_text} | {latency.get('mean', 0.0):.3f} |"
            )
        if stratum["blockers"]:
            lines.extend(["", "### Stratum blockers", ""])
            lines.extend(
                f"- `{item['code']}`: {item['message']}" for item in stratum["blockers"]
            )
        lines.append("")
    lines.extend(
        [
            "Rankings are valid only within a stratum.",
            "Legacy and incomplete summaries remain visible but never count as valid repetitions.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _blocker(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def _first_present(metadata: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _integer(value: object, *, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _nested_number(payload: Mapping[str, object], group: str, field: str) -> float | None:
    nested = payload.get(group)
    if not isinstance(nested, dict):
        return None
    value = nested.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == float(value)
        and float(value) not in {float("inf"), float("-inf")}
    )


def _finite_score(value: object) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deduplicate(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
