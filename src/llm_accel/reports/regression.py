"""Decide whether one deployment change made repeated runs better, worse, or neither.

A baseline group and a candidate group of repetitions are compared metric by metric. Each
verdict comes from a 95% Welch interval on the difference of the repetition means, judged
against a materiality threshold: a change is a regression only when the whole interval lies
beyond the threshold, and the interval has to fit inside the threshold to show no material
change. Anything else is inconclusive, which is the honest answer for noisy or few repetitions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

from llm_accel.benchmarks.repeats import REPEAT_METRIC_PATHS
from llm_accel.metrics.aggregation import t_quantile_95
from llm_accel.metrics.environment import ENVIRONMENT_FINGERPRINT_FIELDS, environment_fingerprint
from llm_accel.metrics.io import write_json, write_text_atomic
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.reports.comparison import MIN_PROFILE_REPETITIONS, _comparison_invariants


REGRESSION_METRIC_PATHS: tuple[tuple[str, ...], ...] = (*REPEAT_METRIC_PATHS, ("error_rate",))
# Rates are compared in percentage points: a relative change of a near-zero rate means nothing.
RATE_METRICS = {("goodput", "attainment"), ("error_rate",)}
HIGHER_IS_BETTER = {"throughput", "goodput"}
# Each repetition may draw its own prompts, as it must against a prompt-caching server; the
# sampling difference is part of the between-repetition variance the interval already covers.
PER_REPETITION_FIELDS = {"workload_fingerprint"}
# A run may declare no SLO or quality gate; their absence is a value, not a gap in the record.
DECLARED_ONLY_FIELDS = {"slo", "quality_gate"}


def compare_regression(
    baseline_paths: list[str | Path],
    candidate_paths: list[str | Path],
    output_dir: str | Path,
    *,
    vary: list[str] | None = None,
    threshold_percent: float = 10.0,
    rate_threshold_points: float = 5.0,
) -> dict[str, object]:
    if not baseline_paths or not candidate_paths:
        raise ValueError("both a baseline and a candidate summary are required")
    for name, value in (("threshold_percent", threshold_percent), ("rate_threshold_points", rate_threshold_points)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    out_dir = Path(output_dir)
    baseline = [_load_summary(path) for path in baseline_paths]
    candidate = [_load_summary(path) for path in candidate_paths]
    runs = baseline + candidate
    varied = sorted(set(vary or []))
    blockers: list[str] = []
    warnings: list[str] = []
    for group, members in (("baseline", baseline), ("candidate", candidate)):
        if len({run["path"] for run in members}) != len(members):
            # The same file in both groups is a valid A/A check; twice in one group fakes repetitions.
            blockers.append(f"The {group} group lists the same summary more than once.")

    first = runs[0]["summary"]
    allowed = (
        set(_comparison_invariants(first["metadata"], first["metrics"], None, []))  # type: ignore[index, arg-type]
        | set(ENVIRONMENT_FINGERPRINT_FIELDS)
        | {"slo"}
    )
    unknown = [field for field in varied if field not in allowed]
    if unknown:
        raise ValueError(f"--vary accepts invariant or environment fields; unknown: {', '.join(unknown)}")
    invariants = [_invariants(run["summary"], varied, blockers) for run in runs]
    differing = sorted(
        key for key in invariants[0] if len({_canonical(item.get(key)) for item in invariants}) > 1
    )
    if differing:
        blockers.append(
            "Runs differ in fields that must match: " + ", ".join(differing)
            + "; pass --vary for a field the change is meant to alter."
        )
    absent = sorted(
        key
        for key in invariants[0]
        if key not in DECLARED_ONLY_FIELDS and all(item.get(key) in (None, "") for item in invariants)
    )
    if absent:
        warnings.append(f"No run records {', '.join(absent)}, so the check cannot confirm they match.")
    group_values = {
        "baseline": _varied_values(baseline, varied),
        "candidate": _varied_values(candidate, varied),
    }
    for group, values in group_values.items():
        mixed = [field for field, seen in values.items() if len(seen) > 1]
        if mixed:
            blockers.append(f"The {group} group mixes values of {', '.join(mixed)}.")
    unchanged = [
        field for field in varied if group_values["baseline"][field] == group_values["candidate"][field]
    ]
    if unchanged:
        warnings.append(f"Varied fields have the same value in both groups: {', '.join(unchanged)}.")
    if any(_backend(run["summary"]) == "mock" for run in runs):
        warnings.append("Mock runs exercise the workflow; their verdict says nothing about hardware.")
    failed = [run["path"] for run in runs if _number(run["summary"], ("metrics", "failed_count"))]
    if failed:
        warnings.append(f"{len(failed)} runs contain failed requests; the error_rate row compares them.")

    metrics = [
        _compare_metric(
            path,
            [_number(run["summary"], ("metrics", *path)) for run in baseline],
            [_number(run["summary"], ("metrics", *path)) for run in candidate],
            threshold_percent=threshold_percent,
            rate_threshold_points=rate_threshold_points,
        )
        for path in REGRESSION_METRIC_PATHS
    ]
    one_sided = [".".join(row["metric"]) for row in metrics if row.get("one_sided")]
    if one_sided:
        warnings.append(f"Only one group reports {', '.join(one_sided)}; check why the other stopped or started.")
    if all(row["verdict"] in {"missing", "undefined"} or row.get("one_sided") for row in metrics):
        blockers.append("No metric is reported by every run, so there is nothing to compare.")
    verdict = _overall_verdict(metrics, blockers)
    report = {
        "regression_schema_version": "0.1",
        "verdict": verdict,
        "threshold_percent": threshold_percent,
        "rate_threshold_points": rate_threshold_points,
        "minimum_repetitions": MIN_PROFILE_REPETITIONS,
        "varied_fields": varied,
        "baseline": {
            "summaries": [_display_path(run["path"]) for run in baseline],
            "varied_values": {field: _decoded(seen) for field, seen in group_values["baseline"].items()},
        },
        "candidate": {
            "summaries": [_display_path(run["path"]) for run in candidate],
            "varied_values": {field: _decoded(seen) for field, seen in group_values["candidate"].items()},
        },
        "blockers": blockers,
        "warnings": warnings,
        "metrics": metrics,
        "regressed_metrics": [".".join(item["metric"]) for item in metrics if item["verdict"] == "regressed"],
        "notes": [
            "Each verdict uses a 95% Welch interval on the difference of repetition means.",
            "regressed: the whole interval is worse than the threshold; improved: the whole interval is better by more than it.",
            "no_material_change: the whole interval lies inside the threshold; inconclusive: anything else.",
        ],
    }
    write_json(out_dir / "regression.json", report)
    write_text_atomic(out_dir / "regression.md", _markdown(report))
    write_run_manifest(
        out_dir,
        run_type="regression_check",
        artifacts=["manifest.json", "regression.json", "regression.md"],
    )
    return report


def _load_summary(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    try:
        summary = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read summary {path}: {exc}") from exc
    if not isinstance(summary, dict) or not isinstance(summary.get("metadata"), dict) or not isinstance(
        summary.get("metrics"), dict
    ):
        raise ValueError(f"summary {path} needs metadata and metrics objects")
    if "repeats" in summary:
        raise ValueError(f"{path} aggregates repetitions; pass each repetition's summary.json instead")
    return {"path": resolved, "summary": summary}


def _invariants(summary: dict[str, object], varied: list[str], blockers: list[str]) -> dict[str, object]:
    metadata = summary["metadata"]
    metrics = summary["metrics"]
    found: list[dict[str, object]] = []
    values = _comparison_invariants(metadata, metrics, None, found)  # type: ignore[arg-type]
    values["slo"] = metadata.get("slo")  # type: ignore[union-attr]
    # Keep the integrity blockers (a fingerprint that does not reproduce). A field absent from every run
    # still has to match, so absence alone does not block: plain Ollama runs record no tokenizer.
    for blocker in found:
        if blocker["code"] != "missing_invariant":
            message = str(blocker["message"])
            if message not in blockers:
                blockers.append(message)
    varied_environment = [field for field in varied if field in ENVIRONMENT_FINGERPRINT_FIELDS]
    if varied_environment:
        # Recomputing from incomplete metadata would hash the gaps as None and hide a real difference.
        absent = [
            field
            for field in ENVIRONMENT_FINGERPRINT_FIELDS
            if field not in metadata and field not in varied_environment  # type: ignore[operator]
        ]
        if absent:
            message = f"Varying an environment field needs every fingerprint field; missing: {', '.join(absent)}."
            if message not in blockers:
                blockers.append(message)
        masked = {**metadata, **{field: None for field in varied_environment}}  # type: ignore[dict-item]
        values["environment_fingerprint"] = environment_fingerprint(masked)
    for field in (*varied, *PER_REPETITION_FIELDS):
        values.pop(field, None)
    return values


def _varied_values(runs: list[dict[str, object]], varied: list[str]) -> dict[str, set[str]]:
    """Each varied field's values per group, read through the same normalization as the invariants."""
    seen: dict[str, set[str]] = {field: set() for field in varied}
    for run in runs:
        summary = run["summary"]
        normalized = _comparison_invariants(summary["metadata"], summary["metrics"], None, [])  # type: ignore[index, arg-type]
        normalized["slo"] = summary["metadata"].get("slo")  # type: ignore[index, union-attr]
        for field in varied:
            # Environment-only fields such as git_commit are not invariants, so they come from metadata.
            value = normalized[field] if field in normalized else summary["metadata"].get(field)  # type: ignore[index, union-attr]
            seen[field].add(_canonical(value))
    return seen


def _decoded(values: set[str]) -> list[object]:
    return [json.loads(value) for value in sorted(values)]


def _compare_metric(
    path: tuple[str, ...],
    baseline: list[float | None],
    candidate: list[float | None],
    *,
    threshold_percent: float,
    rate_threshold_points: float,
) -> dict[str, object]:
    is_rate = path in RATE_METRICS
    higher_is_better = path[0] in HIGHER_IS_BETTER
    row: dict[str, object] = {
        "metric": list(path),
        "unit": "percentage points" if is_rate else "percent of baseline",
        "higher_is_better": higher_is_better,
        "threshold": rate_threshold_points if is_rate else threshold_percent,
        "baseline_count": sum(value is not None for value in baseline),
        "candidate_count": sum(value is not None for value in candidate),
    }
    if None in baseline or None in candidate:
        # A metric that one group reports in full and the other does not changed, so it cannot pass quietly.
        if (None in baseline) != (None in candidate):
            return {**row, "verdict": "inconclusive", "one_sided": True, "reason": "Only one group reports this metric in every run."}
        return {**row, "verdict": "missing", "reason": "Not every run reports this metric."}
    base = [float(value) for value in baseline]  # type: ignore[arg-type]
    cand = [float(value) for value in candidate]  # type: ignore[arg-type]
    base_mean, cand_mean = mean(base), mean(cand)
    row.update(baseline_mean=base_mean, candidate_mean=cand_mean)
    scale = 100.0 if is_rate else (100.0 / abs(base_mean) if base_mean else None)
    if scale is None:
        return {**row, "verdict": "undefined", "reason": "A zero baseline has no relative change."}
    difference = cand_mean - base_mean
    row["change"] = difference * scale
    if min(len(base), len(cand)) < MIN_PROFILE_REPETITIONS:
        return {
            **row,
            "verdict": "insufficient_repetitions",
            "reason": f"Each group needs {MIN_PROFILE_REPETITIONS} repetitions for an interval.",
        }
    base_var = stdev(base) ** 2 / len(base)
    cand_var = stdev(cand) ** 2 / len(cand)
    standard_error = math.sqrt(base_var + cand_var)
    denominator = (base_var**2 / (len(base) - 1)) + (cand_var**2 / (len(cand) - 1))
    # Welch-Satterthwaite degrees of freedom; zero spread in both groups leaves a point estimate.
    dof = (base_var + cand_var) ** 2 / denominator if denominator else float("inf")
    half_width = t_quantile_95(dof) * standard_error if math.isfinite(dof) else 0.0
    sign = -1.0 if higher_is_better else 1.0
    worse_low, worse_high = sorted(
        ((difference - half_width) * scale * sign, (difference + half_width) * scale * sign)
    )
    threshold = row["threshold"]
    if worse_low >= threshold:  # type: ignore[operator]
        verdict = "regressed"
    elif worse_high <= -threshold:  # type: ignore[operator]
        verdict = "improved"
    elif worse_low > -threshold and worse_high < threshold:  # type: ignore[operator]
        verdict = "no_material_change"
    else:
        verdict = "inconclusive"
    return {
        **row,
        "change_ci95": sorted(((difference - half_width) * scale, (difference + half_width) * scale)),
        "worsening_ci95": [worse_low, worse_high],
        "degrees_of_freedom": dof if math.isfinite(dof) else None,
        "verdict": verdict,
    }


def _overall_verdict(metrics: list[dict[str, object]], blockers: list[str]) -> str:
    if blockers:
        return "not_comparable"
    verdicts = {item["verdict"] for item in metrics} - {"missing", "undefined"}
    for verdict in ("regressed", "insufficient_repetitions", "inconclusive", "improved"):
        if verdict in verdicts:
            return verdict
    return "no_material_change"


def _markdown(report: dict[str, object]) -> str:
    rows = []
    for item in report["metrics"]:  # type: ignore[union-attr]
        unit = "pp" if item["unit"] == "percentage points" else "%"
        change = f"{item['change']:+.1f} {unit}" if "change" in item else "n/a"
        interval = (
            f"[{item['change_ci95'][0]:+.1f}, {item['change_ci95'][1]:+.1f}] {unit}"
            if "change_ci95" in item
            else "n/a"
        )
        means = (
            f"{item['baseline_mean']:.3f} | {item['candidate_mean']:.3f}"
            if "baseline_mean" in item
            else "n/a | n/a"
        )
        rows.append(f"| `{'.'.join(item['metric'])}` | {means} | {change} | {interval} | {item['verdict']} |")
    varied = ", ".join(report["varied_fields"]) or "none"  # type: ignore[arg-type]
    return "\n".join(
        [
            "# Regression Check",
            "",
            f"- Verdict: `{report['verdict']}`",
            f"- Threshold: `{report['threshold_percent']:g}%` of the baseline, `{report['rate_threshold_points']:g}` points for rates",
            f"- Fields allowed to differ: `{varied}`",
            f"- Baseline varied values: `{json.dumps(report['baseline']['varied_values'], sort_keys=True)}`",  # type: ignore[index]
            f"- Candidate varied values: `{json.dumps(report['candidate']['varied_values'], sort_keys=True)}`",  # type: ignore[index]
            f"- Repetitions: `{len(report['baseline']['summaries'])}` baseline, `{len(report['candidate']['summaries'])}` candidate",  # type: ignore[index]
            "",
            "| Metric | Baseline mean | Candidate mean | Change | 95% interval | Verdict |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "## Blockers",
            "",
            *([f"- {item}" for item in report["blockers"]] or ["- None"]),  # type: ignore[union-attr]
            "",
            "## Warnings",
            "",
            *([f"- {item}" for item in report["warnings"]] or ["- None"]),  # type: ignore[union-attr]
            "",
            "## Method",
            "",
            *[f"- {note}" for note in report["notes"]],  # type: ignore[union-attr]
            "",
        ]
    )


def _number(payload: object, path: tuple[str, ...]) -> float | None:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)) or not math.isfinite(float(current)):
        return None
    return float(current)


def _backend(summary: dict[str, object]) -> object:
    return summary["metadata"].get("backend")  # type: ignore[union-attr]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _display_path(path: object) -> str:
    """The path relative to the working directory, else under `~`, so reports never carry a home path."""
    resolved = Path(str(path))
    for root, prefix in ((Path.cwd().resolve(), ""), (Path.home().resolve(), "~/")):
        try:
            return prefix + resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()
