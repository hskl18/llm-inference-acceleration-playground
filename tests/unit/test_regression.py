from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.cli import main
from llm_accel.metrics.environment import environment_fingerprint
from llm_accel.reports.regression import _compare_metric, compare_regression


THROUGHPUT = ("throughput", "output_tokens_per_second")


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    run_dir = tmp_path_factory.mktemp("mock-run")
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=run_dir,
        request_count=2,
    )
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _group(
    root: Path,
    name: str,
    template: dict[str, object],
    values: list[float],
    *,
    path: tuple[str, ...] = THROUGHPUT,
    **metadata: object,
) -> list[str]:
    """Write one summary per value, each a copy of the template with that metric and metadata."""
    paths = []
    for index, value in enumerate(values):
        summary = copy.deepcopy(template)
        target = summary["metrics"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        summary["metadata"].update(metadata)
        summary["metadata"]["environment_fingerprint"] = environment_fingerprint(summary["metadata"])
        file = root / f"{name}-{index}" / "summary.json"
        file.parent.mkdir(parents=True)
        file.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(str(file))
    return paths


def _cli(baseline: list[str], candidate: list[str], output_dir: Path, *extra: str) -> int:
    argv = ["report", "regression", "--output-dir", str(output_dir), *extra]
    for path in baseline:
        argv += ["--baseline", path]
    for path in candidate:
        argv += ["--candidate", path]
    return main(argv)


def _row(report: dict[str, object], path: tuple[str, ...]) -> dict[str, object]:
    return next(row for row in report["metrics"] if tuple(row["metric"]) == path)


def test_a_throughput_drop_beyond_the_threshold_regresses_and_fails_the_command(tmp_path, template) -> None:
    baseline = _group(tmp_path, "base", template, [100.0, 101.0, 99.0])
    candidate = _group(tmp_path, "cand", template, [80.0, 81.0, 79.0])

    assert _cli(baseline, candidate, tmp_path / "out") == 1

    report = json.loads((tmp_path / "out" / "regression.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "regressed"
    assert report["regressed_metrics"] == ["throughput.output_tokens_per_second"]
    assert _row(report, THROUGHPUT)["change"] == pytest.approx(-20.0)
    assert (tmp_path / "out" / "regression.md").exists()


def test_identical_groups_show_no_material_change(tmp_path, template) -> None:
    baseline = _group(tmp_path, "base", template, [100.0, 101.0, 99.0])
    candidate = _group(tmp_path, "cand", template, [100.0, 101.0, 99.0])

    assert _cli(baseline, candidate, tmp_path / "out") == 0

    report = json.loads((tmp_path / "out" / "regression.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "no_material_change"


def test_noise_wider_than_the_threshold_is_inconclusive(tmp_path, template) -> None:
    report = compare_regression(
        _group(tmp_path, "base", template, [80.0, 100.0, 120.0]),
        _group(tmp_path, "cand", template, [85.0, 95.0, 105.0]),
        tmp_path / "out",
    )

    assert _row(report, THROUGHPUT)["verdict"] == "inconclusive"
    assert report["verdict"] == "inconclusive"


def test_more_than_thirty_degrees_of_freedom_keep_a_finite_t_quantile() -> None:
    # With the normal quantile this interval's worse end clears 10%; with t at 32 df it does not.
    baseline = [92.0 + step for step in range(17)]
    candidate = [105.5 + step for step in range(17)]

    row = _compare_metric(
        ("latency_ms", "p50"),
        baseline,
        candidate,
        threshold_percent=10.0,
        rate_threshold_points=5.0,
    )

    assert row["degrees_of_freedom"] == pytest.approx(32.0)
    assert row["worsening_ci95"][0] < 10.0
    assert row["verdict"] == "inconclusive"


def test_two_repetitions_are_too_few_for_a_verdict(tmp_path, template) -> None:
    report = compare_regression(
        _group(tmp_path, "base", template, [100.0, 101.0]),
        _group(tmp_path, "cand", template, [50.0, 51.0]),
        tmp_path / "out",
    )

    assert _row(report, THROUGHPUT)["verdict"] == "insufficient_repetitions"
    assert report["verdict"] == "insufficient_repetitions"


def test_a_backend_upgrade_is_comparable_only_when_the_version_may_vary(tmp_path, template) -> None:
    baseline = _group(tmp_path, "base", template, [100.0, 101.0, 99.0], backend_version="0.34.1")
    candidate = _group(tmp_path, "cand", template, [100.0, 101.0, 99.0], backend_version="0.35.0")

    strict = compare_regression(baseline, candidate, tmp_path / "strict")
    varied = compare_regression(baseline, candidate, tmp_path / "varied", vary=["backend_version"])

    assert strict["verdict"] == "not_comparable"
    assert "backend_version" in strict["blockers"][0]
    assert "environment_fingerprint" in strict["blockers"][0]
    assert varied["blockers"] == []
    assert varied["baseline"]["varied_values"] == {"backend_version": ["0.34.1"]}
    assert varied["candidate"]["varied_values"] == {"backend_version": ["0.35.0"]}


def test_a_different_slo_makes_goodput_incomparable(tmp_path, template) -> None:
    baseline = _group(tmp_path, "base", template, [100.0, 101.0, 99.0], slo={"ttft_ms": 1000.0})
    candidate = _group(tmp_path, "cand", template, [100.0, 101.0, 99.0], slo={"ttft_ms": 500.0})

    report = compare_regression(baseline, candidate, tmp_path / "out")

    assert report["verdict"] == "not_comparable"
    assert "slo" in report["blockers"][0]


def test_one_summary_listed_twice_in_a_group_is_rejected(tmp_path, template) -> None:
    once = _group(tmp_path, "base", template, [100.0])
    candidate = _group(tmp_path, "cand", template, [100.0, 101.0, 99.0])

    duplicated = compare_regression(once * 3, candidate, tmp_path / "dup")
    a_a = compare_regression(candidate, candidate, tmp_path / "aa")

    assert duplicated["verdict"] == "not_comparable"
    assert "baseline group lists the same summary" in duplicated["blockers"][0]
    assert a_a["blockers"] == []


def test_varied_dictionaries_compare_by_content_not_key_order(tmp_path, template) -> None:
    config = dict(template["metadata"]["client_configuration"])
    reordered = dict(reversed(list(config.items())))
    baseline = _group(tmp_path, "base", template, [100.0, 101.0], client_configuration=config)
    baseline += _group(tmp_path, "base-reordered", template, [99.0], client_configuration=reordered)
    candidate = _group(
        tmp_path,
        "cand",
        template,
        [100.0, 101.0, 99.0],
        client_configuration={**config, "timeout_seconds": 30.0},
    )

    report = compare_regression(baseline, candidate, tmp_path / "out", vary=["client_configuration"])

    assert report["blockers"] == []
    assert len(report["baseline"]["varied_values"]["client_configuration"]) == 1


def test_a_lost_first_answer_is_inconclusive_rather_than_an_improvement(tmp_path, template) -> None:
    baseline = _group(tmp_path, "base", template, [900.0, 950.0, 1000.0], path=("first_answer_ms", "p50"))
    candidate = _group(tmp_path, "cand", template, [None, None, None], path=("first_answer_ms", "p50"))

    report = compare_regression(baseline, candidate, tmp_path / "out")

    # The candidate stopped answering altogether, so the change cannot pass as no change.
    assert _row(report, ("first_answer_ms", "p50"))["verdict"] == "inconclusive"
    assert report["verdict"] == "inconclusive"
    assert "first_answer_ms.p50" in report["warnings"][-1]


def test_a_metric_missing_from_some_runs_of_both_groups_is_left_out(tmp_path, template) -> None:
    baseline = _group(tmp_path, "base", template, [900.0, None, 1000.0], path=("first_answer_ms", "p50"))
    candidate = _group(tmp_path, "cand", template, [950.0, 960.0, None], path=("first_answer_ms", "p50"))

    report = compare_regression(baseline, candidate, tmp_path / "out")

    assert _row(report, ("first_answer_ms", "p50"))["verdict"] == "missing"
    assert report["verdict"] == "no_material_change"


def test_varying_an_environment_field_needs_every_fingerprint_field(tmp_path, template) -> None:
    trimmed = copy.deepcopy(template)
    del trimmed["metadata"]["torch_version"]
    baseline = _group(tmp_path, "base", trimmed, [100.0, 101.0, 99.0], backend_version="0.34.1")
    candidate = _group(tmp_path, "cand", trimmed, [100.0, 101.0, 99.0], backend_version="0.35.0")

    report = compare_regression(baseline, candidate, tmp_path / "out", vary=["backend_version"])

    assert report["verdict"] == "not_comparable"
    assert any("missing: torch_version" in blocker for blocker in report["blockers"])


def test_an_unknown_vary_field_is_an_error(tmp_path, template) -> None:
    group = _group(tmp_path, "base", template, [100.0, 101.0, 99.0])

    with pytest.raises(ValueError, match="unknown: colour"):
        compare_regression(group, group, tmp_path / "out", vary=["colour"])


def test_an_aggregate_repeats_summary_is_rejected(tmp_path, template) -> None:
    aggregate = tmp_path / "repeats_summary.json"
    aggregate.write_text(json.dumps({"repeats": 3, "metadata": template["metadata"], "metrics": {}}), encoding="utf-8")
    group = _group(tmp_path, "base", template, [100.0, 101.0, 99.0])

    with pytest.raises(ValueError, match="aggregates repetitions"):
        compare_regression([str(aggregate)], group, tmp_path / "out")


def test_runs_that_share_no_metric_cannot_pass(tmp_path, template) -> None:
    empty = {**template, "metrics": {}}
    group = []
    for index in range(3):
        file = tmp_path / f"empty-{index}" / "summary.json"
        file.parent.mkdir()
        file.write_text(json.dumps(empty), encoding="utf-8")
        group.append(str(file))

    assert _cli(group, group, tmp_path / "out") == 1

    report = json.loads((tmp_path / "out" / "regression.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "not_comparable"
    assert "nothing to compare" in report["blockers"][-1]
