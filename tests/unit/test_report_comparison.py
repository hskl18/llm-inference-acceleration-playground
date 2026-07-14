import json

import pytest

from llm_accel.metrics.environment import environment_fingerprint
from llm_accel.metrics.optimization_profile import (
    create_optimization_profile,
    write_optimization_profile,
)
from llm_accel.reports.comparison import compare_run_summaries


REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40


def _profile(
    name: str,
    *,
    backend: str = "vllm",
    environment: str = "environment-a",
    prefix_cache: bool = False,
    quantization: str = "none",
):
    flags = " --enable-prefix-caching" if prefix_cache else ""
    if quantization != "none":
        flags += f" --quantization {quantization}"
    return create_optimization_profile(
        name=name,
        backend=backend,
        backend_version="builtin" if backend == "mock" else "0.10.0",
        server_command=(
            "python -m vllm.entrypoints.openai.api_server --model model "
            f"--revision {REVISION} --dtype float16{flags}\n"
        ),
        model="model",
        model_revision=REVISION,
        tokenizer="model",
        tokenizer_revision=TOKENIZER_REVISION,
        dtype="float16",
        quantization=quantization,
        environment_fingerprint=environment,
        prefix_cache=prefix_cache,
    )


def _summary(
    path,
    throughput: float,
    *,
    profile=None,
    request_count: int = 8,
    failed_count: int = 0,
    workload_fingerprint: str = "workload-a",
    request_schedule=None,
    client_configuration=None,
    quality_gate=None,
    metadata_overrides=None,
) -> None:
    profile = profile or _profile("baseline")
    payload = {
        "schema_version": "0.2",
        "metadata": {
            "model": "model",
            "backend": "vllm",
            "backend_version": "0.10.0",
            "optimization_profile_spec": profile.to_dict(),
            "environment_fingerprint": profile.environment_fingerprint,
            "api_kind": "chat",
            "stream": True,
            "token_count_method": "tokenizers.encode(add_special_tokens=false)",
            "workload_mode": "fixed_prompts",
            "workload_fingerprint": workload_fingerprint,
            "concurrency": 8,
            "input_tokens": 128,
            "output_tokens": 64,
            "warmup_count": 5,
            "request_schedule": request_schedule
            or {"mode": "open_loop", "rate_per_second": 10.0, "seed": 42},
            "client_configuration": client_configuration
            or {"processes": 2, "workers_per_process": 4},
            "quality_gate": quality_gate or {"suite_fingerprint": "quality-a", "required": True},
            "quality_score": 1.0,
            "quality_score_drop_from_baseline": 0.0,
            "quality_task_passed": True,
            "quality_passed": True,
        },
        "metrics": {
            "request_count": request_count,
            "failed_count": failed_count,
            "latency_ms": {"p95": 10.0},
            "throughput": {"output_tokens_per_second": throughput},
        },
    }
    if metadata_overrides:
        payload["metadata"].update(metadata_overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _three_repetitions(tmp_path, profile, *, start: float):
    paths = []
    for repetition in range(3):
        path = tmp_path / f"{profile.name}-{repetition}.json"
        _summary(path, start + repetition * 10.0, profile=profile)
        paths.append(path)
    return paths


def test_compare_uses_declared_baseline_aggregate_not_input_order(tmp_path) -> None:
    baseline = _profile("baseline")
    prefix = _profile("prefix-cache", prefix_cache=True)
    paths = _three_repetitions(tmp_path, prefix, start=200.0)
    paths += _three_repetitions(tmp_path, baseline, start=100.0)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is True
    aggregates = report["profile_aggregates"]
    assert aggregates[0]["optimization_profile"] == "baseline"
    assert aggregates[0]["valid_repetitions"] == 3
    assert aggregates[0]["relative_to_baseline"] == 1.0
    assert aggregates[1]["relative_to_baseline"] == pytest.approx(210.0 / 110.0)
    assert (tmp_path / "out" / "comparison.json").exists()
    assert (tmp_path / "out" / "comparison.md").exists()


def test_comparison_rejects_inline_profile_that_differs_from_artifact(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    baseline = _profile("baseline")
    treatment = _profile("prefix-cache", prefix_cache=True)
    write_optimization_profile(run_dir, baseline)
    summary_path = run_dir / "summary.json"
    _summary(summary_path, 100.0, profile=treatment)
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    write_optimization_profile(control_dir, baseline)
    control_path = control_dir / "summary.json"
    _summary(control_path, 100.0, profile=baseline)

    report = compare_run_summaries([summary_path, control_path], tmp_path / "out")

    blockers = report["runs"][0]["evidence_blockers"]
    assert any(blocker["code"] == "optimization_profile_mismatch" for blocker in blockers)
    assert report["runs"][0]["optimization_profile_fingerprint"] == baseline.semantic_fingerprint


def test_legacy_summaries_remain_readable_but_cannot_rank(tmp_path) -> None:
    paths = []
    for repetition in range(3):
        path = tmp_path / f"legacy-{repetition}.json"
        payload = {
            "metadata": {
                "model": "model",
                "backend": "vllm",
                "optimization_profile": "baseline",
                "concurrency": 8,
                "input_tokens": 128,
                "output_tokens": 64,
            },
            "metrics": {
                "request_count": 8,
                "failed_count": 0,
                "latency_ms": {"p95": 10.0},
                "throughput": {"output_tokens_per_second": 100.0},
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is False
    assert all(run["valid_repetition"] is False for run in report["runs"])
    assert any("inspection-only" in warning for warning in report["warnings"])


def test_mock_profiles_never_contribute_valid_rankings(tmp_path) -> None:
    profile = _profile("baseline", backend="mock")
    paths = _three_repetitions(tmp_path, profile, start=100.0)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is False
    assert all(run["valid_repetition"] is False for run in report["runs"])
    assert all(
        any(blocker["code"] == "mock_evidence" for blocker in run["evidence_blockers"])
        for run in report["runs"]
    )


@pytest.mark.parametrize(
    ("field", "alternate"),
    [
        ("request_schedule", {"mode": "closed_loop", "concurrency": 8}),
        ("client_configuration", {"processes": 1, "workers_per_process": 8}),
        ("quality_gate", {"suite_fingerprint": "quality-b", "required": True}),
    ],
)
def test_strict_comparison_blocks_schedule_client_and_quality_mismatches(
    tmp_path, field, alternate
) -> None:
    baseline = _profile("baseline")
    paths = _three_repetitions(tmp_path, baseline, start=100.0)
    changed = tmp_path / "changed.json"
    kwargs = {field: alternate}
    _summary(changed, 120.0, profile=baseline, **kwargs)
    paths.append(changed)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is False
    assert any(blocker["code"] == "invariant_mismatch" for blocker in report["blockers"])
    assert len(report["strata"]) == 2


def test_stratified_mode_allows_rankings_within_each_environment(tmp_path) -> None:
    paths = []
    for environment, directory in [("environment-a", "a"), ("environment-b", "b")]:
        root = tmp_path / directory
        root.mkdir()
        baseline = _profile("baseline", environment=environment)
        treatment = _profile("prefix-cache", environment=environment, prefix_cache=True)
        paths += _three_repetitions(root, baseline, start=100.0)
        paths += _three_repetitions(root, treatment, start=150.0)

    report = compare_run_summaries(
        paths,
        tmp_path / "out",
        comparison_mode="stratified",
    )

    assert report["comparable"] is False
    assert report["ranking_allowed"] is True
    assert report["cross_stratum_ranking_allowed"] is False
    assert len(report["strata"]) == 2
    assert all(stratum["ranking_allowed"] for stratum in report["strata"])


def test_quantization_is_a_treatment_dimension(tmp_path) -> None:
    baseline = _profile("baseline")
    quantized = _profile("quantized-awq", quantization="awq")
    paths = _three_repetitions(tmp_path, baseline, start=100.0)
    paths += _three_repetitions(tmp_path, quantized, start=150.0)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is True
    assert len(report["strata"]) == 1
    assert [item["optimization_profile"] for item in report["profile_aggregates"]] == [
        "baseline",
        "quantized-awq",
    ]


def test_failed_runs_do_not_count_as_valid_repetitions(tmp_path) -> None:
    baseline = _profile("baseline")
    paths = _three_repetitions(tmp_path, baseline, start=100.0)
    _summary(paths[-1], 120.0, profile=baseline, failed_count=1)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is False
    aggregate = report["profile_aggregates"][0]
    assert aggregate["repetitions"] == 3
    assert aggregate["valid_repetitions"] == 2
    assert aggregate["invalid_repetitions"] == 1
    assert any(
        blocker["code"] == "insufficient_repetitions"
        for blocker in report["strata"][0]["blockers"]
    )


def test_duplicate_summary_paths_cannot_inflate_repetitions(tmp_path) -> None:
    baseline = _profile("baseline")
    treatment = _profile("prefix-cache", prefix_cache=True)
    baseline_path = tmp_path / "baseline.json"
    treatment_path = tmp_path / "treatment.json"
    _summary(baseline_path, 100.0, profile=baseline)
    _summary(treatment_path, 150.0, profile=treatment)

    report = compare_run_summaries(
        [baseline_path] * 3 + [treatment_path] * 3,
        tmp_path / "out",
    )

    assert report["ranking_allowed"] is False
    assert any(blocker["code"] == "duplicate_summary_path" for blocker in report["blockers"])


def test_same_profile_name_with_different_treatments_blocks_ranking(tmp_path) -> None:
    baseline = _profile("baseline")
    mislabeled = _profile("baseline", prefix_cache=True)
    control_dir = tmp_path / "control"
    treatment_dir = tmp_path / "treatment"
    control_dir.mkdir()
    treatment_dir.mkdir()
    paths = _three_repetitions(control_dir, baseline, start=100.0)
    paths += _three_repetitions(treatment_dir, mislabeled, start=150.0)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is False
    assert any(
        blocker["code"] == "profile_name_collision"
        for blocker in report["strata"][0]["blockers"]
    )


def test_missing_invariant_never_compares_equal(tmp_path) -> None:
    baseline = _profile("baseline")
    paths = _three_repetitions(tmp_path, baseline, start=100.0)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["metadata"]["workload_fingerprint"]
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = compare_run_summaries(paths, tmp_path / "out", comparison_mode="stratified")

    assert report["ranking_allowed"] is False
    assert len(report["strata"]) == 3
    assert all(run["valid_repetition"] is False for run in report["runs"])


def test_complete_environment_metadata_must_reproduce_fingerprint(tmp_path) -> None:
    environment = {
        "backend": "vllm",
        "backend_version": "0.10.0",
        "project_version": "0.2.0",
        "git_commit": "c" * 40,
        "python_version": "3.13.5",
        "operating_system": "test-os",
        "hardware_label": "test-host",
        "gpu_name": "test-gpu",
        "gpu_driver_version": "1.0",
        "cuda_version": "12.0",
        "cuda_driver_api_version": "12.0",
        "torch_version": "2.0",
    }
    fingerprint = environment_fingerprint(environment)
    baseline = _profile("baseline", environment=fingerprint)
    treatment = _profile("prefix-cache", environment=fingerprint, prefix_cache=True)
    paths = []
    for repetition, profile in enumerate([baseline] * 3 + [treatment] * 3):
        path = tmp_path / f"run-{repetition}.json"
        _summary(path, 100.0 + repetition, profile=profile, metadata_overrides=environment)
        paths.append(path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["metadata"]["gpu_name"] = "tampered-gpu"
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    report = compare_run_summaries(paths, tmp_path / "out", comparison_mode="stratified")

    assert report["ranking_allowed"] is False
    assert any(
        blocker["code"] == "environment_fingerprint_mismatch"
        for run in report["runs"]
        for blocker in run["evidence_blockers"]
    )
