from __future__ import annotations

import json

import pytest

from llm_accel.benchmarks.latency import (
    _build_run_warnings,
    _summarize_client_load,
    run_latency_benchmark,
)
from llm_accel.benchmarks.repeats import run_repeated_benchmark
from llm_accel.cli import main
from llm_accel.reports.validation import validate_run_dir


def test_repeated_benchmark_reports_means_with_confidence_intervals(tmp_path) -> None:
    payload = run_repeated_benchmark(
        run_latency_benchmark,
        repeats=3,
        output_dir=tmp_path / "repeated",
        base_url="mock://local",
        model="mock-model",
        concurrency=2,
        input_tokens=16,
        output_tokens=8,
        request_count=4,
    )

    assert payload["repeats"] == 3
    assert payload["repeat_dirs"] == ["repeat-01", "repeat-02", "repeat-03"]
    throughput = payload["metrics"]["throughput.output_tokens_per_second"]
    assert throughput["count"] == 3
    assert throughput["ci95_low"] <= throughput["mean"] <= throughput["ci95_high"]
    assert throughput["min"] <= throughput["mean"] <= throughput["max"]
    for repeat_dir in payload["repeat_dirs"]:
        assert (tmp_path / "repeated" / repeat_dir / "summary.json").exists()
    assert validate_run_dir(tmp_path / "repeated")["valid"] is True


def test_repeated_benchmark_requires_more_than_one_repetition(tmp_path) -> None:
    with pytest.raises(ValueError):
        run_repeated_benchmark(
            run_latency_benchmark,
            repeats=1,
            output_dir=tmp_path / "single",
            base_url="mock://local",
            model="mock-model",
            concurrency=1,
            input_tokens=8,
            output_tokens=4,
            request_count=2,
        )


def test_cli_repeats_and_slo_flags_write_goodput_and_repeat_aggregates(tmp_path) -> None:
    output_dir = tmp_path / "cli-repeats"

    assert (
        main(
            [
                "bench",
                "latency",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--request-count",
                "4",
                "--output-tokens",
                "8",
                "--repeats",
                "2",
                "--slo-ttft-ms",
                "1000",
                "--slo-tpot-ms",
                "1000",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    aggregate = json.loads((output_dir / "repeats_summary.json").read_text(encoding="utf-8"))
    assert aggregate["repeats"] == 2
    assert aggregate["metrics"]["goodput.attainment"]["mean"] == 1.0
    summary = json.loads((output_dir / "repeat-01" / "summary.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["slo"] == {"ttft_ms": 1000.0, "tpot_ms": 1000.0}
    assert summary["metrics"]["goodput"]["good_request_count"] == 4
    # The SLO is a reporting threshold, so it must not become a comparison invariant.
    assert "slo" not in summary["metadata"]["client_configuration"]


def test_cli_rejects_a_non_positive_slo_threshold(tmp_path) -> None:
    assert (
        main(
            [
                "bench",
                "latency",
                "--base-url",
                "mock://local",
                "--request-count",
                "1",
                "--slo-ttft-ms",
                "0",
                "--output-dir",
                str(tmp_path / "bad-slo"),
            ]
        )
        == 2
    )


def test_client_cpu_saturation_is_reported_and_warned() -> None:
    load = _summarize_client_load(
        client_cpu_seconds=1.9,
        elapsed_seconds=2.0,
        client_processes=1,
    )

    assert load["cpu_cores_used"] == pytest.approx(0.95)
    assert load["covers_all_client_processes"] is True

    warnings = _build_run_warnings(
        backend="openai-compatible",
        backend_version="1.0",
        backend_version_source="server_version_endpoint",
        stream=True,
        memory={"available": True},
        metrics={"failed_count": 0, "queue_delay_ms": {"p95": 0.0}},
        request_schedule="open-loop",
        client_processes=1,
        client_workers=4,
        queue_delay_warning_ms=10.0,
        request_count=4,
        unique_prompt_count=4,
        token_count_method="server_usage",
        completed_output_tokens=[8, 8, 8, 8],
        client_load=load,
    )

    assert any("0.95 CPU cores" in warning for warning in warnings)


def test_multiprocess_client_cpu_is_not_treated_as_a_saturation_signal() -> None:
    load = _summarize_client_load(
        client_cpu_seconds=1.9,
        elapsed_seconds=2.0,
        client_processes=4,
    )

    assert load["covers_all_client_processes"] is False

    warnings = _build_run_warnings(
        backend="openai-compatible",
        backend_version="1.0",
        backend_version_source="server_version_endpoint",
        stream=True,
        memory={"available": True},
        metrics={"failed_count": 0, "queue_delay_ms": {"p95": 0.0}},
        request_schedule="open-loop",
        client_processes=4,
        client_workers=8,
        queue_delay_warning_ms=10.0,
        request_count=8,
        unique_prompt_count=8,
        token_count_method="server_usage",
        completed_output_tokens=[8] * 8,
        client_load=load,
    )

    assert not any("CPU cores" in warning for warning in warnings)


def test_report_repeats_aggregates_runs_that_used_different_workloads(tmp_path) -> None:
    aggregate_dir = tmp_path / "prefix-arm"
    for repetition, seed in enumerate([1, 2, 3], start=1):
        # Each repetition uses a different prompt set, which is why it cannot be produced by
        # rerunning one identical configuration.
        run_latency_benchmark(
            base_url="mock://local",
            model="mock-model",
            concurrency=1,
            input_tokens=16,
            output_tokens=8,
            request_count=4,
            seed=seed,
            output_dir=aggregate_dir / f"repeat-{repetition:02d}",
        )

    assert (
        main(
            [
                "report",
                "repeats",
                "--run-dir",
                str(aggregate_dir / "repeat-01"),
                "--run-dir",
                str(aggregate_dir / "repeat-02"),
                "--run-dir",
                str(aggregate_dir / "repeat-03"),
                "--output-dir",
                str(aggregate_dir),
            ]
        )
        == 0
    )

    payload = json.loads((aggregate_dir / "repeats_summary.json").read_text(encoding="utf-8"))
    assert payload["repeat_dirs"] == ["repeat-01", "repeat-02", "repeat-03"]
    assert payload["metrics"]["ttft_ms.p50"]["count"] == 3
    assert validate_run_dir(aggregate_dir)["valid"] is True


def test_report_repeats_rejects_a_repetition_outside_the_aggregate_directory(tmp_path) -> None:
    outside = tmp_path / "outside"
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        request_count=2,
        output_dir=outside,
    )
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()

    assert (
        main(
            [
                "report",
                "repeats",
                "--run-dir",
                str(outside),
                "--run-dir",
                str(outside),
                "--output-dir",
                str(aggregate_dir),
            ]
        )
        == 2
    )


def test_repeats_validation_rejects_a_missing_repetition(tmp_path) -> None:
    aggregate_dir = tmp_path / "broken"
    payload = {
        "schema_version": "0.2",
        "repeats": 2,
        "repeat_dirs": ["repeat-01", "repeat-02"],
        "metadata": {},
        "metrics": {"ttft_ms.p50": {"count": 2, "mean": 1.0, "sample_stddev": 0.0, "min": 1.0, "max": 1.0, "ci95_low": 1.0, "ci95_high": 1.0}},
        "failed_request_count": 0,
        "warnings": [],
        "notes": [],
    }
    aggregate_dir.mkdir()
    (aggregate_dir / "repeats_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    result = validate_run_dir(aggregate_dir)

    assert result["valid"] is False
    assert any("missing repetition repeat-01" in error for error in result["errors"])
