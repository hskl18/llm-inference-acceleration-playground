import json
import csv

import pytest

import llm_accel.benchmarks.latency as latency
from llm_accel.benchmarks.latency import _RequestSpec, _partition_requests, run_latency_benchmark


def test_latency_benchmark_records_timeouts(tmp_path) -> None:
    output_dir = tmp_path / "timeout"

    summary = run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=output_dir,
        request_count=2,
        timeout_seconds=0.001,
    )

    assert summary["metrics"]["failed_count"] >= 1
    assert summary["metrics"]["timeout_count"] >= 1
    raw_lines = (output_dir / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines]
    assert any(row["completed"] is False and "timed out" in row["error"] for row in rows)
    csv_rows = list(csv.DictReader((output_dir / "raw_requests.csv").open(encoding="utf-8")))
    assert any(row["completed"] == "False" and "timed out" in row["error"] for row in csv_rows)


def test_latency_benchmark_preserves_raw_artifacts_when_report_generation_fails(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "report-failure"

    def fail_report(*_args, **_kwargs) -> None:
        raise RuntimeError("report generation failed")

    monkeypatch.setattr(latency, "write_summary_markdown", fail_report)

    with pytest.raises(RuntimeError, match="report generation failed"):
        run_latency_benchmark(
            base_url="mock://local",
            model="mock-model",
            concurrency=1,
            input_tokens=16,
            output_tokens=8,
            output_dir=output_dir,
            request_count=2,
        )

    assert (output_dir / "raw_requests.jsonl").exists()
    assert (output_dir / "raw_requests.csv").exists()
    assert (output_dir / "resolved_config.json").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert (output_dir / "summary.json").exists()
    assert not (output_dir / "summary.md").exists()


def test_open_loop_records_queue_delay_and_warns_on_client_saturation(tmp_path) -> None:
    output_dir = tmp_path / "open-loop"

    summary = run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=output_dir,
        request_count=4,
        request_schedule="open-loop",
        request_rate_rps=1000.0,
        queue_delay_warning_ms=1.0,
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["metadata"]["request_schedule"] == "open-loop"
    assert summary["metadata"]["request_rate_rps"] == 1000.0
    assert summary["metrics"]["queue_delay_ms"]["p95"] > 1.0
    assert any("Client saturation detected" in warning for warning in summary["warnings"])
    assert not any("Closed-loop scheduling" in warning for warning in summary["warnings"])
    assert [row["scheduled_offset_ms"] for row in rows] == [0.0, 1.0, 2.0, 3.0]
    assert all(row["scheduled_offset_ms"] <= row["dispatch_offset_ms"] <= row["completed_offset_ms"] for row in rows)
    assert all(row["end_to_end_latency_ms"] >= row["total_latency_ms"] for row in rows)


def test_closed_loop_warns_about_coordinated_omission(tmp_path) -> None:
    summary = run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=2,
        input_tokens=16,
        output_tokens=2,
        output_dir=tmp_path / "closed-loop",
        request_count=2,
    )

    assert summary["metadata"]["request_schedule"] == "closed-loop"
    assert summary["metadata"]["client_workers"] == 2
    assert any("coordinated omission" in warning for warning in summary["warnings"])


def test_multiprocess_load_generator_records_client_evidence(tmp_path) -> None:
    summary = run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=2,
        input_tokens=8,
        output_tokens=1,
        output_dir=tmp_path / "multiprocess",
        request_count=2,
        client_processes=2,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "multiprocess" / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["metadata"]["client_processes"] == 2
    assert summary["metadata"]["client_workers"] == 2
    assert summary["metrics"]["completed_count"] == 2
    assert [row["request_id"] for row in rows] == ["req-000001", "req-000002"]


def test_multiprocess_partition_is_proportional_to_worker_count() -> None:
    requests = [_RequestSpec(index=index, prompt="prompt", scheduled_offset_ms=0.0) for index in range(6)]

    groups, worker_counts = _partition_requests(requests, concurrency=3, client_processes=2)

    assert worker_counts == [2, 1]
    assert [len(group) for group in groups] == [4, 2]


@pytest.mark.parametrize("field", ["timeout_seconds", "queue_delay_warning_ms", "request_rate_rps"])
def test_latency_benchmark_rejects_nonfinite_client_values(tmp_path, field) -> None:
    kwargs = {
        "base_url": "mock://local",
        "model": "mock-model",
        "concurrency": 1,
        "input_tokens": 8,
        "output_tokens": 1,
        "output_dir": tmp_path / field,
        "request_count": 1,
        "request_schedule": "open-loop",
        "request_rate_rps": 1.0,
    }
    kwargs[field] = float("nan")

    with pytest.raises(ValueError):
        run_latency_benchmark(**kwargs)
