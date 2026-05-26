import json
import csv

import pytest

import llm_accel.benchmarks.latency as latency
from llm_accel.benchmarks.latency import run_latency_benchmark


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
