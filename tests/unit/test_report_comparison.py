import json

from llm_accel.reports.comparison import compare_run_summaries


def _summary(path, concurrency: int, throughput: float, *, request_count: int = 8, hardware_label: str = "test-host") -> None:
    payload = {
        "metadata": {
            "model": "mock-model",
            "backend": "mock",
            "dtype": "fp16",
            "quantization": "none",
            "hardware_label": hardware_label,
            "concurrency": concurrency,
            "input_tokens": 128,
            "output_tokens": 64,
        },
        "metrics": {
            "request_count": request_count,
            "failed_count": 0,
            "latency_ms": {"p95": 10.0},
            "throughput": {"output_tokens_per_second": throughput},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_run_summaries_writes_report(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    out = tmp_path / "out"
    _summary(a, 1, 100.0)
    _summary(b, 2, 200.0)

    report = compare_run_summaries([a, b], out)

    assert report["summary_count"] == 2
    assert report["runs"][1]["relative_to_first"] == 2.0
    assert report["comparable"] is True
    assert report["ranking_allowed"] is True
    assert report["warnings"] == []
    assert (out / "comparison.json").exists()
    assert (out / "comparison.md").exists()
    assert (out / "manifest.json").exists()


def test_compare_run_summaries_warns_for_low_sample_and_incompatibility(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    out = tmp_path / "out"
    _summary(a, 1, 100.0, request_count=2, hardware_label="host-a")
    _summary(b, 2, 200.0, request_count=2, hardware_label="host-b")

    report = compare_run_summaries([a, b], out)

    assert report["comparable"] is False
    assert report["ranking_allowed"] is False
    assert any("hardware_label differs" in warning for warning in report["warnings"])
    assert any("ranking is not justified" in warning for warning in report["warnings"])
