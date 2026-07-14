from llm_accel.metrics.aggregation import percentile, summarize_requests
from llm_accel.metrics.schemas import RequestMetrics


def test_percentile_interpolates() -> None:
    assert percentile([10, 20, 30], 50) == 20
    assert percentile([10, 20], 50) == 15


def test_summarize_requests_counts_failures() -> None:
    records = [
        RequestMetrics("req-1", "m", "mock", 10, 5, 1, 1.0, 2.0, 9.0),
        RequestMetrics("req-2", "m", "mock", 10, 0, 1, 0.0, 0.0, 0.0, completed=False, error="boom"),
        RequestMetrics(
            "req-3",
            "m",
            "mock",
            10,
            0,
            1,
            0.0,
            0.0,
            1000.0,
            completed=False,
            error="request timed out after 1 seconds",
        ),
    ]

    summary = summarize_requests(records)

    assert summary["request_count"] == 3
    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 2
    assert summary["timeout_count"] == 1
    assert summary["error_rate"] == 2 / 3
    assert summary["ttft_ms"]["p99"] == 1.0
    assert summary["tpot_ms"]["p99"] == 2.0


def test_summarize_requests_uses_measured_elapsed_seconds() -> None:
    records = [
        RequestMetrics("req-1", "m", "mock", 10, 10, 2, 1.0, 1.0, 10.0),
        RequestMetrics("req-2", "m", "mock", 10, 10, 2, 1.0, 1.0, 10.0),
    ]

    summary = summarize_requests(records, elapsed_seconds=2.0)
    throughput = summary["throughput"]

    assert throughput["measured_elapsed_seconds"] == 2.0
    assert throughput["output_tokens_per_second"] == 10.0


def test_summarize_requests_includes_client_queue_and_end_to_end_latency() -> None:
    records = [
        RequestMetrics(
            "req-1",
            "m",
            "mock",
            10,
            5,
            1,
            1.0,
            2.0,
            9.0,
            scheduled_offset_ms=10.0,
            dispatch_offset_ms=15.0,
            queue_delay_ms=5.0,
            end_to_end_latency_ms=14.0,
        ),
        RequestMetrics(
            "req-2",
            "m",
            "mock",
            10,
            5,
            1,
            1.0,
            2.0,
            9.0,
            scheduled_offset_ms=20.0,
            dispatch_offset_ms=35.0,
            queue_delay_ms=15.0,
            end_to_end_latency_ms=24.0,
        ),
    ]

    summary = summarize_requests(records)

    assert summary["queue_delay_ms"]["mean"] == 10.0
    assert summary["queue_delay_ms"]["p95"] == 14.5
    assert summary["queue_delay_ms"]["max"] == 15.0
    assert summary["end_to_end_latency_ms"]["p50"] == 19.0
