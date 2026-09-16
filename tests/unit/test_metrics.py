import pytest

from llm_accel.metrics.aggregation import distribution, normalize_slo, percentile, summarize_requests
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


def test_summarize_requests_pools_inter_token_latencies_across_completed_requests() -> None:
    records = [
        RequestMetrics(
            "req-1", "m", "mock", 10, 3, 1, 5.0, 10.0, 25.0,
            inter_token_latencies_ms=(10.0, 20.0),
        ),
        RequestMetrics(
            "req-2", "m", "mock", 10, 2, 1, 5.0, 30.0, 35.0,
            inter_token_latencies_ms=(30.0,),
        ),
        RequestMetrics(
            "req-3", "m", "mock", 10, 0, 1, 0.0, 0.0, 0.0,
            completed=False, error="boom", inter_token_latencies_ms=(999.0,),
        ),
    ]

    itl = summarize_requests(records)["inter_token_latency_ms"]

    assert itl["sample_count"] == 3
    assert itl["p50"] == 20.0
    assert itl["max"] == 30.0


def test_goodput_counts_only_requests_meeting_every_declared_threshold() -> None:
    records = [
        RequestMetrics(
            "req-1", "m", "mock", 10, 4, 1, 50.0, 10.0, 100.0, end_to_end_latency_ms=100.0
        ),
        RequestMetrics(
            "req-2", "m", "mock", 10, 4, 1, 50.0, 40.0, 200.0, end_to_end_latency_ms=200.0
        ),
        RequestMetrics(
            "req-3", "m", "mock", 10, 4, 1, 200.0, 10.0, 300.0, end_to_end_latency_ms=300.0
        ),
        RequestMetrics(
            "req-4", "m", "mock", 10, 0, 1, 0.0, 0.0, 0.0, completed=False, error="boom"
        ),
    ]

    summary = summarize_requests(
        records,
        elapsed_seconds=2.0,
        slo={"ttft_ms": 100.0, "tpot_ms": 20.0},
    )
    goodput = summary["goodput"]

    assert goodput["good_request_count"] == 1
    assert goodput["attainment"] == 0.25
    assert goodput["good_requests_per_second"] == 0.5
    assert goodput["good_output_tokens_per_second"] == 2.0
    assert goodput["slo"] == {"ttft_ms": 100.0, "tpot_ms": 20.0}


def test_goodput_is_absent_without_a_declared_slo() -> None:
    records = [RequestMetrics("req-1", "m", "mock", 10, 4, 1, 1.0, 1.0, 10.0)]

    assert summarize_requests(records)["goodput"] is None


def test_normalize_slo_rejects_non_positive_thresholds() -> None:
    with pytest.raises(ValueError):
        normalize_slo({"ttft_ms": 0.0})
    with pytest.raises(ValueError):
        normalize_slo({"tpot_ms": float("inf")})
    assert normalize_slo({"unrelated": 5.0}) is None


def test_distribution_reports_a_student_t_confidence_interval() -> None:
    stats = distribution([10.0, 12.0, 14.0])

    assert stats["count"] == 3
    assert stats["mean"] == 12.0
    assert stats["sample_stddev"] == pytest.approx(2.0)
    # t(0.975, df=2) = 4.303, so the half width is 4.303 * 2 / sqrt(3).
    assert stats["ci95_half_width"] == pytest.approx(4.9686, abs=1e-3)
    assert stats["ci95_low"] == pytest.approx(12.0 - 4.9686, abs=1e-3)
    assert stats["ci95_high"] == pytest.approx(12.0 + 4.9686, abs=1e-3)


def test_distribution_of_one_sample_has_no_interval() -> None:
    stats = distribution([7.5])

    assert stats["ci95_low"] == 7.5 and stats["ci95_high"] == 7.5
    assert stats["sample_stddev"] == 0.0
