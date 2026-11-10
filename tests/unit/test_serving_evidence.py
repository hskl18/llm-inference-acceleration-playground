from __future__ import annotations

from llm_accel.metrics.aggregation import summarize_requests
from llm_accel.metrics.schemas import RequestMetrics


def _request(
    index: int,
    *,
    dispatch: float,
    ttft: float,
    completed_at: float,
    completed: bool = True,
    streamed: bool = True,
    first_answer: float | None = None,
) -> RequestMetrics:
    return RequestMetrics(
        request_id=f"req-{index}",
        model="m",
        backend="b",
        input_tokens=8,
        output_tokens=4 if completed else 0,
        concurrency=2,
        ttft_ms=ttft,
        tpot_ms=1.0,
        total_latency_ms=completed_at - dispatch,
        completed=completed,
        error=None if completed else "timed out",
        started_offset_ms=dispatch,
        completed_offset_ms=completed_at,
        dispatch_offset_ms=dispatch,
        inter_token_latencies_ms=(1.0, 1.0) if streamed and completed else (),
        first_answer_ms=first_answer,
    )


def test_requests_in_flight_together_but_streamed_one_after_another_show_a_streaming_peak_of_one() -> None:
    # Both dispatched at 0; the server streams the first until 100, then starts the second.
    records = [
        _request(1, dispatch=0.0, ttft=50.0, completed_at=100.0),
        _request(2, dispatch=0.0, ttft=150.0, completed_at=200.0),
    ]

    observed = summarize_requests(records)["observed_concurrency"]

    assert observed["in_flight_peak"] == 2
    assert observed["streamed_in_flight_peak"] == 2
    assert observed["streaming_peak"] == 1
    assert observed["streamed_request_count"] == 2


def test_overlapping_streams_show_the_server_interleaving_them() -> None:
    records = [
        _request(1, dispatch=0.0, ttft=10.0, completed_at=100.0),
        _request(2, dispatch=0.0, ttft=20.0, completed_at=110.0),
    ]

    observed = summarize_requests(records)["observed_concurrency"]

    assert observed["streaming_peak"] == 2
    assert 1.0 < observed["streaming_mean"] <= 2.0


def test_a_failed_request_still_counts_in_flight_but_not_as_streaming() -> None:
    records = [
        _request(1, dispatch=0.0, ttft=10.0, completed_at=100.0),
        _request(2, dispatch=0.0, ttft=0.0, completed_at=120.0, completed=False),
    ]

    observed = summarize_requests(records)["observed_concurrency"]

    assert observed["in_flight_peak"] == 2
    assert observed["streamed_in_flight_peak"] == 1
    assert observed["streamed_request_count"] == 1


def test_non_streaming_requests_have_no_streaming_figures() -> None:
    records = [_request(1, dispatch=0.0, ttft=100.0, completed_at=100.0, streamed=False)]

    observed = summarize_requests(records)["observed_concurrency"]

    assert observed["in_flight_peak"] == 1
    assert observed["streaming_peak"] is None
    assert observed["streaming_mean"] is None


def test_first_answer_statistics_cover_only_requests_that_answered() -> None:
    records = [
        _request(1, dispatch=0.0, ttft=10.0, completed_at=100.0, first_answer=60.0),
        _request(2, dispatch=0.0, ttft=10.0, completed_at=100.0, first_answer=None),
    ]

    first_answer = summarize_requests(records)["first_answer_ms"]

    assert first_answer["sample_count"] == 1
    assert first_answer["p50"] == 60.0


def test_first_answer_statistics_are_null_when_nothing_answered() -> None:
    records = [_request(1, dispatch=0.0, ttft=10.0, completed_at=100.0, first_answer=None)]

    first_answer = summarize_requests(records)["first_answer_ms"]

    assert first_answer == {"sample_count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
