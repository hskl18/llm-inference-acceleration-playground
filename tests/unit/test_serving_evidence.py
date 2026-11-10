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
