from __future__ import annotations

import math
from statistics import mean, pstdev, stdev
from typing import Iterable, Mapping

from llm_accel.metrics.schemas import RequestMetrics


# Two-sided 95% Student t quantiles by degrees of freedom; beyond 30 the normal value is used.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_T_95_LARGE_SAMPLE = 1.960
# One client process saturating a core means the load generator, not the server, sets the pace.
CLIENT_CPU_SATURATION = 0.8

SLO_FIELDS = ("ttft_ms", "tpot_ms", "end_to_end_latency_ms")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct < 0 or pct > 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    """Mean, spread, and a 95% confidence interval for the mean of repeated measurements.

    The interval assumes the repetitions are independent draws whose mean is approximately
    normal; with fewer than two samples there is no interval and the bounds equal the mean.
    """
    if not values:
        raise ValueError("distribution requires at least one value")
    sample_mean = mean(values)
    count = len(values)
    if count < 2:
        half_width = 0.0
        sample_stddev = 0.0
    else:
        sample_stddev = stdev(values)
        quantile = _T_95.get(count - 1, _T_95_LARGE_SAMPLE)
        half_width = quantile * sample_stddev / math.sqrt(count)
    return {
        "count": count,
        "mean": sample_mean,
        "stddev": pstdev(values),
        "sample_stddev": sample_stddev,
        "min": min(values),
        "max": max(values),
        "ci95_half_width": half_width,
        "ci95_low": sample_mean - half_width,
        "ci95_high": sample_mean + half_width,
    }


def normalize_slo(slo: Mapping[str, object] | None) -> dict[str, float] | None:
    """Keep only the declared, finite, positive SLO thresholds, or None when none are declared."""
    if not slo:
        return None
    thresholds: dict[str, float] = {}
    for field in SLO_FIELDS:
        value = slo.get(field)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"slo.{field} must be a number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"slo.{field} must be finite and positive")
        thresholds[field] = number
    return thresholds or None


def _goodput(
    completed: list[RequestMetrics],
    *,
    request_count: int,
    elapsed_seconds: float,
    slo: dict[str, float],
) -> dict[str, object]:
    """A request counts only when it completed and met every declared threshold."""
    good = [
        item
        for item in completed
        if all(getattr(item, field) <= threshold for field, threshold in slo.items())
    ]
    return {
        "slo": slo,
        "good_request_count": len(good),
        "attainment": len(good) / request_count if request_count else 0.0,
        "good_requests_per_second": len(good) / elapsed_seconds if elapsed_seconds else 0.0,
        "good_output_tokens_per_second": (
            sum(item.output_tokens for item in good) / elapsed_seconds if elapsed_seconds else 0.0
        ),
    }


def _overlap(intervals: list[tuple[float, float]]) -> tuple[int, float]:
    """Peak and time-weighted mean number of intervals open at once."""
    # An interval that ends when another starts does not overlap it, so ends sort first.
    events = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    if not events:
        return 0, 0.0
    open_now = peak = 0
    area = 0.0
    last = events[0][0]
    for at, delta in events:
        area += open_now * (at - last)
        last = at
        open_now += delta
        peak = max(peak, open_now)
    span = events[-1][0] - events[0][0]
    return peak, area / span if span > 0 else float(peak)


def _observed_concurrency(items: list[RequestMetrics]) -> dict[str, object]:
    """How many requests the client had in flight, and how many the server streamed at once.

    A request streams from its first token to its completion. Only requests with at least two
    streamed chunks count, which leaves out non-streaming calls, whose first token time is unknown.
    A streaming peak of 1 under higher client concurrency means the server answered one request
    at a time and the extra concurrency only queued.
    """
    # Failed requests still held a slot, so they count in flight; only completed ones streamed.
    in_flight_peak, _ = _overlap([(item.dispatch_offset_ms, item.completed_offset_ms) for item in items])
    streamed = [item for item in items if item.completed and item.inter_token_latencies_ms]
    # A start clamped to the completion keeps a skewed row from inverting its interval.
    streaming = [
        (min(item.dispatch_offset_ms + item.ttft_ms, item.completed_offset_ms), item.completed_offset_ms)
        for item in streamed
    ]
    streaming_peak, streaming_mean = _overlap(streaming)
    streamed_in_flight_peak, _ = _overlap([(item.dispatch_offset_ms, item.completed_offset_ms) for item in streamed])
    return {
        "in_flight_peak": in_flight_peak,
        "streamed_in_flight_peak": streamed_in_flight_peak if streaming else None,
        "streamed_request_count": len(streaming),
        "streaming_peak": streaming_peak if streaming else None,
        "streaming_mean": streaming_mean if streaming else None,
    }


def summarize_requests(
    records: Iterable[RequestMetrics],
    elapsed_seconds: float | None = None,
    slo: Mapping[str, object] | None = None,
) -> dict[str, object]:
    items = list(records)
    completed = [item for item in items if item.completed]
    failed = [item for item in items if not item.completed]
    timed_out = [
        item
        for item in failed
        if item.error and ("timed out" in item.error.lower() or "timeout" in item.error.lower())
    ]
    latencies = [item.total_latency_ms for item in completed]
    ttfts = [item.ttft_ms for item in completed]
    tpots = [item.tpot_ms for item in completed]
    queue_delays = [item.queue_delay_ms for item in items]
    end_to_end_latencies = [item.end_to_end_latency_ms for item in completed]
    output_tokens = sum(item.output_tokens for item in completed)
    total_wall_ms = sum(latencies)
    max_latency_ms = max(latencies, default=0.0)
    concurrency = max((item.concurrency for item in items), default=1)
    estimated_elapsed_s = max_latency_ms / 1000 if concurrency > 1 else total_wall_ms / 1000
    effective_elapsed_s = elapsed_seconds if elapsed_seconds is not None else estimated_elapsed_s
    inter_token_latencies = [gap for item in completed for gap in item.inter_token_latencies_ms]
    first_answers = [item.first_answer_ms for item in completed if item.first_answer_ms is not None]
    thresholds = normalize_slo(slo)

    return {
        "request_count": len(items),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "timeout_count": len(timed_out),
        "error_rate": len(failed) / len(items) if items else 0.0,
        "output_tokens": output_tokens,
        "latency_ms": {
            "mean": mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
        },
        "ttft_ms": {
            "mean": mean(ttfts) if ttfts else 0.0,
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
        },
        "tpot_ms": {
            "mean": mean(tpots) if tpots else 0.0,
            "p50": percentile(tpots, 50),
            "p95": percentile(tpots, 95),
            "p99": percentile(tpots, 99),
        },
        "queue_delay_ms": {
            "mean": mean(queue_delays) if queue_delays else 0.0,
            "p50": percentile(queue_delays, 50),
            "p95": percentile(queue_delays, 95),
            "p99": percentile(queue_delays, 99),
            "max": max(queue_delays, default=0.0),
        },
        "end_to_end_latency_ms": {
            "mean": mean(end_to_end_latencies) if end_to_end_latencies else 0.0,
            "p50": percentile(end_to_end_latencies, 50),
            "p95": percentile(end_to_end_latencies, 95),
            "p99": percentile(end_to_end_latencies, 99),
        },
        # Streamed chunk-arrival gaps pooled across completed requests; empty without streaming.
        "inter_token_latency_ms": {
            "sample_count": len(inter_token_latencies),
            "mean": mean(inter_token_latencies) if inter_token_latencies else 0.0,
            "p50": percentile(inter_token_latencies, 50),
            "p95": percentile(inter_token_latencies, 95),
            "p99": percentile(inter_token_latencies, 99),
            "max": max(inter_token_latencies, default=0.0),
        },
        # Time to the first answer text; it trails TTFT when a reasoning model streams reasoning first.
        # With no answer at all the statistics are null, so a lost answer never reads as a fast one.
        "first_answer_ms": {
            "sample_count": len(first_answers),
            "mean": mean(first_answers) if first_answers else None,
            "p50": percentile(first_answers, 50) if first_answers else None,
            "p95": percentile(first_answers, 95) if first_answers else None,
            "p99": percentile(first_answers, 99) if first_answers else None,
        },
        "observed_concurrency": _observed_concurrency(items),
        "goodput": (
            _goodput(
                completed,
                request_count=len(items),
                elapsed_seconds=effective_elapsed_s,
                slo=thresholds,
            )
            if thresholds
            else None
        ),
        "throughput": {
            "output_tokens_per_second": output_tokens / effective_elapsed_s if effective_elapsed_s else 0.0,
            "requests_per_second": len(completed) / effective_elapsed_s if effective_elapsed_s else 0.0,
            "estimated_elapsed_seconds": estimated_elapsed_s,
            "measured_elapsed_seconds": elapsed_seconds,
        },
    }
