from __future__ import annotations

from statistics import mean
from typing import Iterable

from llm_accel.metrics.schemas import RequestMetrics


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


def summarize_requests(records: Iterable[RequestMetrics], elapsed_seconds: float | None = None) -> dict[str, object]:
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
    output_tokens = sum(item.output_tokens for item in completed)
    total_wall_ms = sum(latencies)
    max_latency_ms = max(latencies, default=0.0)
    concurrency = max((item.concurrency for item in items), default=1)
    estimated_elapsed_s = max_latency_ms / 1000 if concurrency > 1 else total_wall_ms / 1000
    effective_elapsed_s = elapsed_seconds if elapsed_seconds is not None else estimated_elapsed_s

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
        "throughput": {
            "output_tokens_per_second": output_tokens / effective_elapsed_s if effective_elapsed_s else 0.0,
            "requests_per_second": len(completed) / effective_elapsed_s if effective_elapsed_s else 0.0,
            "estimated_elapsed_seconds": estimated_elapsed_s,
            "measured_elapsed_seconds": elapsed_seconds,
        },
    }
