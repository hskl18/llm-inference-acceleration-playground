from __future__ import annotations

import time

import pytest

from llm_accel.serving.openai_client import MockOpenAIClient


def test_mock_client_uses_injected_sleep_without_changing_reported_timings() -> None:
    sleeps: list[float] = []
    client = MockOpenAIClient(sleep=sleeps.append)

    started = time.perf_counter()
    result = client.complete("one two three four", max_tokens=64)
    elapsed = time.perf_counter() - started

    assert sleeps == [pytest.approx(result.total_latency_ms / 1000)]
    assert result.total_latency_ms > 100.0
    assert elapsed < 0.05


def test_mock_client_timeout_uses_injected_sleep() -> None:
    sleeps: list[float] = []
    client = MockOpenAIClient(request_timeout_seconds=0.001, sleep=sleeps.append)

    with pytest.raises(TimeoutError):
        client.complete("prompt", max_tokens=64)

    assert sleeps == [0.001]


@pytest.mark.realtime_mock
def test_mock_client_sleeps_in_real_time_by_default() -> None:
    client = MockOpenAIClient()

    started = time.perf_counter()
    result = client.complete("prompt", max_tokens=2)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert elapsed_ms >= result.total_latency_ms * 0.9
