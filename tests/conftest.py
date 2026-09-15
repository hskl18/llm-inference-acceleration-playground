from __future__ import annotations

import pytest

from llm_accel.serving.openai_client import MockOpenAIClient


@pytest.fixture(autouse=True)
def _fast_mock_client(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real waiting in the mock backend unless a test needs wall-clock timing.

    Reported mock latencies are unchanged, but requests in one client worker can overlap
    in the recorded timeline. Mark tests that depend on real queueing with realtime_mock.
    Spawned client processes do not inherit this patch and still sleep in real time.
    """
    if request.node.get_closest_marker("realtime_mock") is None:
        monkeypatch.setattr(MockOpenAIClient, "sleep", staticmethod(lambda _seconds: None))
