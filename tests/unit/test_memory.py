from llm_accel.metrics.memory import GpuMemorySnapshot, summarize_memory


def test_summarize_memory_handles_unavailable_snapshots() -> None:
    before = GpuMemorySnapshot(available=False, backend="nvidia-smi", error="missing")
    after = GpuMemorySnapshot(available=False, backend="nvidia-smi", error="missing")

    summary = summarize_memory(before, after)

    assert summary["available"] is False
    assert summary["delta_used_mib"] is None


def test_summarize_memory_reports_delta() -> None:
    before = GpuMemorySnapshot(available=True, backend="nvidia-smi", used_mib=100)
    after = GpuMemorySnapshot(available=True, backend="nvidia-smi", used_mib=140)

    summary = summarize_memory(before, after)

    assert summary["available"] is True
    assert summary["delta_used_mib"] == 40
