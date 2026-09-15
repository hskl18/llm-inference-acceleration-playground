from __future__ import annotations

import json

import pytest

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.workloads.prompts import shared_prefix_tokens
from llm_accel.workloads.synthetic import prompt_batch, synthetic_prompt


def test_prompt_batch_is_unique_and_shares_no_prefix() -> None:
    batch = prompt_batch(32, 128, 42)

    assert len(set(batch)) == 32
    assert shared_prefix_tokens(batch) == 0
    assert all(len(prompt.split()) == 128 for prompt in batch)


def test_prompt_batch_is_deterministic_across_calls_and_seeds() -> None:
    assert prompt_batch(8, 16, 42) == prompt_batch(8, 16, 42)
    assert prompt_batch(8, 16, 42) != prompt_batch(8, 16, 43)
    assert prompt_batch(4, 16, 42) == prompt_batch(8, 16, 42)[:4]


def test_warmup_indices_do_not_reuse_measured_prompts() -> None:
    measured = prompt_batch(8, 16, 42)
    warmup = prompt_batch(4, 16, 42, first_index=8)

    assert not set(measured) & set(warmup)


def test_synthetic_prompt_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="input_tokens must be positive"):
        synthetic_prompt(0)
    with pytest.raises(ValueError, match="count must be positive"):
        prompt_batch(0, 8)
    with pytest.raises(ValueError, match="first_index must be non-negative"):
        prompt_batch(1, 8, first_index=-1)


def test_synthetic_benchmark_records_prompt_uniqueness(tmp_path) -> None:
    summary = run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=32,
        output_tokens=2,
        output_dir=tmp_path / "synthetic",
        request_count=8,
        warmup_count=2,
    )

    metadata = summary["metadata"]
    assert metadata["unique_prompt_count"] == 8
    assert metadata["shared_prefix_tokens_estimate"] == 0
    assert metadata["workload_fingerprint"]
    assert not any("distinct" in warning for warning in summary["warnings"])
    resolved = json.loads((tmp_path / "synthetic" / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["unique_prompt_count"] == 8


def test_repeated_prompts_are_recorded_and_warned(tmp_path) -> None:
    summary = run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=32,
        output_tokens=2,
        output_dir=tmp_path / "repeats",
        request_count=6,
        prompt_texts=["first prompt", "second prompt"],
    )

    assert summary["metadata"]["unique_prompt_count"] == 2
    assert any(
        "Only 2 of 6 measured prompts are distinct" in warning for warning in summary["warnings"]
    )
