from __future__ import annotations

import json

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.reports.claim_audit import audit_hardware_claim


def test_claim_audit_rejects_mock_results_even_with_enough_requests(tmp_path) -> None:
    command_file = tmp_path.parent / "server-command.txt"
    command_file.write_text(
        "python -m vllm.entrypoints.openai.api_server --model mock-model "
        f"--revision {'a' * 40} --dtype float16\n",
        encoding="utf-8",
    )
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        model_revision="a" * 40,
        concurrency=4,
        input_tokens=128,
        output_tokens=32,
        output_dir=tmp_path,
        request_count=100,
        warmup_count=5,
        dtype="float16",
        quantization="none",
        hardware_label="ci-fixture",
        optimization_profile="baseline",
        server_command_file=command_file,
    )

    report = audit_hardware_claim(tmp_path)

    assert report["publishable_hardware_claim"] is False
    assert any("vLLM endpoint" in blocker for blocker in report["blockers"])
    assert any("GPU name" in blocker for blocker in report["blockers"])
    assert any("GPU memory telemetry" in blocker for blocker in report["blockers"])


def test_claim_audit_detects_raw_request_count_mismatch(tmp_path) -> None:
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=tmp_path,
        request_count=2,
    )
    raw_path = tmp_path / "raw_requests.jsonl"
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    raw_path.write_text(rows[0] + "\n", encoding="utf-8")

    report = audit_hardware_claim(tmp_path)

    assert any("raw request count 1" in blocker for blocker in report["blockers"])
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["metrics"]["request_count"] == 2


def test_claim_audit_fails_closed_on_invalid_manifest_json(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{invalid\n", encoding="utf-8")

    report = audit_hardware_claim(tmp_path)

    assert report["publishable_hardware_claim"] is False
    assert any("could not read" in blocker for blocker in report["blockers"])


def test_claim_audit_recomputes_server_command_hash(tmp_path) -> None:
    command_file = tmp_path.parent / "server-command.txt"
    command_file.write_text(
        "python -m vllm.entrypoints.openai.api_server --model mock-model "
        f"--revision {'b' * 40} --dtype float16\n",
        encoding="utf-8",
    )
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        model_revision="b" * 40,
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=tmp_path,
        request_count=2,
        dtype="float16",
        server_command_file=command_file,
    )
    (tmp_path / "server_command.txt").write_text("tampered\n", encoding="utf-8")

    report = audit_hardware_claim(tmp_path)

    assert any("hash does not match" in blocker for blocker in report["blockers"])


def test_claim_audit_recomputes_performance_metrics_from_raw_rows(tmp_path) -> None:
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=tmp_path,
        request_count=2,
    )
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["latency_ms"]["p50"] = 0.001
    summary["metrics"]["throughput"]["measured_elapsed_seconds"] = 0.001
    summary["metrics"]["throughput"]["output_tokens_per_second"] = (
        summary["metrics"]["output_tokens"] / 0.001
    )
    summary["metrics"]["throughput"]["requests_per_second"] = (
        summary["metrics"]["completed_count"] / 0.001
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = audit_hardware_claim(tmp_path)

    assert any("latency_ms.p50 does not match" in blocker for blocker in report["blockers"])
    assert any("measured_elapsed_seconds does not match" in blocker for blocker in report["blockers"])


def test_claim_audit_binds_token_count_method_to_raw_rows(tmp_path) -> None:
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=tmp_path,
        request_count=2,
    )
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metadata"]["token_count_method"] = "tokenizers.encode(add_special_tokens=false)"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = audit_hardware_claim(tmp_path)

    assert any("token_count_method does not match raw requests" in blocker for blocker in report["blockers"])


def test_claim_audit_binds_optimization_profile_to_command_flags(tmp_path) -> None:
    command_file = tmp_path.parent / "baseline-command.txt"
    command_file.write_text(
        "python -m vllm.entrypoints.openai.api_server --model mock-model "
        f"--revision {'c' * 40} --dtype float16\n",
        encoding="utf-8",
    )
    run_latency_benchmark(
        base_url="mock://local",
        model="mock-model",
        model_revision="c" * 40,
        concurrency=1,
        input_tokens=16,
        output_tokens=8,
        output_dir=tmp_path,
        request_count=2,
        dtype="float16",
        optimization_profile="prefix-cache",
        server_command_file=command_file,
    )

    report = audit_hardware_claim(tmp_path)

    assert any("optimization_profile does not match" in blocker for blocker in report["blockers"])
