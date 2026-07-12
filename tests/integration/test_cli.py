import json
import csv
from pathlib import Path

from llm_accel.cli import main


def test_cli_doctor(capsys) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert '"status": "ok"' in output
    assert '"backend_capability"' in output
    assert '"gpu_memory"' in output


def test_cli_kv_cache_json(capsys) -> None:
    assert (
        main(
            [
                "kv-cache",
                "estimate",
                "--layers",
                "1",
                "--seq-len",
                "2",
                "--batch-size",
                "3",
                "--kv-heads",
                "4",
                "--head-dim",
                "5",
                "--dtype",
                "fp16",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"bytes": 480' in output
    assert '"explanation"' in output


def test_cli_kv_cache_preset_json(capsys) -> None:
    assert (
        main(
            [
                "kv-cache",
                "estimate",
                "--preset",
                "llama-3-8b",
                "--seq-len",
                "8",
                "--batch-size",
                "2",
                "--dtype",
                "fp16",
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"preset": "llama-3-8b"' in output
    assert '"layers": 32' in output


def test_cli_kv_cache_lists_presets(capsys) -> None:
    assert main(["kv-cache", "presets"]) == 0
    output = capsys.readouterr().out
    assert '"llama-3-8b"' in output


def test_cli_kv_cache_requires_shape_without_preset(capsys) -> None:
    assert main(["kv-cache", "estimate", "--seq-len", "8", "--batch-size", "2", "--dtype", "fp16"]) == 2
    output = capsys.readouterr().err
    assert "--layers" in output


def test_cli_latency_benchmark_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "bench"

    assert (
        main(
            [
                "bench",
                "latency",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--concurrency",
                "2",
                "--input-tokens",
                "16",
                "--output-tokens",
                "8",
                "--request-count",
                "3",
                "--hardware-label",
                "test-host",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "raw_requests.jsonl").exists()
    assert (output_dir / "raw_requests.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "resolved_config.json").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "plots" / "latency.svg").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["throughput"]["measured_elapsed_seconds"] is not None
    assert "timeout_count" in summary["metrics"]
    assert summary["metadata"]["api_kind"] == "chat"
    assert summary["metadata"]["backend_version"]
    assert summary["metadata"]["hardware_label"] == "test-host"
    assert summary["metadata"]["python_version"]
    assert summary["metadata"]["operating_system"]
    assert "git_commit" in summary["metadata"]
    assert "gpu_name" in summary["metadata"]
    assert "gpu_driver_version" in summary["metadata"]
    assert "cuda_version" in summary["metadata"]
    assert "torch_version" in summary["metadata"]
    assert summary["warnings"]
    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Hardware label" in summary_md
    assert "API kind" in summary_md
    assert "Backend version" in summary_md
    assert "Warnings" in summary_md
    rows = list(csv.DictReader((output_dir / "raw_requests.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert rows[0]["request_id"] == "req-000001"
    assert rows[0]["completed"] == "True"
    assert float(rows[0]["completed_offset_ms"]) >= float(rows[0]["started_offset_ms"])


def test_cli_claim_audit_rejects_missing_hardware_artifacts(tmp_path: Path, capsys) -> None:
    assert main(["report", "claim-audit", "--run-dir", str(tmp_path)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["publishable_hardware_claim"] is False
    assert "summary.json is required" in payload["blockers"]


def test_cli_latency_benchmark_defaults_to_timestamped_results_dir(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["bench", "latency", "--base-url", "mock://local", "--request-count", "1"]) == 0

    payload = json.loads(capsys.readouterr().out)
    output_dir = Path(payload["output_dir"])
    assert output_dir.parts[0:2] == ("results", "runs")
    assert output_dir.name.endswith("-latency")
    assert (tmp_path / output_dir / "summary.json").exists()
    assert (tmp_path / output_dir / "raw_requests.csv").exists()


def test_cli_latency_benchmark_accepts_completion_api_kind(tmp_path: Path) -> None:
    output_dir = tmp_path / "completion-bench"

    assert (
        main(
            [
                "bench",
                "latency",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--api-kind",
                "completion",
                "--request-count",
                "2",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["api_kind"] == "completion"
    assert resolved["api_kind"] == "completion"


def test_cli_examples_write_creates_runnable_configs(tmp_path: Path, capsys, monkeypatch) -> None:
    examples_dir = tmp_path / "examples"
    output_dir = tmp_path / "example-sweep"
    prompt_output_dir = tmp_path / "example-prompt-sweep"
    prefix_output_dir = tmp_path / "example-prefix-sweep"

    assert main(["examples", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert "benchmark_small.yaml" in listed["examples"]
    assert "spec_prompts.jsonl" in listed["examples"]
    assert "benchmark_prefix_cache.yaml" in listed["examples"]

    assert main(["examples", "write", "--output-dir", str(examples_dir)]) == 0
    written = json.loads(capsys.readouterr().out)
    assert str(examples_dir / "benchmark_prompts.yaml") in written["files"]
    assert (examples_dir / "benchmark_small.yaml").exists()
    assert (examples_dir / "spec_prompts.jsonl").exists()
    assert (examples_dir / "prefix_cache_prompts.jsonl").exists()

    monkeypatch.chdir(tmp_path)
    assert main(["bench", "sweep", "--config", str(examples_dir / "benchmark_small.yaml"), "--output-dir", str(output_dir)]) == 0
    assert (output_dir / "aggregate_summary.json").exists()
    assert main(["bench", "sweep", "--config", str(examples_dir / "benchmark_prompts.yaml"), "--output-dir", str(prompt_output_dir)]) == 0
    assert (prompt_output_dir / "c1-prompts-out64" / "summary.json").exists()
    assert main(["bench", "sweep", "--config", str(examples_dir / "benchmark_prefix_cache.yaml"), "--output-dir", str(prefix_output_dir)]) == 0
    prefix_summary = json.loads((prefix_output_dir / "c1-prompts-out64" / "summary.json").read_text(encoding="utf-8"))
    assert prefix_summary["metadata"]["shared_prefix_tokens_estimate"] > 0
    assert main(["examples", "write", "--output-dir", str(examples_dir)]) == 2
    assert "pass --overwrite" in capsys.readouterr().err


def test_cli_latency_benchmark_accepts_fixed_prompt_file(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "fixed-prompts"
    prompts.write_text('{"prompt": "short prompt"}\n{"prompt": "a much longer fixed benchmark prompt"}\n', encoding="utf-8")

    assert (
        main(
            [
                "bench",
                "latency",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--prompts",
                str(prompts),
                "--request-count",
                "3",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "raw_requests.csv").open(encoding="utf-8")))
    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert summary["metadata"]["workload_mode"] == "fixed_prompts"
    assert summary["metadata"]["prompt_count"] == 2
    assert summary["metadata"]["workload_fingerprint"]
    assert summary["metadata"]["shared_prefix_tokens_estimate"] == 0
    assert summary["metadata"]["shared_prefix_fingerprint"] is None
    assert resolved["workload_mode"] == "fixed_prompts"
    assert resolved["prompt_count"] == 2
    assert resolved["shared_prefix_tokens_estimate"] == 0
    assert "Shared prefix tokens estimate" in summary_md
    assert "short prompt" not in (output_dir / "resolved_config.json").read_text(encoding="utf-8")
    assert [row["input_tokens"] for row in rows] == ["2", "6", "2"]


def test_cli_throughput_benchmark_writes_throughput_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "throughput"

    assert (
        main(
            [
                "bench",
                "throughput",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--concurrency",
                "2",
                "--input-tokens",
                "16",
                "--output-tokens",
                "8",
                "--request-count",
                "3",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    throughput = json.loads((output_dir / "throughput_summary.json").read_text(encoding="utf-8"))
    assert manifest["run_type"] == "throughput_benchmark"
    assert "throughput_summary.json" in manifest["artifacts"]
    assert throughput["throughput"]["output_tokens_per_second"] > 0
    assert throughput["completed_count"] == 3
    assert (output_dir / "throughput_summary.md").exists()
    assert (output_dir / "raw_requests.jsonl").exists()


def test_cli_latency_benchmark_warns_for_non_streaming(tmp_path: Path) -> None:
    output_dir = tmp_path / "non-streaming"

    assert (
        main(
            [
                "bench",
                "latency",
                "--base-url",
                "mock://local",
                "--request-count",
                "2",
                "--no-stream",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert any("Non-streaming mode cannot observe TTFT" in warning for warning in summary["warnings"])


def test_cli_sweep_writes_aggregate(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"

    assert main(["bench", "sweep", "--config", "configs/benchmark_small.yaml", "--output-dir", str(output_dir)]) == 0

    assert (output_dir / "aggregate_summary.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "resolved_config.json").exists()
    assert (output_dir / "aggregate_summary.md").exists()
    assert (output_dir / "plots" / "sweep_throughput.svg").exists()
    assert (output_dir / "plots" / "latency_throughput.svg").exists()
    assert (output_dir / "c1-in128-out64" / "summary.json").exists()
    assert (output_dir / "c4-in512-out64" / "summary.md").exists()
    summary = json.loads((output_dir / "c1-in128-out64" / "summary.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["hardware_label"] == "local-dev"
    assert summary["metadata"]["api_kind"] == "chat"


def test_cli_sweep_rejects_invalid_config(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  measured_requests: 0",
                "endpoint:",
                "  base_url: mock://local",
                "  api_key: inline-secret",
                "model:",
                "  name: mock-model",
                "workload:",
                "  input_tokens: [128]",
                "  output_tokens: [64]",
                "  concurrency: [0]",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["bench", "sweep", "--config", str(config_path)]) == 2
    captured = capsys.readouterr()
    assert "invalid benchmark config" in captured.err
    assert "endpoint.api_key must not contain an inline secret" in captured.err


def test_cli_sweep_redacts_remote_endpoint_in_resolved_config(tmp_path: Path) -> None:
    config_path = tmp_path / "remote.yaml"
    output_dir = tmp_path / "sweep"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: remote-sweep",
                "  measured_requests: 1",
                "endpoint:",
                "  base_url: https://api.example.com/v1",
                "  api_key_env: OPENAI_API_KEY",
                "  backend: mock",
                "model:",
                "  name: mock-model",
                "workload:",
                "  input_tokens: [16]",
                "  output_tokens: [8]",
                "  concurrency: [1]",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["bench", "sweep", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0

    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    child_summary = json.loads((output_dir / "c1-in16-out8" / "summary.json").read_text(encoding="utf-8"))
    assert resolved["endpoint"]["base_url"] == "redacted"
    assert resolved["endpoint"]["api_key_env"] == "OPENAI_API_KEY"
    assert child_summary["metadata"]["base_url"] == "redacted"


def test_cli_sweep_accepts_prompt_workload(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    config_path = tmp_path / "prompt-sweep.yaml"
    output_dir = tmp_path / "sweep"
    prompts.write_text('{"prompt": "fixed prompt one"}\n{"prompt": "fixed prompt two"}\n', encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: prompt-sweep",
                "  measured_requests: 2",
                "endpoint:",
                "  base_url: mock://local",
                "  backend: mock",
                "model:",
                "  name: mock-model",
                "workload:",
                f"  prompts_path: {prompts}",
                "  output_tokens: [8]",
                "  concurrency: [1]",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["bench", "sweep", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0

    summary = json.loads((output_dir / "c1-prompts-out8" / "summary.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["workload_mode"] == "fixed_prompts"
    assert summary["metadata"]["prompt_count"] == 2


def test_cli_backend_list(capsys) -> None:
    assert main(["backend", "list"]) == 0
    output = capsys.readouterr().out
    assert '"vllm"' in output
    assert '"quantization_modes"' in output


def test_cli_backend_profile(capsys) -> None:
    assert main(["backend", "profile", "--backend", "vllm", "--base-url", "http://localhost:8000/v1"]) == 0
    output = capsys.readouterr().out
    assert '"backend_version"' in output
    assert '"adapter_status": "implemented"' in output
    assert "OpenAICompatibleClient" in output


def test_cli_quantization_compare(tmp_path: Path) -> None:
    output_dir = tmp_path / "quant"

    assert (
        main(
            [
                "quantization",
                "compare",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--modes",
                "none,int8",
                "--request-count",
                "2",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "quantization_comparison.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "quantization_comparison.md").exists()
    assert (output_dir / "none" / "summary.json").exists()
    assert (output_dir / "int8" / "plots" / "latency.svg").exists()
    report = json.loads((output_dir / "quantization_comparison.json").read_text(encoding="utf-8"))
    assert report["runs"][0]["quality_sanity"]["passed"] is True
    assert report["runs"][0]["support_status"] == "supported"
    assert report["runs"][0]["measured"] is True


def test_cli_quantization_compare_reports_unsupported_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "quant-unsupported"

    assert (
        main(
            [
                "quantization",
                "compare",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--modes",
                "none,fp8",
                "--request-count",
                "1",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    report = json.loads((output_dir / "quantization_comparison.json").read_text(encoding="utf-8"))
    assert report["runs"][1]["quantization"] == "fp8"
    assert report["runs"][1]["support_status"] == "unsupported"
    assert report["runs"][1]["measured"] is False
    assert not (output_dir / "fp8" / "summary.json").exists()


def test_cli_vllm_command(capsys) -> None:
    assert main(["vllm", "command", "--model", "test-model", "--port", "8001", "--dtype", "auto"]) == 0
    output = capsys.readouterr().out
    assert "vllm.entrypoints.openai.api_server" in output
    assert "--model test-model" in output


def test_cli_vllm_validate_writes_blocker_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "vllm"

    assert (
        main(
            [
                "vllm",
                "validate",
                "--model",
                "test-model",
                "--revision",
                "a" * 40,
                "--base-url",
                "mock://local",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 1
    )

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "vllm_validation.json").exists()
    assert (output_dir / "vllm_validation.md").exists()
    report = json.loads((output_dir / "vllm_validation.json").read_text(encoding="utf-8"))
    assert "ready_for_hardware_benchmark" in report
    assert isinstance(report["blockers"], list)


def test_cli_vllm_plan_writes_runbook(tmp_path: Path) -> None:
    output_dir = tmp_path / "vllm-plan"

    assert (
        main(
            [
                "vllm",
                "plan",
                "--model",
                "test-model",
                "--revision",
                "a" * 40,
                "--hardware-label",
                "a100-80gb",
                "--dtype",
                "float16",
                "--base-url",
                "http://localhost:8000/v1",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "vllm_benchmark_plan.json").exists()
    assert (output_dir / "vllm_benchmark_plan.md").exists()
    assert (output_dir / "server_command.txt").exists()


def test_cli_speculative_outputs_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "spec"

    assert main(["speculative", "run", "--lookahead", "4", "--output-dir", str(output_dir)]) == 0

    assert (output_dir / "speculative_summary.json").exists()
    assert (output_dir / "speculative_summary.md").exists()
    assert (output_dir / "acceptance_curve.json").exists()
    assert (output_dir / "baseline_comparison.json").exists()
    assert (output_dir / "baseline_comparison.md").exists()
    assert (output_dir / "manifest.json").exists()
    report = json.loads((output_dir / "speculative_summary.json").read_text(encoding="utf-8"))
    assert "baseline_comparison" in report


def test_cli_eval_sanity(tmp_path: Path) -> None:
    output_dir = tmp_path / "eval"

    assert (
        main(
            [
                "eval",
                "sanity",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--prompts",
                "configs/spec_prompts.jsonl",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "quality_eval.json").exists()
    assert (output_dir / "quality_eval.md").exists()
    report = json.loads((output_dir / "quality_eval.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_cli_eval_task(tmp_path: Path) -> None:
    output_dir = tmp_path / "task"

    assert (
        main(
            [
                "eval",
                "task",
                "--base-url",
                "mock://local",
                "--model",
                "mock-model",
                "--tasks",
                "configs/task_eval_small.jsonl",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "task_eval.json").exists()
    assert (output_dir / "task_eval.md").exists()
    report = json.loads((output_dir / "task_eval.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_cli_report_validate_and_compare(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    comparison = tmp_path / "comparison"

    assert main(["bench", "latency", "--base-url", "mock://local", "--request-count", "2", "--output-dir", str(run_a)]) == 0
    assert main(["bench", "latency", "--base-url", "mock://local", "--request-count", "2", "--concurrency", "2", "--output-dir", str(run_b)]) == 0
    assert main(["report", "validate", "--run-dir", str(run_a)]) == 0
    assert (
        main(
            [
                "report",
                "compare",
                "--summary",
                str(run_a / "summary.json"),
                "--summary",
                str(run_b / "summary.json"),
                "--output-dir",
                str(comparison),
            ]
        )
        == 0
    )

    assert (comparison / "manifest.json").exists()
    assert (comparison / "comparison.json").exists()
    assert (comparison / "comparison.md").exists()
    report = json.loads((comparison / "comparison.json").read_text(encoding="utf-8"))
    assert report["ranking_allowed"] is False
    assert report["warnings"]


def test_cli_report_generate_regenerates_run_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "bench"

    assert main(["bench", "latency", "--base-url", "mock://local", "--request-count", "2", "--output-dir", str(output_dir)]) == 0
    (output_dir / "summary.md").unlink()
    (output_dir / "plots" / "latency.svg").unlink()

    assert main(["report", "generate", "--run-dir", str(output_dir)]) == 0

    assert (output_dir / "summary.md").exists()
    assert (output_dir / "plots" / "latency.svg").exists()
    assert main(["report", "validate", "--run-dir", str(output_dir)]) == 0


def test_cli_report_generate_supports_summary_to_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "bench"
    output = tmp_path / "report.md"

    assert main(["bench", "latency", "--base-url", "mock://local", "--request-count", "2", "--output-dir", str(run_dir)]) == 0
    assert main(["report", "generate", "--summary", str(run_dir / "summary.json"), "--output", str(output)]) == 0

    assert output.exists()
    assert "Benchmark Summary" in output.read_text(encoding="utf-8")
