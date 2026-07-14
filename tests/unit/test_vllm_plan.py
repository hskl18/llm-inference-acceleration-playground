import shlex

from llm_accel.serving.vllm_plan import create_vllm_benchmark_plan


REVISION = "a" * 40


def test_create_vllm_benchmark_plan_writes_runbook(tmp_path) -> None:
    plan = create_vllm_benchmark_plan(
        model="test-model",
        base_url="http://localhost:8000/v1",
        output_dir=tmp_path,
        revision=REVISION,
        tokenizer="test-tokenizer",
        tokenizer_revision="b" * 40,
        hardware_label="NVIDIA A100 80GB",
        dtype="float16",
        enable_prefix_caching=True,
        max_num_batched_tokens=4096,
    )

    assert len(plan["steps"]) >= 5
    step_names = [step["name"] for step in plan["steps"]]
    assert "run_throughput_benchmark" in step_names
    assert "validate_throughput_run" in step_names
    assert "results/runs/vllm-throughput/throughput_summary.json" in plan["required_artifacts"]
    assert "--enable-prefix-caching" in plan["server_command"]["argv"]
    assert REVISION in plan["server_command"]["argv"]
    assert plan["tokenizer"] == "test-tokenizer"
    assert plan["tokenizer_revision"] == "b" * 40
    assert f"--model-revision {REVISION}" in next(
        step["command"] for step in plan["steps"] if step["name"] == "run_throughput_benchmark"
    )
    assert "--request-count 128" in next(
        step["command"] for step in plan["steps"] if step["name"] == "run_throughput_benchmark"
    )
    throughput_command = next(
        step["command"] for step in plan["steps"] if step["name"] == "run_throughput_benchmark"
    )
    throughput_argv = shlex.split(throughput_command)
    assert throughput_argv[throughput_argv.index("--hardware-label") + 1] == "NVIDIA A100 80GB"
    validation_command = next(
        step["command"] for step in plan["steps"] if step["name"] == "validate_environment"
    )
    assert f"--revision {REVISION}" in validation_command
    assert "--tokenizer test-tokenizer" in validation_command
    assert f"--tokenizer-revision {'b' * 40}" in validation_command
    assert "--dtype float16" in validation_command
    assert "--enable-prefix-caching" in validation_command
    assert plan["steps"][0]["name"] == "start_vllm_server"
    assert len(plan["server_command_sha256"]) == 64
    assert "--max-num-batched-tokens" in plan["server_command"]["argv"]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "vllm_benchmark_plan.json").exists()
    assert (tmp_path / "vllm_benchmark_plan.md").exists()
    assert (tmp_path / "server_command.txt").exists()
