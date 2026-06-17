from llm_accel.serving.vllm_plan import create_vllm_benchmark_plan


def test_create_vllm_benchmark_plan_writes_runbook(tmp_path) -> None:
    plan = create_vllm_benchmark_plan(
        model="test-model",
        base_url="http://localhost:8000/v1",
        output_dir=tmp_path,
        enable_prefix_caching=True,
        max_num_batched_tokens=4096,
    )

    assert len(plan["steps"]) >= 5
    step_names = [step["name"] for step in plan["steps"]]
    assert "run_throughput_benchmark" in step_names
    assert "validate_throughput_run" in step_names
    assert "results/runs/vllm-throughput/throughput_summary.json" in plan["required_artifacts"]
    assert "--enable-prefix-caching" in plan["server_command"]["argv"]
    assert "--max-num-batched-tokens" in plan["server_command"]["argv"]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "vllm_benchmark_plan.json").exists()
    assert (tmp_path / "vllm_benchmark_plan.md").exists()
