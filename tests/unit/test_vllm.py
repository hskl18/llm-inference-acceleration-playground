from llm_accel.serving.vllm import build_vllm_command


def test_build_vllm_command_includes_optional_flags() -> None:
    command = build_vllm_command(
        model="meta-llama/example",
        port=8001,
        dtype="fp16",
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    argv = command.argv()

    assert argv[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--quantization" in argv
    assert "awq" in argv
    assert "--gpu-memory-utilization" in argv
