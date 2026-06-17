from llm_accel.serving.vllm import build_vllm_command


def test_build_vllm_command_includes_optional_flags() -> None:
    command = build_vllm_command(
        model="meta-llama/example",
        port=8001,
        dtype="fp16",
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        speculative_model="draft-model",
        num_speculative_tokens=4,
    )

    argv = command.argv()

    assert argv[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--quantization" in argv
    assert "awq" in argv
    assert "--gpu-memory-utilization" in argv
    assert "--enable-prefix-caching" in argv
    assert "--enable-chunked-prefill" in argv
    assert "--max-num-batched-tokens" in argv
    assert "8192" in argv
    assert "--max-num-seqs" in argv
    assert "--speculative-model" in argv
    assert "--num-speculative-tokens" in argv


def test_build_vllm_command_requires_speculative_model_for_speculative_tokens() -> None:
    try:
        build_vllm_command(model="meta-llama/example", num_speculative_tokens=4)
    except ValueError as exc:
        assert "speculative_model is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")
