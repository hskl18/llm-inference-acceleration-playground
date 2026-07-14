from llm_accel.serving.vllm import build_vllm_command


REVISION = "a" * 40


def test_build_vllm_command_includes_optional_flags() -> None:
    command = build_vllm_command(
        model="meta-llama/example",
        port=8001,
        dtype="fp16",
        revision=REVISION,
        tokenizer="meta-llama/example-tokenizer",
        tokenizer_revision="b" * 40,
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
    assert argv[argv.index("--dtype") + 1] == "float16"
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--tokenizer") + 1] == "meta-llama/example-tokenizer"
    assert argv[argv.index("--tokenizer-revision") + 1] == "b" * 40
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


def test_build_vllm_command_rejects_mutable_revision() -> None:
    try:
        build_vllm_command(model="meta-llama/example", revision="main")
    except ValueError as exc:
        assert "full 40 to 64 character" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_distinct_tokenizer_does_not_inherit_model_revision() -> None:
    command = build_vllm_command(
        model="meta-llama/example",
        revision=REVISION,
        tokenizer="independent/tokenizer",
    )

    assert command.tokenizer == "independent/tokenizer"
    assert command.tokenizer_revision is None
    assert "--tokenizer-revision" not in command.argv()
