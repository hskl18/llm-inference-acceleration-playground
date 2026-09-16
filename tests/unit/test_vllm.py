import json

import pytest

from llm_accel.serving.vllm import (
    build_vllm_command,
    normalize_vllm_quantization,
    vllm_boolean_flag,
    vllm_served_model,
    vllm_speculative_config,
)


REVISION = "a" * 40


def test_build_vllm_command_uses_the_documented_serve_entrypoint() -> None:
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

    assert argv[:3] == ["vllm", "serve", "meta-llama/example"]
    assert vllm_served_model(argv) == "meta-llama/example"
    assert "--model" not in argv
    assert argv[argv.index("--dtype") + 1] == "float16"
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--tokenizer") + 1] == "meta-llama/example-tokenizer"
    assert argv[argv.index("--tokenizer-revision") + 1] == "b" * 40
    assert argv[argv.index("--quantization") + 1] == "awq"
    assert "--gpu-memory-utilization" in argv
    assert "--enable-prefix-caching" in argv
    assert "--enable-chunked-prefill" in argv
    assert "--max-num-batched-tokens" in argv
    assert "--max-num-seqs" in argv


def test_speculative_decoding_uses_speculative_config_json() -> None:
    command = build_vllm_command(
        model="meta-llama/example",
        speculative_model="draft-model",
        num_speculative_tokens=5,
    )

    argv = command.argv()

    assert "--speculative-model" not in argv
    assert "--num-speculative-tokens" not in argv
    payload = json.loads(argv[argv.index("--speculative-config") + 1])
    assert payload == {"method": "draft_model", "model": "draft-model", "num_speculative_tokens": 5}
    assert vllm_speculative_config(argv) == payload
    assert "--speculative-config '{\"method\":\"draft_model\"" in command.shell_command()


def test_modelless_speculative_method_needs_no_draft_model() -> None:
    command = build_vllm_command(
        model="meta-llama/example",
        speculative_method="ngram",
        num_speculative_tokens=4,
    )

    assert command.speculative_config() == {"method": "ngram", "num_speculative_tokens": 4}


def test_no_speculative_settings_emit_no_speculative_flag() -> None:
    command = build_vllm_command(model="meta-llama/example")

    assert command.speculative_config() is None
    assert "--speculative-config" not in command.argv()


def test_prefix_caching_and_chunked_prefill_are_always_explicit() -> None:
    # vLLM defaults both to True, so a baseline that omits them is not a baseline.
    argv = build_vllm_command(model="meta-llama/example").argv()

    assert "--no-enable-prefix-caching" in argv
    assert "--no-enable-chunked-prefill" in argv
    assert vllm_boolean_flag(argv, "enable-prefix-caching") is False
    assert vllm_boolean_flag(argv, "enable-chunked-prefill") is False
    assert vllm_boolean_flag(argv, "enable-sleep-mode") is None


@pytest.mark.parametrize("quantization", ["int8", "int4", "not-a-method"])
def test_invalid_quantization_names_are_rejected(quantization: str) -> None:
    with pytest.raises(ValueError, match="unsupported vLLM quantization method"):
        build_vllm_command(model="meta-llama/example", quantization=quantization)


@pytest.mark.parametrize("quantization", ["none", "awq", "gptq", "fp8", "compressed-tensors", "torchao", "modelopt"])
def test_documented_quantization_names_are_accepted(quantization: str) -> None:
    assert normalize_vllm_quantization(quantization) == quantization
    assert build_vllm_command(model="meta-llama/example", quantization=quantization) is not None


def test_build_vllm_command_requires_token_count_for_speculative_decoding() -> None:
    with pytest.raises(ValueError, match="num_speculative_tokens must be a positive integer"):
        build_vllm_command(model="meta-llama/example", speculative_model="draft-model")


def test_draft_model_method_requires_a_draft_model() -> None:
    with pytest.raises(ValueError, match="speculative_model is required"):
        build_vllm_command(
            model="meta-llama/example",
            speculative_method="draft_model",
            num_speculative_tokens=4,
        )


def test_build_vllm_command_rejects_mutable_revision() -> None:
    with pytest.raises(ValueError, match="full 40 to 64 character"):
        build_vllm_command(model="meta-llama/example", revision="main")


def test_distinct_tokenizer_does_not_inherit_model_revision() -> None:
    command = build_vllm_command(
        model="meta-llama/example",
        revision=REVISION,
        tokenizer="independent/tokenizer",
    )

    assert command.tokenizer == "independent/tokenizer"
    assert command.tokenizer_revision is None
    assert "--tokenizer-revision" not in command.argv()
